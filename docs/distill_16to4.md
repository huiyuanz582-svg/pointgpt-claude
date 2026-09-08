# 固定 16-step Teacher → 4-step Student

独立入口 `tools/runner_distill.py`。原 `runner_finetune.py`、backbone 和数据集代码不变。
Teacher/Student 都使用完整 PointGPT-L，Teacher checkpoint 严格加载后直接复制给 Student，
保留已训练的 epsilon 输出头；不经过原 fine-tune 的输出头重初始化。

## 训练

在仓库根目录、原 Linux/CUDA 训练环境运行。将路径替换为实际最佳 **去噪** checkpoint，
不能使用 `pretrainModel/` 中的生成式预训练 checkpoint。当前工作区未提供最佳权重。

```bash
python tools/runner_distill.py --mode train \
  --config cfgs/PointGPT-L/distill_16to4.yaml \
  --teacher_ckpt /path/to/PointGPT-L/ckpt-best.pth \
  --output_dir experiments/distill_16to4/train
```

冒烟可附加 `--epochs 1 --max_patch_batches 1`（一个 baseline DataLoader batch），
并使用不同输出目录。`--max_shapes` 在训练时限制 patch 样本数，测试时限制整云数。
每 epoch 保存 `ckpt-last.pth`（包含优化器状态和固定日程）、`train.jsonl`，
`manifest.json` 记录配置、checkpoint 路径和命令参数。不按训练损失冒充验证最优，
不做自动长训练、调参或下一阶段训练。已有非空输出目录会被拒绝。

训练直接调用原 `ScoreDenoise.train_dataloader()`，使用 `PairedPatchDataset` 的
train split、验证 shape 排除规则、三种分辨率和 `TRAIN_OVERSAMPLE=50`。
每次访问先将 clean 整云单位球归一化、在线加 Gaussian noise（sigma 对数均匀采样），
再用 noisy 中随机种子点的 KNN 同索引取 noisy/clean 1024-point patch，不对 patch 另行归一化。
`total_bs=8` 与 PointGPT-L baseline 一致，并保留原 shuffle、drop_last、worker 和 collate。
每 epoch 有效 patch 数为 `floor(整云条目数 * 50 / 8) * 8`；120 条目时为 6000，
与 baseline 相同。四个蒸馏 stage 是同一 patch 的四份监督，不计作四倍独立样本。
默认 20 epochs、AdamW lr=1e-5、weight_decay=0。

Teacher 永久 `eval()`、`requires_grad_(False)`，`capture_teacher()` 在 `torch.no_grad()`
内直接对已采出的同一个 patch 做 16 步；更新公式逐式沿用原 `patch_based_denoise` 的
patch 内循环（eta=0.3、decay=0.95），不再做外层 FPS/KNN 切块、重排或融合。
模型内部原有 group 操作不变。保存 `[0,4,8,12,16]` 五个节点并 detach 到 CPU，
T0 与 DataLoader noisy patch 的坐标和点顺序完全一致；每个 patch 使用自己的 sigma0。
四个 stage 使用 Teacher 起点，互相不喂 Student 预测：

```
sigma_j = sigma0 * 0.95 ** [0,4,8,12][j]
S_next = X_teacher_start + sigma_j * eps_student
L_j = mean_points ||S_next - detach(T_target)||²
L_traj = (L_0 + L_1 + L_2 + L_3) / 4
```

`student.train()` 下调用 `forward(type='val')`：这里只选择返回坐标的分支，
不关闭 autograd；epsilon head 完全保留。每个 stage 单独 backward；Student micro-batch
按样本数加权累积到有效 batch=8 后才 optimizer.step，四 stage loss 定义不变。
`teacher_patch_batch`/`student_patch_batch` 仅控制显存。默认 Student micro-batch=1 的
BatchNorm 统计与 baseline batch=8 不同；显存允许时可设 `student_patch_batch: 8`。
抽样规则和样本数一致，不保证同 seed 的整个训练轨迹逐样本相同（模型操作会消耗随机数）。
运行时检查 Teacher 没有梯度、Student 有有限梯度，并打印首次检查结果。
不计算 clean epsilon、P2M、Chamfer 或任何其他监督 loss；不做 curriculum、corrective
rollout 或 time/stage condition。原 backbone 构造器/模块导入仍需要原 CUDA 扩展（包括
Chamfer/PyTorch3D 的历史依赖），但这些几何损失不会被调用。

## 测试

```bash
python tools/runner_distill.py --mode test \
  --config cfgs/PointGPT-L/distill_16to4.yaml \
  --student_ckpt experiments/distill_16to4/train/ckpt-last.pth \
  --output_dir experiments/distill_16to4/test --save_trajectory
```

测试使用原 `ScoreDenoise.test_dataloader()` 的归一化约定和 TEST_NOISE fallback，
不加载 Teacher。原 `patch_based_denoise` 配置为 `num_steps=4, step_size=1.0,
decay=0.95**4`，每个固定外层 patch 内真实连续运行 S0→S1→S2→S3→S4，
sigma 分别为 sigma0×0.95^[0,4,8,12]，四步后才融合整云。模型内部每步仍重新 FPS/KNN/group。

输出 raw 去噪世界坐标 `.xyz`；不做 SOR/表面投影或 CD/P2M 评估。
`--save_trajectory` 保存 S0..S4 的 patch/global 状态、索引、融合权重和 sigma 到 NPZ。
NPZ 状态在归一化空间，用其 `center` / `scale` 还原世界坐标。

## 检查

```bash
python -m unittest discover -s tests -p test_distill_16to4.py -v
```

CPU 测试使用真实 PyTorch autograd 和原 `patch_based_denoise` 函数 AST，
仅替换 CUDA FPS 与网络为 CPU 测试模型；验证冻结、节点、Teacher forcing、target detach、
四 stage 梯度等权累积、真实连续推理和 sigma 日程。另外执行原数据类及 transform 的
CPU 测试，检查 split 排除、50 倍过采样、相同随机状态下 baseline/蒸馏样本完全一致，
以及每个 patch 的 sigma 和不等长 micro-batch 梯度权重。它不替代真实 PointGPT-L/checkpoint
在原 CUDA 环境的训练/测试冒烟。
