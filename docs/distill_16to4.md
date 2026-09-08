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
只保存两份 checkpoint：每 epoch 更新 `ckpt-last.pth`（包含优化器状态，可续训），
仅在验证指标改善时更新 `ckpt-best.pth`，不再保存逐 epoch 的独立权重文件。
`train.jsonl` 每 epoch 记录训练 loss、验证 loss、best epoch；`manifest.json` 记录运行参数。
best 文件只保存模型权重及元数据，不重复保存优化器状态。非空输出目录仍拒绝覆盖。

best 的标准是**固定验证 patch 上四阶段 teacher-forced trajectory loss 的均值**，越小越好；
不是训练 loss，也不代表 CD/P2M 或连续四步去噪质量一定最好。
验证沿用原 `ScoreDenoise.val_dataloader()`：`VAL_NUM=0` 时仍使用原 test split，
`VAL_NUM>0` 时使用已从训练中排除的 shape；没有新增训练划分。
验证噪声固定，每个整云以 `validation_seed=2024` 固定抽取 4 个 patch，Teacher targets
只生成一次并缓存到 CPU；`validation_manifest.json` 记录样本和种子点。
Student 验证使用 eval/no_grad，结束后恢复训练模式，不更新 BN 或梯度；不修改训练 loss。

已启动的旧进程不会自动应用保存逻辑。等它生成 `ckpt-last.pth` 后，可在更新代码后续跑：

```bash
python tools/runner_distill.py --mode train \
  --config cfgs/PointGPT-L/distill_16to4.yaml \
  --teacher_ckpt /path/to/PointGPT-L/ckpt-best.pth \
  --resume experiments/distill_16to4/train/ckpt-last.pth \
  --output_dir experiments/distill_16to4/train_resumed
```

续跑恢复 Student、optimizer 和 epoch（`--epochs` 指目标总 epoch 数），Teacher 仍加载原
去噪 checkpoint。新目录会先保留并验证恢复时的权重，再与后续 epoch 比较。
旧进程已覆盖的历史权重无法恢复；该续跑不是随机数状态逐位复现。

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
  --student_ckpt experiments/distill_16to4/train/ckpt-best.pth \
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
