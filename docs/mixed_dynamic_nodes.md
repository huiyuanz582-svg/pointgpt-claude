# 原 Mixed + 动态节点

分支 `paper2-mixed-dynamic-nodes` 的工作树以第一版 Mixed `cae84e8` 为基础，
保留原 canonical Teacher 节点目标和输入来源调度。此前 noise-anchor 与
matched-target 已撤回；本次不增加噪声条件。

## 关于 20 epoch

原 Mixed20 的 best 为 epoch11，10k/1% 验证 Score 为 2.157374。
epoch15、18、19 分别为 2.159548、2.158996、2.158913，epoch20 为 2.169685。
后期没有持续下降，不能认定 20 epoch 明显不足，也不能由这组 1% 验证证明
3% 已收敛。相对截图基础蒸馏，原 Mixed 在 10k/3% 的 Score 好约 1.70%，
在 50k/3% 差约 4.18%。训练时长是否影响后者，现有结果无法直接回答。

本次默认仍为 20 epoch，先比较固定/动态节点。`EPOCHS=40` 可以另建一个
40-epoch run，但同时改变训练预算和节点策略时，不能将差异都归因于动态节点。
Mixed resume 尚不支持；扩展总轮数必须在新 run 开始前确定，不会自动续跑。

## 训练语义

专用配置为 `cfgs/PointGPT-L/distill_16to4_input_scheduled_dynamic.yaml`。
直接组合已有 `runner_distill.train`、`backward_student_input_stages` 和
`update_dynamic_curriculum`，没有复制训练循环或新增节点选择公式。

- 初始节点 `[0,4,8,12,16]`；始终为 16 步 Teacher 到 4 步 Student。
- 每个 epoch 内只有一套全局节点，训练、TF 验证和整云 rollout 使用同一套。
- 第 1 阶段始终输入 T0。第 2～4 阶段每 patch 独立取 Teacher/Student 状态。
  q 仍为 epochs1–3=1、4=.84、5=.68、6=.52、7=.36、8 以后=.2。
- 不等长区间使用 `sigma0 * .95**start` 和 `[start,target,gap]/16` 条件。
  Student-source 输入取上一阶段实际输出 detach，监督仍为 canonical T[target]。
  四个平方 L2 loss 等权、逐阶段 backward，每 batch 一个 optimizer step。
- 每个 epoch 先训练并验证，再搜索 next nodes，最后保存；新节点从下一轮生效。

本配置显式选择 `curriculum_metric.type: original_pcd`，搜索时 Student 输入为
Teacher 区间起点。它衡量局部 Teacher 区间拟合难度，不等于 Mixed 训练误差或
连续 rollout 的误差；训练来源计数和搜索指标分别记录。
本轮不使用 rollout-aware / Pilot-A 的评分或校准配置。

## 搜索与开销

固定从训练 Dataset 独立采样 64 个 calibration patch，seed2025，保留原始
0.5%–2% loguniform 加噪与 noisy-KNN 切片。缓存冻结 Teacher 的 T0..T16。
校准准备与搜索恢复随机数状态；校准数据不进入 optimizer，不使用 3% 测试集。

每轮枚举全部 455 条合法四区间路径。先逐 patch 计算
`PCD=E_imit/(D_move+1e-12)`，再对 patch 平均。选择目标为：

```text
mean(abs(stage_PCD - 0.3)) + 1.0 * population_std(stage_PCD)
```

每轮重评 130 个区间；64 patch、batch8 对应 1040 次 Student forward，
共 8320 个 patch/interval 评估。训练本身仍是每 microbatch 四次 Student forward。
真实耗时和显存尚未实测。节点允许大幅改变，也可能多轮保持不变；没有 EMA、
移动约束或提升保证。较低 PCD 搜索目标不代表更低的 clean CD/P2M。

## 对照、日志与测试节点

模型、数据、噪声分布、q、优化器、20-epoch 预算、SOR=false、projection=false
均沿用原 Mixed。专用配置 `rollout_val_interval: 1` 对齐实际 Mixed20 run；
旧固定 Mixed 模板仍保留其历史值 5，不能直接把该模板当成历史有效配置。
当前下一轮消融将验证条件由 10k/1% 改为 10k/3%，每 epoch 按该条件的
`CD+0.3*P2M` 选择 best。共享 `cfgs/dataset_configs/ScoreDenoise.yaml` 中
`VAL_NOISE: 0.030` 与 `VAL_NOISE_LEVELS: [0.03]` 同步修改；后者优先。
所有引用该共享数据配置的入口均会采用新的验证噪声。TF 验证 patch 与整云
rollout 验证都由同一配置生成 3% 噪声；训练和动态节点校准仍采样 0.5%–2%，
六条件测试仍由各自的 TEST_* 配置决定。

历史动态节点实验在 10k/1% 验证下 best 为 epoch11。本轮只改变验证噪声，
用于观察 best epoch 是否改变；实际轮次需由后续运行的 `train.jsonl` 确认，
目前尚未运行。后续先核对 `train/manifest.json` 的验证噪声配置，以及
`validation_manifest.json` 中每个样本的 `sigma0` 是否为 0.03，再比较
`train.jsonl` 的最终 `best_epoch` 与历史的 11；两种噪声下的验证 Score
不直接作优劣比较。修改配置不会重新评估历史权重；现有保存逻辑仅保留
best/last，没有全部 epoch 权重时不能事后重排原 20 轮。

验证继续使用 `VAL_NUM=0` 的 test 形状身份和固定生成噪声，区别于磁盘上
六条件测试使用的噪声文件；这次属于选模协议消融，不是独立留出验证。

`train.jsonl` 包含输入来源计数、使用节点、stage gaps、下一轮节点、搜索 PCD、
目标值、forward 数和分阶段耗时。Mixed 的训练 `mean_PCD` 保持 null，不能把
混合训练误差当成搜索 PCD。`calibration_manifest.json` 保存校准样本与数据指纹。

last/best 的 `distillation.teacher_nodes`、`nodes_used_this_epoch` 和
`current_teacher_nodes` 绑定该 checkpoint 权重实际训练和验证使用的节点。
`next_teacher_nodes` 只描述下一轮计划。现有 `--mode test` 从 checkpoint 读取
实际节点，并用正确 start-step sigma 做连续四步推理，不在测试集搜索节点。
测试无需 Teacher 模型，也不新增噪声特征；仍沿用原测试接口。

六条件测试配置应从该 run 的 `train/manifest.json` 继承，再切换测试分辨率、
噪声和数据路径；保留 `curriculum_mode: dynamic_pcd`。不能把 YAML 的初始节点
或 last 的 next nodes 当作 best 的推理节点。之后先读训练 JSONL、训练 manifest、
validation manifest，再核对测试完整性和来源。

## 手动运行

在原服务器容器内、激活 `pointgpt` 环境后，从 `/workspace` 执行：

```bash
cd /workspace
conda activate pointgpt
bash scripts/train_mixed_dynamic_nodes.sh
```

默认 Teacher 为 `experiments/L_consistency_plus/ckpt-best.pth`，seed0、device0、
20 epoch。每次创建新的 `experiments/input_ablation/mixed_dynamic_nodes_seed0_*`，
训练在其 `train/` 子目录，拒绝已有 run 目录。不会自动运行后续测试。
`config.yaml` 记录实际总轮数；`source_manifest.json` 记录 branch、HEAD、dirty
状态、关键源码 SHA256、实际命令和完成状态。尚未提交的源码也能通过哈希追溯。

如用户另行选择 40 epoch，从同一 Teacher 新建独立实验：

```bash
EPOCHS=40 bash scripts/train_mixed_dynamic_nodes.sh
```

`TEACHER_CKPT`、`RUN_ROOT`、`SEED`、`DEVICE`、`PYTHON_BIN` 可显式覆盖。
device 是进程可见的 CUDA 设备序号；设置 CUDA_VISIBLE_DEVICES 时对应其映射。

## 本次验证范围

新增 `tests/test_training_input_dynamic.py` 覆盖动态节点与 Mixed 的组合：
配置对照、跨轮节点传递、canonical 目标、mask/RNG 独立、验证节点、best/last
节点绑定、推理加载与 resume 拒绝。用例仅供将来获准后手动执行，尚未运行。
本次只做 Python/YAML/嵌入 Python/Bash 静态检查和 Git diff 检查；不启动训练、
CPU 单测、模型诊断、GPU 程序或 Pilot-A，不提交代码。
