# 混合 / 连续 Student 输入

分支：`paper2-scheduled-rollout-input`，从 `paper2-rollout-aware-curriculum`
的 `e472b7b` 创建。本次实验只改变训练输入来源，先固定
`[0, 4, 8, 12, 16]` + Step Condition，评估是否减轻 Teacher-forced
训练与 Student 连续推理的输入差异。效果尚待实验验证。

## 输入与监督

设 Teacher 节点为 `T[n0]..T[n4]`。第 1 阶段总从 `T0` 出发，之后每个
patch、每个阶段独立选择输入。这里的“混合”是选择整个点云 patch 的来源，
不是对 Teacher/Student 坐标加权平均。

| `training_input.type` | 第 2～4 阶段输入 |
| --- | --- |
| `teacher_forced`（默认） | `T[n(k-1)]`，调用原 `backward_stages` |
| `scheduled_rollout` | 概率 q 取 Teacher 节点，否则取上一阶段实际输出 `S[k-1].detach()` |
| `student_rollout` | 始终取上一阶段输出 `S[k-1].detach()`，即 q=0 |

混合模式允许中途重新回到 Teacher 节点；下一步若再选择 Student，使用的就是
这次实际输入所产生的输出。纯 Student 模式四步连续，不在中途重置。

所有模式仍以 `T[nk]` 为该阶段监督目标，计算对应点的平方 L2：
先对 xyz 求和，再对点/patch 求均值，四阶段等权。每个阶段单独 backward，
micro-batch 按样本数加权，一个 DataLoader batch 只做一次 optimizer step。
Student 状态在阶段之间 detach，**不通过前面的阶段反传梯度**；没有四步 BPTT。
Teacher 保持冻结/eval。Student 保持 train，原 clipping、AdamW 和 best 选择不变。

噪声参数仍用 `sigma0 * 0.95**start_step`，Step Condition 仍用相同的
`start_step/target_step`，与推理的名义日程一致；不根据 Student 残差重新估计 sigma。
这项改动针对输入来源差异，训练 patch 与整云重叠 patch 推理、train/eval 模式等
其他差异仍然存在，不代表整个训练过程与测试完全相同。

## 默认混合调度

```yaml
training_input:
  type: scheduled_rollout
  warmup_epochs: 2
  decay_epochs: 6
  teacher_prob_start: 1.0
  teacher_prob_end: 0.2
  sampling_seed: 2025
```

q 是第 2～4 阶段使用 **Teacher** 的概率，Student 概率为 `1-q`。

| epoch | 1～2 | 3 | 4 | 5 | 6 | 7 | 8 及以后 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| q | 1 | 1 | 0.84 | 0.68 | 0.52 | 0.36 | 0.20 |

衰减窗口包含起止值；`decay_epochs: 1` 表示 warmup 后立即取结束概率。
可以设置起止概率相同实现恒定混合。`warmup_epochs >= 0`、`decay_epochs >= 1`，
且 `0 <= teacher_prob_end <= teacher_prob_start <= 1`。未知/无效参数直接报错。
第 1 阶段的 Teacher 比例始终是 1，因此不能把四阶段合并的 Teacher 比例当成 q。

采样使用独立 CPU generator，每轮种子为 `(sampling_seed + epoch - 1) % 2**32`，
不推进全局 Torch 随机数。整批采好 `[4, B]` mask 再切 micro-batch，调整
`student_patch_batch` 不会改变这个 batch 的输入来源选择。但真实模型的 dropout、
BatchNorm 等仍可能使不同 micro-batch 大小得到不同数值。q=0/1 不抽随机数。
CLI `--seed` 控制原来的模型/数据随机数，**不覆盖** YAML 的 `sampling_seed`。

训练仍只有四次 Student forward / micro-batch，逐阶段释放计算图，只保留上一阶段
detach 后的状态。没有增加 Teacher 轨迹长度或搜索候选。沿用 48 GiB 目标预算
（47 GiB PyTorch allocator + 1 GiB 外部分配预留）和 OOM 退出机制；真实显存未验证。

## 三份对照配置

- `cfgs/PointGPT-L/distill_16to4_input_teacher.yaml`：TF 对照。
- `cfgs/PointGPT-L/distill_16to4_input_scheduled.yaml`：建议首先比较的混合输入。
- `cfgs/PointGPT-L/distill_16to4_input_student.yaml`：从 epoch 1 开始连续 Student 的对照。

三份配置除 `training_input` 外完全相同，均固定等间隔节点、训练时启用现有
Step Condition、关闭 shadow 搜索。其余参数沿用 `distill_16to4_baseline.yaml`，
测试 SOR 开启、surface projection 关闭。原配置没有改写。
相比历史 baseline 关闭了只读 shadow 诊断；严格对照应使用这里的 TF 配置重新跑。

输入策略与 `original_pcd` / `rollout_aware` 的选择独立。将新策略用于已有动态配置时，
原节点选择公式、更新时机、校准集及 TF/FR 诊断仍保留；Student 的训练改变后，
搜索结果当然可能随之变化。首轮先使用上述固定节点配置，以便单独判断输入策略。

## 日志与 checkpoint

`train.jsonl` 增加 `training_input`、`training_input_protocol` 和
`training_input_stages`。每阶段记录 Teacher/Student 样本数、实际 Teacher 比例、
按实际来源分组的 `teacher_input_loss` / `student_input_loss`；某来源没有样本时为
`null`。按来源累加损失总和，再除实际计数，避免不同 micro-batch 混合比例带来偏差。

这些分组损失来自实际训练 forward，Student 组可能包含之前重置过 Teacher 的轨迹，
**不是独立 eval 得到的完整 TF/FR 诊断**。`stage_losses` 和 `loss_traj` 表示该输入
策略下的训练损失，`loss_traj_input` 标明来源；原 `val_loss_traj` 仍是 TF 验证，
由 `val_loss_traj_input` 标明。新输入模式下 `mean_PCD/mean_D_move/mean_E_imit`
以及动态日志里的训练 PCD std/range 为 `null`，避免把混合误差称作原 PCD。
独立搜索/诊断中的 PCD/TF/FR 指标不改定义。

每 5 个 epoch 的整云连续四步验证保持不变，best 仍按
`val_rollout_score = CD + 0.3 * P2M`，不按混合训练损失选模型。

新输入 checkpoint 保存策略、调度和当轮概率。模型权重格式、推理所用节点不变，
现有 Student 测试入口可加载。**新输入模式的 resume / 方法迁移暂不支持**：
新模式下 `--resume`，或以旧 TF 配置恢复新输入 checkpoint，均明确拒绝。
旧 TF + 原 PCD 的 resume 保持支持，rollout-aware 的原有 resume 限制保持。
`--stop_after_epoch 10` 会保存并停止，但不能承诺从新输入 epoch 10 续到 20。

## 验证状态与用户手动命令

本次不自动执行训练、测试、诊断或 GPU 程序。新增的 CPU 合成用例覆盖原 TF
损失/梯度一致、四步输入与 detach 梯度、patch 混合/重置、micro-batch 权重、
采样隔离、配置边界和 mock 训练日志/checkpoint。**这些用例尚未执行**；静态语法
检查不等于测试通过。

更新服务器代码后，用户可在 `/workspace` 的项目环境先运行定向 CPU 用例：

```bash
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p 'test_training_input*.py' -v
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p 'test_rollout*.py' -v
```

确认通过后，再按需要手动启动一个新混合输入实验（不是 Pilot-A）：

```bash
CUDA_VISIBLE_DEVICES=0 python -u tools/runner_distill.py --mode train \
  --config cfgs/PointGPT-L/distill_16to4_input_scheduled.yaml \
  --teacher_ckpt experiments/L_consistency_plus/ckpt-best.pth \
  --output_dir "experiments/input_ablation/mixed_$(date +%Y%m%d_%H%M%S)_$$" \
  --device 0 --seed 0 --stop_after_epoch 10
```

TF 对照改用 `distill_16to4_input_teacher.yaml` 和新的输出目录，保持相同 Teacher、
seed、数据、轮数和后处理。连续 Student 对照同理改用 `input_student` 配置。
默认计划为 20 epoch，上面命令在 10 epoch 正常保存/验证后停止。若要完整 20 epoch，
一开始去掉 `--stop_after_epoch 10`。两种训练模式的输入调度使用绝对 epoch，
不随计划总轮数重新缩放。不要覆盖既有实验输出。

下一步比较 epoch 5/10 的整云 rollout score 与实际输入比例，再用相同固定诊断
协议比较 TF/FR 阶段误差。训练 loss 或 TF/FR 差距缩小本身不能证明最终 CD/P2M 改善。
