# 固定节点 Mixed：从 Student 实际输入生成 Teacher 监督

本轮撤回上一次按 sigma 调整 Teacher 概率的代码、配置、启动脚本和相关用例。
原 Mixed 仍作为对照，训练噪声仍为 0.5%–2%，3% 不进入训练；新方法效果尚未验证。

## 输入概率

恢复每个 epoch 统一的 Teacher 概率：epoch 1–3 为 1，4–8 依次为
0.84、0.68、0.52、0.36、0.20，之后为 0.20。概率不再依赖 sigma。
原标量采样实现与独立 CPU generator 恢复为原 Mixed 版本，仍在整批抽 mask 后切 micro-batch。

## 新的监督方式

新配置 `cfgs/PointGPT-L/distill_16to4_input_matched.yaml` 仅额外启用：

```yaml
training_input:
  student_target: teacher_from_student_input
```

时间节点固定为 `[0,4,8,12,16]`。阶段 k 对应区间 `[t,u]`：

- Teacher 输入样本：输入原轨迹 T[t]，目标仍是原轨迹 T[u]。
- Student 输入样本：输入上一阶段真实输出 `.detach()`，冻结 Teacher 从该输入出发，
  以初始 sigma `sigma0 * 0.95**t` 执行 `u-t` 次原更新，生成该样本的新监督目标。

例如第二阶段输入 Student 第一步输出时，Teacher 从这个状态执行4个小步作为目标，
而不是直接使用原 noisy 输入出发的 T8。后续阶段同理；中途重置到 Teacher 的语义保持不变。
第一阶段始终用 T0→T4，不增加 Teacher 计算。

Teacher 更新逐式沿用原实现：先由 `(out-x)/sigma` 得到 epsilon，
`x += 0.3*sigma*epsilon`，每步 `sigma *= 0.95`。Teacher 冻结、eval、no_grad，
目标与上一阶段 Student 状态均 detach。原 Teacher 节点缓存不会被新标签写回或覆盖。
额外标签生成保存/恢复 Python、NumPy、Torch CPU 和对应 CUDA device 的 RNG 状态，
避免推进 Student dropout 和后续数据采样所用的全局随机流。

损失仍是对应点平方 L2、四阶段等权、逐阶段 backward，micro-batch 按样本数加权，
每个 DataLoader batch 一次 optimizer step。没有增加 clean/CD/P2M 训练损失，
没有跨阶段 BPTT，也没有修改模型、推理与评分。

不配置 student_target 时保持原 Teacher 节点目标；也可显式设为 `teacher_target_node`。
这个参数只接受 scheduled_rollout / student_rollout，teacher_forced 下会报错，避免被静默忽略。
Student 输入训练仍不支持 resume，需要从同一 Teacher 新开实验。

## 计算代价与记录

额外 Teacher 标签按 `teacher_patch_batch`（本轮为1）分批计算，只保留终点。
每个 Student-source 样本增加当前区间长度的 Teacher sample-steps；Teacher-source 样本无额外开销。
节点等间隔且后期 q=0.2 时，每个训练 patch 期望增加 `3*0.8*4=9.6` 个 Teacher 步，
原 canonical Teacher16 和 Student4 仍保留。实际耗时和显存需实测，不能承诺与原 Mixed 相同。
推理仍为4次 Student 更新，无额外 Teacher 调用。

manifest/checkpoint 的 training_input_protocol 区分 Teacher 输入与 Student 输入的目标。
train.jsonl 增加 `training_target_stages`，每阶段记录：

- `teacher_node_target_count`：使用原节点目标的样本数。
- `student_input_teacher_target_count`：从 Student 实际输入生成 Teacher 目标的样本数。
- `extra_teacher_sample_steps`：额外 Teacher 样本步数，不是 batched forward 调用次数。

`loss_traj_target=input_matched_teacher_interval` 标记新的训练目标。
`val_loss_traj` 仍在原固定80个 Teacher-forced patch 上评估原节点目标，
由 `val_loss_traj_target=teacher_target_node` 标记，不能把两种 loss 直接当作同一个目标。
每轮整云4步 rollout 验证仍不使用 Teacher，best 继续按原 `CD+0.3*P2M` 选择。

## 手动验证与训练

本轮没有自动运行 CPU 用例、训练、测试、模型诊断或 GPU 程序，没有提交代码。
静态语法和配置检查不等于真实效果验证。同步代码到原服务器环境后，可手动运行：

```bash
cd /workspace
conda activate pointgpt
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p 'test_training_input*.py' -v
```

通过后，从同一 Teacher 新跑20 epoch：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_fixed_matched_targets.sh \
  experiments/L_consistency_plus/ckpt-best.pth 0
```

配置与实际原 Mixed20 对照保持相同的 seed0、20 epoch、每轮整云验证、SOR=false、projection=false。
训练输出为 `experiments/input_ablation/mixed_matched_targets_seed0_<timestamp>_<pid>/train`。
六条件测试流程不变，只替换 TRAIN_ROOT。先核对新训练日志和两个 manifest，再评估同一个 best。
3% 两档应对照用户给出的基础蒸馏（截图 CD/P2M：10k 为 4.067/1.911，50k 为 1.234/0.825），
并披露原始 baseline 精度与训练预算；固定节点下取得收益后再考虑动态节点。

本设计借鉴在学习者访问状态上获取专家监督的思路，
参考 [DAgger](https://proceedings.mlr.press/v15/ross11a.html)。这里没有实现完整的数据聚合算法，
也不将该论文的理论保证直接套用于点云蒸馏。
