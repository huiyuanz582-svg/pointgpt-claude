# 第二篇论文阶段 1：固定 16→4 少步轨迹蒸馏

## 一、阶段目标

把第一篇论文中需要多次迭代的点云去噪模型压缩成少步模型：

- 教师执行 16 次去噪更新。
- 学生只执行 4 次更新。
- 固定映射为 `0 → 4 → 8 → 12 → 16`。
- 在尽量保持去噪质量的同时，将模型前向调用次数从 16 次降到 4 次。

阶段 1 只验证“多步轨迹能否被压缩”。暂时不同时改变模型大小，也不加入新的几何损失，避免多个变量混在一起。

## 二、教师模型是什么

教师是第一篇论文已经训练完成的 **PointGPT-L 点云去噪模型**，使用最终微调得到的最佳 checkpoint。

教师保持冻结，不参与反向传播。训练学生时，教师按照原来的参数执行迭代去噪：

- 初始步长：`0.3`
- 噪声衰减：`0.95`
- 目标终点：第 16 次更新

教师运行后，只保留第 `0、4、8、12、16` 步的点云状态，作为学生的学习目标。

这里的教师不是重新训练的标准 DDPM 教师，而是第一篇论文中的确定性迭代去噪模型。因此，本阶段更准确的名称是 **点云去噪轨迹蒸馏**。

## 三、学生模型是什么

阶段 1 的学生仍然使用 **PointGPT-L**，并从教师 checkpoint 初始化。

选择同容量学生的原因是先排除网络容量变化的影响。如果直接换成 PointGPT-S，实验失败时无法判断原因是蒸馏方法不成立，还是学生容量不足。

学生额外加入噪声和阶段条件，使同一个模型能够区分当前要完成哪个跳跃区间。推理时学生只运行 4 次。

## 四、学生学习教师的什么

学生学习的不是教师某一个单步输出，而是教师一段轨迹产生的**累计位移**：

- 学生第 1 步学习教师的 `0 → 4`。
- 学生第 2 步学习教师的 `4 → 8`。
- 学生第 3 步学习教师的 `8 → 12`。
- 学生第 4 步学习教师的 `12 → 16`。

也就是说，学生的一次前向传播要替代教师连续 4 次前向传播。

## 五、训练路线

训练分成两个阶段：

### 1. 教师状态预热

把教师轨迹中的真实中间状态输入学生，随机训练一个跳跃区间。这样学生先分别学会四种基本跳跃，训练更稳定、显存消耗也较低。

### 2. 学生完整展开

学生从原始噪声点云出发，连续执行自己的 4 次更新。训练同时约束：

- 学生中间状态接近教师对应轨迹节点。
- 学生最终状态接近教师第 16 步结果。
- 学生最终状态接近干净点云。

完整展开可以让学生适应自己前面步骤产生的误差，减少训练和推理之间的输入差异。

## 六、推理流程

测试时只加载学生，不再运行教师：

```text
噪声点云 → 学生步骤 1 → 学生步骤 2 → 学生步骤 3 → 学生步骤 4 → 去噪点云
```

对应的教师时间节点是：

```text
0 → 4 → 8 → 12 → 16
```

## 七、阶段 1 的判断标准

主要比较以下模型：

- 原始噪声点云。
- 教师 4 步结果。
- 教师 8 步结果。
- 教师 16 步结果。
- 蒸馏后的学生 4 步结果。

重点观察 CD、P2M、HD/HD95、推理时间和实际模型前向次数。如果学生 4 步明显优于教师直接截断到 4 步，并接近教师 16 步，就说明轨迹蒸馏具有继续研究的价值。

## 八、当前代码对应关系

- 蒸馏训练入口：`tools/runner_distill.py`
- 轨迹生成和蒸馏损失：`utils/trajectory_distill.py`
- 学生条件输入：`models/PointGPT.py`
- 完整点云 4 步推理：`tools/runner_finetune.py`
- 阶段 1 配置：`cfgs/PointGPT-L/distill_fixed16_to4.yaml`

## 九、下一阶段方向

固定 16→4 基线成立后，再逐项尝试：

1. 加入法向、曲率或局部邻域等几何感知损失。
2. 把均匀节点改成非均匀教师节点。
3. 尝试 PointGPT-S 等更小学生。
4. 尝试 2 步或 1 步学生。
5. 研究根据噪声和局部几何动态决定迭代次数。

## 十、运行方式

在服务器仓库根目录训练：

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config cfgs/PointGPT-L/distill_fixed16_to4.yaml \
  --distill_model \
  --ckpts experiments/finetune_scoredenoise/done-best/L_consistency_plus/ckpt-best.pth \
  --exp_name L_T16_S4_baseline \
  --val_freq 5
```

测试蒸馏学生：

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config cfgs/PointGPT-L/distill_fixed16_to4.yaml \
  --test \
  --ckpts experiments/distill_fixed16_to4/PointGPT-L/L_T16_S4_baseline/ckpt-best.pth \
  --exp_name L_T16_S4_baseline
```

训练时传入的 `--ckpts` 是冻结教师；测试时传入的 `--ckpts` 是已经训练好的学生。

## 十一、第一版试跑后的第二版课程

第一版在纯 teacher-forced 阶段结束后直接切到完整 rollout，出现验证退化；同时，
未经尺度归一化的轨迹、终点和 clean 损失远小于 jump loss。第二版配置
`distill_fixed16_to4_curriculum.yaml` 做了两项修正：

- 按每个样本的初始噪声尺度归一化状态损失。
- 用 12 个 epoch 逐渐把 rollout batch 比例从 0 增加到 1。

可以使用第一版保存的 epoch 5 最佳学生作为初始化，但冻结教师仍然使用第一篇论文的
最终 checkpoint：

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config cfgs/PointGPT-L/distill_fixed16_to4_curriculum.yaml \
  --distill_model \
  --ckpts experiments/finetune_scoredenoise/done-best/L_consistency_plus/ckpt-best.pth \
  --start_ckpts experiments/distill_fixed16_to4/PointGPT-L/L_T16_S4_baseline/ckpt-best.pth \
  --exp_name L_T16_S4_curriculum_v2 \
  --val_freq 5
```

程序会在训练前使用同一个留出验证集分别评估：学生初值、教师 4 步和教师 16 步，
并把学生初值保存为 `ckpt-init.pth` 和初始 `ckpt-best.pth`。

## 十二、第三版：rollout 纠偏与教师状态锚点

第二版在 rollout 比例约为 0.67 时得到更好的结果，但 rollout 提高到 1 后再次退化。
第三版做两项修改：

- rollout 的 jump 目标由“教师当前节点到教师下一节点”改为“当前学生状态到教师下一节点”，
  让每一步主动消除前面累计的轨迹偏差。
- rollout 概率最高为 0.75，始终保留约 25% teacher-forced 样本，防止学生遗忘教师状态上的
  正确区间映射。

从第二版 epoch 10 的最佳学生启动第三版实验：

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config cfgs/PointGPT-L/distill_fixed16_to4_corrective.yaml \
  --distill_model \
  --ckpts experiments/finetune_scoredenoise/done-best/L_consistency_plus/ckpt-best.pth \
  --start_ckpts experiments/distill_fixed16_to4_curriculum/PointGPT-L/L_T16_S4_curriculum_v2/ckpt-best.pth \
  --exp_name L_T16_S4_corrective_v3 \
  --val_freq 5
```
