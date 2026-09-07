# 第二篇论文阶段 1：固定 16→4 点云去噪轨迹蒸馏

分支：`paper2-stage1-fixed16-distill`

## 1. 研究目标

将第一篇论文的多步点云去噪模型压缩为少步模型：

- 教师执行 16 次去噪更新。
- 学生只执行 4 次更新。
- 固定轨迹节点为 `0 → 4 → 8 → 12 → 16`。
- 学生的一次前向传播替代教师连续 4 次前向传播。
- 教师和学生都使用 PointGPT-L，暂时不改变模型容量，也不加入新的几何损失。

## 2. 教师与学生

教师是第一篇论文训练完成的 PointGPT-L 最佳模型：

```text
experiments/finetune_scoredenoise/done-best/L_consistency_plus/ckpt-best.pth
```

教师使用 `step_size=0.3`、`decay=0.95` 生成轨迹，只保留 `T0、T4、T8、T12、T16` 五个状态。训练期间教师始终冻结，不参与反向传播。

学生同样使用 PointGPT-L，并从教师 checkpoint 初始化。学生增加噪声尺度、学生阶段和教师时间等条件输入，用同一个模型完成四种轨迹跳跃。测试时只运行学生，不再加载教师。

## 3. 学生学习的映射

```text
学生第 1 步：教师 T0  → T4
学生第 2 步：教师 T4  → T8
学生第 3 步：教师 T8  → T12
学生第 4 步：教师 T12 → T16
```

学生学习的是教师一段轨迹产生的累计位移，而不是教师的单次更新。

## 4. 当前训练方法

第一篇论文同协议实验发现，未经蒸馏的学生初值已经具有很强的 4 步去噪能力。旧方案更新全部 PointGPT 参数并强制拟合教师中间节点，反而破坏了这个强初始化。

当前版本改为保守的端点残差蒸馏：

- 冻结 PointGPT 主干，只训练零初始化的条件 MLP。
- 全部 batch 使用学生完整 4 步 rollout。
- 不再强制学生经过教师的 T4、T8、T12。
- 主要监督学生最终状态接近教师 T16 和干净点云。
- 冻结主干保持 eval 模式，避免 DropPath 改变初始化行为。

程序仍统计完整 rollout 的各项误差，但当前总损失只启用：

- `endpoint loss`：学生最终状态与教师 T16 之间的误差。
- `clean loss`：学生最终状态与干净点云之间的误差。

状态损失按照初始噪声尺度归一化，权重为：

```text
jump=0.0, trajectory=0.0, endpoint=1.0, clean=0.1
```

当前不使用 teacher-forced 课程，避免逐段轨迹目标改变学生原本有效的中间路径。

## 5. 正式数据协议

正式实验与第一篇论文保持一致。

### 训练集

- PUNet train 的全部 40 个 shape。
- 使用 10k、30k、50k 三种分辨率，共 120 个完整点云。
- 每个点云每轮动态采样 50 次，共 6000 个训练 patch。
- 每个 patch 包含 1024 个点。
- 噪声标准差在 `[0.005, 0.02]` 内按对数均匀分布采样。

### 验证集

- PUNet test 的 20 个 shape。
- 使用 10k 分辨率和 1% 固定高斯噪声。
- 开启与第一篇论文一致的 SOR 后处理。

## 6. 代码位置

- 蒸馏训练入口：`tools/runner_distill.py`
- 教师轨迹和蒸馏损失：`utils/trajectory_distill.py`
- 学生条件输入：`models/PointGPT.py`
- 完整点云 4 步推理：`tools/runner_finetune.py`
- 快速验证配置：`cfgs/PointGPT-L/distill_fixed16_to4_endpoint_residual.yaml`
- 正式配置：`cfgs/PointGPT-L/distill_fixed16_to4_endpoint_residual_paper1_protocol.yaml`

## 7. 运行方法

### 快速验证

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config cfgs/PointGPT-L/distill_fixed16_to4_endpoint_residual.yaml \
  --exp_name L_T16_S4_endpoint_residual_v4 \
  --distill_model \
  --ckpts experiments/finetune_scoredenoise/done-best/L_consistency_plus/ckpt-best.pth \
  --val_freq 1
```

只有快速验证的 `ckpt-best.pth` 优于 `ckpt-init.pth` 后，才使用 50 倍采样正式配置重新训练。训练时的 `--ckpts` 是冻结教师，不要添加 `--start_ckpts`。

### 正式测试

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config cfgs/PointGPT-L/distill_fixed16_to4_endpoint_residual.yaml \
  --test \
  --ckpts experiments/distill_fixed16_to4_endpoint_residual/PointGPT-L/L_T16_S4_endpoint_residual_v4/ckpt-best.pth \
  --exp_name L_T16_S4_endpoint_residual_v4_test
```

测试时的 `--ckpts` 是蒸馏完成的 4 步学生。

## 8. Checkpoint 含义

- `ckpt-init.pth`：蒸馏开始前的 4 步学生。
- `ckpt-last.pth`：最近一个 epoch 的 4 步学生。
- `ckpt-best.pth`：验证指标最好的 4 步学生。

教师 4 步和教师 16 步只用于基线评估，不会保存为学生 checkpoint。

## 9. 当前状态

开发版实验的最佳学生出现在 Epoch 10：

| 模型 | CD | P2M | 综合分数 |
|---|---:|---:|---:|
| 教师 4 步 | 4.8970 | 1.8778 | 5.4604 |
| 学生 4 步（Epoch 10） | **4.5228** | **1.6919** | **5.0304** |
| 教师 16 步 | 3.5849 | 1.1702 | 3.9360 |

第一篇论文同协议的旧方案训练到 Epoch 30 后，所有验证结果都差于学生初值；`ckpt-best.pth` 因此仍是 `ckpt-init.pth`。这说明旧逐段轨迹蒸馏无效，学生优于教师直接 4 步主要来自强初始化和快速采样日程，不能归因于蒸馏。

当前代码已改为端点残差蒸馏。下一步先运行 5 倍采样快速实验；只有训练后的验证分数低于初始化分数 `2.2127`，才说明新蒸馏方案真正产生了正收益。
