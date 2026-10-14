# 固定节点 Mixed：高噪声 Teacher 输入保护

这是针对当前结果提出的下一轮实验，实现已完成，效果尚未验证。
最新 Mixed20 best 在 10k/1%、10k/2%、50k/1% 改善，但相对旧 Mixed e5，
50k/2% 和两档 3% 退化。训练日志显示 epoch 8 起阶段 2–4 的 Teacher 输入概率
统一降到 0.2。尚不能证明退化由这个比例引起，因此本轮只改变输入概率规则。

## 方法

原调度给每个 epoch 一个概率 `q_base`，新增配置按 patch 的初始 `sigma0` 修正：

```text
w = clip(log(sigma0 / sigma_start) / log(sigma_end / sigma_start), 0, 1)
q_patch = q_base + max(q_high - q_base, 0) * w
```

首轮设 `sigma_start=0.01`、`sigma_end=0.02`、`q_high=0.68`。
0.68 来自旧 Mixed e5 所处的 Teacher 概率，用作可解释的首轮试验值，并非已优化参数。
在 q_base=0.2 的后期：1% 及以下仍为 0.2，约 1.414% 为 0.44，1.5% 约为 0.481，
2% 为 0.68。warmup 的 q=1 保持不变；阶段 1 始终为 T0。

概率只用于阶段 2–4 的整个 patch 来源选择，仍允许中途重置到 Teacher 节点。
监督目标仍为原 Teacher 的 T4/T8/T12/T16，阶段间 detach、名义 sigma 衰减、
四阶段等权损失均沿用已有 Mixed。这里使用初始 sigma0，不能用每阶段衰减后的 sigma。

本轮固定 `[0,4,8,12,16]`，训练噪声仍为 0.5%–2% 的对数均匀采样，3% 不进入训练。
不改变数据划分、归一化、模型、优化器、测试后处理或评分；不增加 Teacher/Student
forward，不增加推理开销。节点策略和输入概率分离，后续可在验证有效后再与动态节点组合。

## 配置与日志

新配置：`cfgs/PointGPT-L/distill_16to4_input_noise_anchor.yaml`。
与实际 Mixed20 对照相同：20 epoch、seed0、每轮整云验证、SOR=false、projection=false。
原 scheduled 模板及公共数据配置保留原值。未设置 `noise_teacher_anchor` 的运行继续使用
原标量概率和原随机采样路径；旧 TF/Student rollout 不接受这个参数。

manifest/checkpoint 保存完整输入策略；`teacher_probability` 仍表示基础 q_base，新增
`teacher_probability_scope=base_before_noise_anchor` 明确其含义。
train.jsonl 的 `training_input_noise_bands` 按 sigma0<=1% 和 >1% 分组记录 patch 数、
计划概率的 min/max/mean，以及阶段 2–4 的实际 Teacher/Student 数。
完整 batch 先抽 mask 再切 micro-batch，继续使用独立 CPU generator，不推进全局采样 RNG。
新模式仍不支持 resume，需从同一 Teacher 新开实验。

best 继续按现有 10k/1% 整云 `CD+0.3*P2M` 选择，没有新增 3% 选模规则。
不能通过从六条件测试挑 epoch 来替代这个协议。

## 手动运行

代理没有启动测试、训练、诊断或 GPU 程序，没有提交代码。
新增/扩展用例覆盖配置边界、概率公式、向量 mask、随机数隔离、原控制配置、
分噪声日志及 mock checkpoint。这些用例尚未执行；静态核对不等于测试通过。

同步修改到服务器后，用户可先在原环境手动执行定向 CPU 用例：

```bash
cd /workspace
conda activate pointgpt
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s tests -p 'test_training_input*.py' -v
```

再从同一 Teacher 新跑 20 epoch：

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/train_fixed_noise_anchor.sh \
  experiments/L_consistency_plus/ckpt-best.pth 0
```

输出为独立 `experiments/input_ablation/mixed_noise_anchor_seed0_<timestamp>_<pid>/train`。
之后读取 train.jsonl、manifest.json、validation_manifest.json，确认实际分组概率和 best，
再沿用既有六条件测试命令，仅替换 TRAIN_ROOT。

## 进入动态节点前的判据

按用户提供的三位小数基础蒸馏表，3% 的参考为：

| 条件 | 基础 CD | 基础 P2M | 基础 Score | 基础+Step Condition Score |
|---|---:|---:|---:|---:|
| 10k / 3% | 4.067 | 1.911 | 4.6403 | 4.5765 |
| 50k / 3% | 1.234 | 0.825 | 1.4815 | 1.5185 |

以预先选出的同一 best 做六条件评估，首先要求两档 3% 均优于基础蒸馏，
同时分别报告 CD/P2M 和逐云改善数，不能只看 Score 抵消。尽量保留已有 1%/2% 收益。
Screenshot 数值是近似参考，正式判断应复用原始 baseline 文件。
对 Mixed 相对原 Mixed20 的归因，保持相同 Teacher、seed、20 epoch 和验证频率；
与旧基础蒸馏的比较还需披露训练预算，单个 seed 不能证明跨训练重复稳定。
