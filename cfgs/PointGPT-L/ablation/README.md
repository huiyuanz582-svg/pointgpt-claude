# 消融实验（对应论文 Method 章 M1–M4 + 框架级）

消融表共 8 行；评测统一用 PUNet 高斯 10k 的 1%/2%/3%（改 `cfgs/dataset_configs/ScoreDenoise.yaml`
的 `TEST_NOISY_DIR` + `TEST_NOISE` 切噪声级）。全部实验与主实验同训练日程（120 epoch、同 lr、同选模型）。

## 一览

| # | 消融表的行 | 验证的模块 | 配置 | 是否重训 |
|---|---|---|---|---|
| 1 | Full method | 基准 | `../finetune_scoredenoise.yaml` | 已有 |
| 2 | w/o pre-training (scratch) | 框架级：预训练有效性 | 同上（改命令行） | 🔴 是 |
| 3 | naive transfer | 框架级：适配必要性 | 上游原始 PointGPT 代码 | 已在跑 |
| 4 | fresh FC head (PMI-style) | M1 复用预训练 decoder | `abl_fc_decoder.yaml` | 🔴 是 |
| 5 | w/o unrolled training | M3 | `abl_single_step_train.yaml` | 🔴 是 |
| 6 | w/o surface term (λ=0) | M2 | `abl_no_p2m.yaml` | 🔴 是 |
| 7 | single-step inference (T=1) | M4 | `abl_T1.yaml` | 🟢 只测试 |
| 8 | w/o soft fusion | M4 | `abl_equal_fusion.yaml` | 🟢 只测试 |
| — | (诊断，不进表) SOR off | P2M 诊断 | `abl_no_sor.yaml` | 🟢 只测试 |

## 需重训的（②④⑤⑥，串行排队）

```bash
# ② scratch —— 注意：--scratch_model 且【不加 --ckpts】
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config cfgs/PointGPT-L/finetune_scoredenoise.yaml \
  --scratch_model --exp_name abl_scratch --val_freq 5

# ④ PMI 式换头
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config cfgs/PointGPT-L/ablation/abl_fc_decoder.yaml \
  --finetune_model --exp_name abl_fc_decoder \
  --ckpts pretrainModel/L/post_pretrained.pth --val_freq 5

# ⑤ 单步训练（⑥ 同理换 abl_no_p2m.yaml / exp_name）
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config cfgs/PointGPT-L/ablation/abl_single_step_train.yaml \
  --finetune_model --exp_name abl_single_step \
  --ckpts pretrainModel/L/post_pretrained.pth --val_freq 5
```

训练完测试：**用训练时的同一份 yaml**（消融开关经 inject_ablation 自动贯通到测试），
`--ckpts` 指向该实验自己的 `ckpt-best.pth`：

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config cfgs/PointGPT-L/ablation/abl_fc_decoder.yaml \
  --test --exp_name abl_fc_decoder \
  --ckpts experiments/abl_fc_decoder/ablation/abl_fc_decoder/ckpt-best.pth
```

## 只需测试的（⑦⑧+诊断，用完整方法的 best ckpt，今天就能出数）

```bash
BEST=experiments/finetune_scoredenoise/PointGPT-L/<你的主实验>/ckpt-best.pth
for CFG in abl_T1 abl_equal_fusion abl_no_sor; do
  CUDA_VISIBLE_DEVICES=0 python main.py \
    --config cfgs/PointGPT-L/ablation/$CFG.yaml \
    --test --exp_name $CFG --ckpts $BEST
done
```

加分项：把 `abl_T1.yaml` 的 `num_steps` 依次改 1/5/10/20/30 各测一次，
画 CD/P2M vs 推理步数曲线（比表格单行更有说服力）。

## 注意

- 可视化输出已按 `exp_name` 分目录（`runner_finetune.py` 的 test()），多组测试不再互相覆盖。
- `scratch` 与 naive 的区别：scratch = **适配后的架构**随机初始化（证预训练价值）；
  naive = 原始 PointGPT 未适配直接微调（证适配价值）。两行都要。
- 每个实验独立 `--exp_name`，checkpoint 互不覆盖。
