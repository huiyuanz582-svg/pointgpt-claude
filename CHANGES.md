# CHANGES

会话改动记录 —— 不会替代 commit message 和 CLAUDE.md,只是给一个鸟瞰视角。

## Session 2026-06-03:PointGPT 去噪重构(3 个 commit)

### 背景

仓库是 PointGPT(NeurIPS 2023)的 fork,被改造用于点云去噪(PUNet / ScoreDenoise pipeline)。微调代码集中在 `tools/runner_finetune.py`、`datasets/ScoreDenoiseDataset.py`、`models/PointGPT.py`、`utils/p2m_loss.py`。详细背景见 `CLAUDE.md`。

本会话围绕一个核心问题:**让 fine-tune 流程跑通 + 把 backbone 从"残差预测"改成"score 估计",对齐 2021 年后强势的去噪方法(Score-Denoise / IterativePFN)**。

### Commit 1 — `eef6d51` `Add CLAUDE.md`

新建 `CLAUDE.md`,给后续 Claude 会话提供:
- 项目实际形态(upstream README 中分类/分割等任务在 fork 里不可用)
- `main.py` 三种 dispatch 模式(pretrain / finetune / test)
- `data/ScoreDenoise/PUNet/` 数据集布局,以及代码里硬编码的 mesh 路径
- `requirements.txt` 没列但必装的依赖(chamfer_dist / emd / pointnet2_ops / KNN_CUDA / pytorch3d 等)
- `DenoiseMetrics` 用 `cd + 0.3*p2m` 评估 checkpoint 的标定理由

### Commit 2 — `6c30dc9` `Refactor denoising fine-tune: patch-based training + critical fixes`

**P0 修复(否则训练跑不起来 / 配置静默失效)**
- `tools/runner_finetune.py`:删 `clean_normals` kwarg 调用,消除 `forward` TypeError
- `tools/builder.py`:把 `config.others.bs` 同步到 `_base_.TRAIN_BATCH_SIZE`,让 `total_bs` 真正生效
- `utils/p2m_loss.py`:绝对路径 → 相对路径 + `PUNET_MESH_ROOT` env var

**Patch-based 训练(方案 B,跟 PointGPT pretrain 尺度对齐)**
- `datasets/ScoreDenoiseDataset.py PairedPatchDataset.__getitem__`:训练时随机种子 + KNN 切 1024-patch;**用 noisy 空间的 KNN index 同步索引 clean 和 noisy**,保证点对齐
- `cfgs/dataset_configs/ScoreDenoise.yaml`:`PATCH_SIZE: 1024`(原 10000)
- `cfgs/PointGPT-S/finetune_scoredenoise.yaml`:`num_group: 64`(原 2048,对齐 S pretrain),`npoints: 1024`,`total_bs: 32`
- `cfgs/PointGPT-L/finetune_scoredenoise.yaml`:`num_group: 128`(原 512,对齐 L pretrain),`npoints: 1024`,`total_bs: 16`

**训练稳定性**
- `models/PointGPT.py`:删 `loss *= 1e4`(配合默认 lr / grad_clip 更稳)
- encoder/generator 改为全 attention(`attn_mask=None`),去噪不需要因果约束

**死代码清理**
- 删所有 `pcl_clean_50k` / `pcl_noisy_50k` 路径 —— 实际只是 `pcl_clean` 的别名,从 transforms、dataset、collate、model.forward 全链路移除
- `test()` 加 `os.makedirs(exist_ok=True)`,避免第一次保存可视化时崩在路径不存在

### Commit 3 — `0ed32b1` `Score-based denoising: Step 1+2 (output semantics + DSM loss)`

调研了 7 个点云去噪 repo(DMRDenoise / Score-Denoise / PointCleanNet / Pointfilter / IterativePFN / P2P-Bridge / GPDNet)的根本流派,选定 **Score-based**(Score-Denoise ICCV 2021 的核心思想)作为唯一主线改造,不堆叠其它小改进。

**Step 1:改输出语义**
- `models/PointGPT.py`:新增 `project_patch_scores_weighted`(融合 patch 内 score 时不加 center,因为 score 是方向向量而非坐标)
- `PointTransformer.forward` 把 generator 的 `[B, G, M, 3]` 输出从"残差"重新解释为"score 场 ∇log p_σ(x)"

**Step 2:换 DSM loss**
- 训练 loss:`0.5 * mean(σ² * (pred_score - target_score)²)`,其中 `target_score = (clean - noisy) / σ²`(Vincent 2011 σ²-加权)
- 训练时 `σ` 从 `noise_std` 字段透传到模型(`transforms.AddNoise` 设置,`PairedPatchDataset` 不再 pop,`denoise_collate_fn_test` 收集成 6-tuple)
- 推理(val/test)单步 Tweedie:`x̂ = x + σ²·score`(Step 3 会替换为 Langevin 多步)

**关键工程细节**
- 取消 encoder 冻结(score 估计需要 encoder 适配,跟 pretrain 的自回归目标不一致)
- ckpt 加载后**重新初始化 `generator_blocks.increase_dim` 输出头**(`std=0.01` weight,zero bias),让模型从"identity"起步而不是被预训练的"生成绝对坐标"输出头污染

### 还没做(Step 3)

`utils/p2m_loss.py:54` 的相对路径只用了 `data/ScoreDenoise/PUNet/meshes/`,如果你 mesh 放在别处,设 `PUNET_MESH_ROOT` 环境变量。

**Langevin 多步推理(Step 3)** 等你跑出 Step 1+2 的收敛信号后再做,需要根据 train loss / val CD 的实际曲线调:
- step_size(默认 0.2)
- decay(默认 0.95)
- num_steps(默认 10-30)

### 跑命令

```bash
# PointGPT-S
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config cfgs/PointGPT-S/finetune_scoredenoise.yaml \
  --finetune_model --exp_name score_s1_dsm \
  --ckpts pretrainModel/S/pretrained.pth

# 测试(假设 Step 1+2 已经训出了一个 ckpt)
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config cfgs/PointGPT-S/finetune_scoredenoise.yaml \
  --test --exp_name score_s1_test \
  --ckpts experiments/finetune_scoredenoise/PointGPT-S/score_s1_dsm/ckpt-best.pth
```

### 期望训练信号

- **Loss 数量级**:`σ² × MSE` 在 σ ∈ [0.005, 0.020] 下应该是 ~0.5 到 ~5 之间(取决于初始 score 残差)。NaN 或暴涨 → sigma 没正确广播或 target_score 出问题。
- **CD on val**:刚开始可能比 noisy 还差(模型从 identity 起步),前几个 epoch 应该看到 CD 单调下降。前 10 个 epoch 内进入 1-3 区间(归一化空间 × 1e4)是好兆头。
- **P2M**:如果 CD 下降但 P2M 不动,说明模型 fit 到"平均位置"但没贴到表面 —— 这是单步 Tweedie 的局限,Step 3 Langevin 多步应该能改善。

### 怎么回滚

如果某个 commit 有问题,可以 `git revert <hash>`:
- 回滚到 P0 修复前:`git revert 0ed32b1 6c30dc9`(留下 CLAUDE.md)
- 回滚 score-based,保留 P0 修复:`git revert 0ed32b1`
- 完全恢复 first commit:`git reset --hard 4ecff66`(注意:`--hard` 会丢失工作目录修改)
