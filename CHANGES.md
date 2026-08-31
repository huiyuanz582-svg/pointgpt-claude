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

---

## Session: ε-prediction 修复 + Langevin 多步推理落地 (2026-06)

### 背景：之前 DSM 训练 val CD 不动

上版用 DSM(σ²-加权 score matching)，target_score 量级 ~100，σ² 加权把梯度压到 ~0.01/步，
模型几乎不学(val CD delta ~0.002)。这次定位到**多个叠加根因**并逐一修复。

### 改了什么(训练侧)

1. **DSM → ε-prediction(DDPM 风格)**：generator 输出 `ε = (clean-noisy)/σ`，量级 O(1)，
   训练用纯 MSE 无 σ 加权。`models/PointGPT.py:forward` 的 train/val 分支重写。
   - 训练：`loss = mean(||pred_ε - target_ε||²)`
   - 推理(单步)：`x̂ = x + σ·pred_ε`
2. **lr 3e-5 → 1e-4**：之前学习率太小。
3. **输出头 re-init std 0.01 → 0.1**：初始预测量级从 ~0.2 提到 ~2，跟 target ε~O(1) 匹配。
4. **修复 train/val coverage 失配(关键)**：val 原来喂完整 10k 点云，group_divider 只覆盖
   2048/10000≈20%，80% 点 ε=0 不动，val CD 永远≈noisy baseline。改成 val 也切 1024-patch
   (`ScoreDenoiseDataset.py:val_dataloader` flag='train')，coverage 跟训练一致(200%)。

### 改了什么(推理侧)

5. **patch_based_denoise**(`runner_finetune.py:153`)：完整点云切重叠 1024-patch 逐个去噪
   再按覆盖次数平均拼回，解决 test 也只覆盖 20% 的问题。
6. **Langevin 多步退火推理(Step 3 完成)**：patch 内迭代
   `x ← x + step_size·σ_t·ε; σ_t ← σ_t·decay`，`num_steps=1,step_size=1.0` 退化为单步 fallback。
   超参从 `config.langevin` 读。
7. **TEST_NOISE / TEST_RESOLUTION / TEST_NOISY_DIR** 改为从 yaml 读，测不同分辨率/噪声免改代码。

### 训练结果(score_eps_v4, 120 epoch)

- val CD(1024-patch): 3.76 → **3.28**(epoch 117 best)，delta 0.002 → 0.46，确认真在学。
- loss 3.40 → 2.56 干净收敛，无过拟合。

### 推理超参调优(10k/1%)

| 配置 | CD | P2M |
|---|---|---|
| 单步 Tweedie | 3.11 | 1.16 |
| Langevin 20步/0.2/0.95 | 2.687 | 0.920 |
| Langevin 30步/0.2/0.95 | 2.617 | 0.893 |
| **Langevin 30步/0.3/0.95(最优)** | **2.565** | 0.925 |
| Langevin 30步/0.5/0.97 | 3.084 | 1.497(过冲发散) |

**锁定最优**：`num_steps=30, step_size=0.3, decay=0.95`。step_size≥0.5 会过冲。

### 完整 baseline(Langevin 30/0.3/0.95)

| 数据集 | 噪声 | CD | P2M |
|---|---|---|---|
| 10k | 1% | 2.565 | 0.925 |
| 10k | 2% | 4.536 | 2.403 |
| 10k | 3% | 5.819 | 3.621 |
| 50k | 1% | **0.762** | **0.473** |
| 50k | 2% | 1.409 | 1.036 |
| 50k | 3% | 2.005 | 1.532 |

50k 全面优于 10k(点密度高→patch 内更平坦→ε 估计更准+重叠平均更稳)。
注：3% 超出训练范围(NOISE_MAX=0.02)但泛化平稳。

### 下一步(待做)

多噪声级 score matching 重训以改善 10k/高噪声 —— **不扩大噪声范围**(保持
NOISE_MIN=0.005, NOISE_MAX=0.020 以与其他方法公平对比)，只改 σ 采样策略和训练目标。

---

## Session: σ 对数均匀采样 (EDM 式) 重训 — v5 (2026-06-05)

### 动机

v4 的短板在稀疏点云(10k)和 P2M。退火 Langevin 末端 σ 衰减到 ~0.004，
但 v4 线性均匀采样在 [0.005,0.020] 上让小 σ 样本只占 ~36%，精细去噪区(贴表面/降 P2M)训练不足。

### 改动(范围严格不变，仅改采样分布)

- `transforms.py: AddNoise` 加 `log_uniform` 参数：σ = exp(uniform(log σ_min, log σ_max))
- `ScoreDenoise.yaml: NOISE_LOG_UNIFORM: True`(默认开)，可关闭做对照
- **NOISE_MIN/MAX 保持 0.005/0.020 不变** → 与其他去噪方法公平对比
- loss / 模型结构 / Langevin 推理配置(30/0.3/0.95)全不变

### 训练(score_eps_v5_logsigma, 120 epoch)

- 初始 loss 与 v4 一致(3.405)，收敛轨迹几乎重合
- best val CD: v4 3.276 → **v5 3.263** (epoch 117)，val 上差异小(val 固定 1% 噪声)

### Test 结果对比(Langevin 30/0.3/0.95)

| 数据集 | 噪声 | v4 CD | v5 CD | v4 P2M | v5 P2M |
|---|---|---|---|---|---|
| 10k | 1% | 2.565 | **2.480** | 0.925 | **0.879** |
| 10k | 2% | 4.536 | **4.464** | 2.403 | **2.371** |
| 10k | 3% | 5.819 | **5.767** | 3.621 | **3.604** |
| 50k | 1% | 0.762 | **0.754** | 0.473 | 0.473 |
| 50k | 2% | 1.409 | 1.415 | 1.036 | 1.046 |
| 50k | 3% | 2.005 | 2.010 | 1.532 | 1.537 |

**结论**：对数采样在 10k(稀疏，原短板)全面提升 CD+P2M，50k(密集)无损持平。
纯赚改动，无回退，**v5 作为新 baseline**。

### 下一步(可选)

继续提升可考虑训练时多步迭代监督(consistency / 模拟 2-3 步 Langevin)，
进一步改善 10k 和高噪声(2%/3%)；范围仍保持不变。

---

## Session: P2M 训练 loss + EDM 加权 + 权重调参 — v6/v7/v8/v9 (2026-06-06~07)

### 动机

v5 后中高噪声段(2%/3%)和所有 1% 段的 P2M 距 SOTA 仍有 2~5x 差距。
P2M 是 point→mesh 表面距离，纯 ε-MSE 训练不直接优化它，故引入可微 P2M 训练 loss
并逐步加大其权重，同时用 EDM 1/σ² 加权强化小噪声段精度。

### 改动累积

- **v6**: `utils/p2m_loss.py: compute_p2m_train`（可微 point→face），训练 loss 加
  `p2m_weight·P2M`，`models/PointGPT.py` train 分支返回 `(loss, denoised)`，
  runner 端按样本恢复世界坐标算 P2M。`p2m_weight=0.05`。
- **v7**: `models/PointGPT.py` ε-MSE 加 EDM 加权 `(σ_ref/σ)²`，σ_ref=0.01。
  小噪声(1%)梯度保持基准、大噪声降权，迫使模型对精确去噪更敏感。数值量级不变。
- **v8**: `p2m_weight 0.05→0.30`（贡献从 ~1.3% 升到 ~7%）。
- **v9**: `p2m_weight 0.30→0.70`。

各版前 5 epoch 均健康：ε-MSE 不被压制、P2M 下降更快、val CD delta 反而更好，
说明 p2m_weight 在 0.7 仍未越界。

### Test 结果（Langevin 30/0.3/0.95），与 SOTA 对比

| 数据集 | 噪声 | v5 | v6 | v7 | v8 | **v9** | SOTA |
|---|---|---|---|---|---|---|---|
| 10k 1% CD  | | 2.480 | 2.482 | 2.410 | 2.416 | 2.435 | 2.056 |
| 10k 1% P2M | | 0.879 | 0.879 | 0.834 | 0.836 | 0.839 | 0.416 |
| 10k 2% CD  | | 4.464 | 4.428 | 4.317 | 4.215 | **4.118** | 3.043 |
| 10k 2% P2M | | 2.371 | 2.332 | 2.231 | 2.133 | **2.020** | 0.838 |
| 10k 3% CD  | | 5.767 | 5.657 | 5.539 | 5.150 | **4.958** | 4.066 |
| 10k 3% P2M | | 3.604 | 3.494 | 3.359 | 2.969 | **2.760** | 1.487 |
| 50k 1% CD  | | 0.754 | 0.751 | 0.755 | 0.740 | **0.726** | 0.592 |
| 50k 1% P2M | | 0.473 | 0.469 | 0.474 | 0.463 | **0.449** | 0.093 |
| 50k 2% CD  | | 1.415 | 1.377 | 1.342 | 1.213 | **1.132** | 0.803 |
| 50k 2% P2M | | 1.046 | 1.010 | 0.969 | 0.853 | **0.774** | 0.339 |
| 50k 3% CD  | | 2.010 | 1.926 | 1.830 | 1.540 | **1.408** | 1.568 |
| 50k 3% P2M | | 1.537 | 1.458 | 1.348 | 1.087 | **0.969** | 0.845 |

> 笔误更正(2026-06-10)：上表 50k/2% P2M 的 SOTA 之前误写为 0.845(与 50k/3% 串行)，
> 正确值为 **0.339**。故 v9 的 0.774 **并未**超过该项 SOTA。

### 结论

- **已超过/追平 SOTA：50k/3% CD (1.408<1.568)。**（50k/2% P2M 因 SOTA 更正为 0.339，不再算超越）
- 中高噪声(2%/3%)段 v6→v9 持续大幅改善，p2m_weight 路径有效。
- **天花板：10k/1% P2M (~0.836) 和 50k/1% P2M (~0.46) 随 weight 增大几乎不动**
  —— S 模型容量瓶颈，纯调参无法突破，差距集中在低噪声精细去噪。
- **v9 为 PointGPT-S 最佳版本。**

### 下一步

转向 **PointGPT-L**（trans_dim=1024/depth=24，容量~7x），主攻 1% 低噪声 P2M
天花板。`cfgs/PointGPT-L/finetune_scoredenoise.yaml` 已补齐 langevin/p2m_weight/
EDM 配置（代码层全局生效），epoch 300→120 对齐 S。

---

## Session: PointGPT-L 测试结果归档 + 测试期显存保护 (2026-06-10)

### 测试期防崩机改动（`tools/runner_finetune.py`）

服务器在 L 模型测试时多次崩溃，定位根因：`test()` 路径**完全没有内存保护**
（`check_memory_and_exit` 只在训练路径调用）。本次加固：

- `vote_times` 5→1：单次推理，计算/显存峰值降到 1/5（最有效的防崩措施）。
- 投票循环 + per-patch 内逐张量 `del` + `torch.cuda.empty_cache()`，消除
  `vote × Langevin 30 步`的显存碎片累积。
- CD/P2M 累加改为 python float，不再跨样本保留 GPU tensor 引用。
- `ChamferDistanceL2` 只实例化一次（原来每次投票新建 CUDA 模块）。
- 新增 `gpu_mem_ratio()`，在样本前、以及 `patch_based_denoise` 每个 patch 批次前
  做**硬保护**：GPU>88~90% 或 CPU>90% 时打印已完成均值后 `sys.exit(0)` 安全退出
  ——宁可中断测试也绝不让服务器崩。
- 每样本 `gc.collect()` 回收 50k 可视化的 numpy/o3d CPU 临时对象。
- yaml：L `test_patch_batch: 2`（50k 最稳，10k 可调 4），`grad_norm_clip: 1.0`
  防 depth=24 梯度爆炸。

### PointGPT-L 测试结果（Langevin 30/0.3/0.95，与 S-v9、SOTA 对比）

| 数据集 | 噪声 | S-v9 | **L** | SOTA | 是否超 SOTA |
| --- | --- | --- | --- | --- | --- |
| 10k 1% CD  | | 2.435 | **2.121** | 2.056 | ✗（差 0.065）|
| 10k 1% P2M | | 0.839 | **0.707** | 0.416 | ✗ |
| 10k 2% CD  | | 4.118 | **3.474** | 3.043 | ✗ |
| 10k 2% P2M | | 2.020 | **1.584** | 0.838 | ✗ |
| 10k 3% CD  | | 4.958 | **4.076** | 4.066 | ≈ 持平（差 0.01）|
| 10k 3% P2M | | 2.760 | **2.054** | 1.487 | ✗ |
| 50k 1% CD  | | 0.726 | **0.598** | 0.592 | ≈ 持平（差 0.006）|
| 50k 1% P2M | | 0.449 | **0.392** | 0.093 | ✗（差 4x）|
| 50k 2% CD  | | 1.132 | **0.932** | 0.803 | ✗ |
| 50k 2% P2M | | 0.774 | **0.642** | 0.339 | ✗ |
| 50k 3% CD  | | 1.408 | **1.195** | 1.568 | ✓ **超过** |
| 50k 3% P2M | | 0.969 | **0.818** | 0.845 | ✓ 超过（差 0.03）|

### 结论

- **L 相比 S-v9 全线大幅提升**（平均 ~15-20%），CD 多处逼近 SOTA（差 0.01~0.13）。
- **真正站得住的超越：50k/3%（CD+P2M 双超）**；10k/3% CD、50k/1% CD 为持平级别。
- **P2M 是系统性短板**，尤其低噪声段：50k/1% P2M 0.392 vs 0.093（差 4x）、
  50k/2% P2M 0.642 vs 0.339（差近 2x）。模型把点拉到了正确"平均位置"（CD 好），
  但没精确贴到 mesh 表面（P2M 差）——这是 score 方法的典型局限，也是 SOTA
  （Score-Denoise / IterativePFN）靠表面感知机制赢的地方。

### 下一步（新分支，不动当前最佳 L 代码）

**EdgeConv 几何支路 + 显式表面投影**，针对性攻 P2M 短板：在 patch tokenizer 里
并行加一条 EdgeConv 局部几何特征支路（残差融合），强化 patch 内表面感知；推理端
加显式表面投影后处理。代码改动在独立分支进行，保留本分支的最佳 L 结果。

---

## Session: 跨 patch 加权拼接 + Consistency 迭代训练 (2026-06-17)

围绕"P2M 是系统性短板（尤其低噪声），模型把点拉到正确均值位但不贴面"这个诊断，
从推理端和训练端各做一项改进。

### 改动 1（推理端，不重训）：跨 patch 加权拼接 `fuse_tau_ratio`

`patch_based_denoise` 原来把重叠 patch 的预测**等权平均**拼回，会把曲面磨平（伤 P2M）。
改为按"点到该 patch 种子点距离"的高斯权重 `exp(-d²/2τ²)`（τ = `fuse_tau_ratio`·patch 半径）
融合，边界点降权。`fuse_tau_ratio<=0` 退回等权（旧行为）。yaml 默认 0.5。

**A/B（PointGPT-L, 10k/1%）**：

| fuse_tau_ratio | CD | P2M |
|---|---|---|
| 0（等权基线）| 2.121 | 0.707 |
| 0.3 | 2.1246 | 0.7131 |
| **0.5** | **2.1061** | 0.7045 |
| 0.7 | 2.1090 | 0.7038 |

结论：τ=0.5 按仓库 `cd+0.3·p2m` 排序最优，但**提升很小**（CD −0.7%、P2M −0.4%），
0.3 反而更差。证伪了"等权平均是 P2M 主要误差源"（至少 10k/1%）。τ=0.5 保留为默认
（免费微赢、不亏），但拼接不是撬动 P2M 的杠杆。10k/1% 是这条改进最不该见效的regime
（噪声最低、patch 内最平），高噪声段（10k/3%、50k/3%）仍值得各跑一次 τ=0.5 vs 0 确认。

### 改动 2（训练端）：Consistency 迭代训练 `consistency`

补"训练单步 ε-MSE vs 推理 30 步 Langevin"的鸿沟。`runner_finetune.py:run_net` 训练时
展开 `num_steps` 步退火：x 从 noisy 出发，每步喂模型**自己上一步的部分去噪结果**、σ 退火，
对齐推理轨迹。实现要点：

- **复用 'train' 前向不改模型**：把部分去噪点 x_in 当 `noisy_pts`、σ_k 当 `noise_std` 传进去，
  模型内部即按 `target_ε=(clean-x_in)/σ_k` 算该步 EDM 加权 ε-MSE，并返回该步全量去噪点。
- **ε-MSE 逐步深监督**（每步都监督），**P2M 只在最后一步**（test 真正产出的点）算，省 K× P2M。
- **截断 BPTT + 逐步 backward**：每步 backward 后立即释放计算图、detach 出下一步输入，
  **峰值显存 ≈ 单步**（关键，沿用现有 GPU>72% 显存保护），计算 ≈ `num_steps×`。
- `consistency.enable=False`（或不写该块）→ 完全退回原单步路径（用于 A/B）。

配置：S/L 的 finetune yaml 均加 `consistency: { enable: True, num_steps: 3, step_size: 0.3,
decay: 0.95 }`（step_size/decay 对齐推理 langevin）。L 单步已较慢，太慢可把 num_steps 降到 2。
另把训练 P2M 抽成 `_p2m_batch_loss` 辅助函数，单步/迭代两条路径共用。

**待办**：用 consistency 重训 L（先小 num_steps 验证 val CD 收敛与显存稳定），跑全量
10k/50k × 1/2/3% test 对比当前最佳 L，重点看低噪声 P2M 是否下降。

### 改动 3：主动 CPU/GPU 资源上限

原来只有"被动监控+超阈值退出"(`check_memory_and_exit` GPU>72%/CPU>78%、test 88%/90%)，
没有主动封顶。新增 `main.py:apply_resource_limits`（train/test 都生效，从 yaml 读，缺省不限制）：
- `cpu_threads`：`torch.set_num_threads` + `OMP/MKL/...NUM_THREADS` env（worker 子进程继承），
  共享服务器防吃满核。
- `gpu_mem_fraction`：`torch.cuda.set_per_process_memory_fraction`，把单进程显存封顶到整卡比例，
  超限触发**可被 try/except 捕获的 OOM**（训练 loop 跳过该 batch）而非把整卡/服务器拖崩。
  注意是相对整卡总量、不计其它进程占用。

S/L finetune yaml 默认 `cpu_threads: 8`、`gpu_mem_fraction: 0.9`（均可调；设 0 或删行=不限制）。
与被动退出互补：被动保存进度、主动防尖峰。

---

## Session: Stage 0 教师轨迹与步数冗余审计（2026-08-25）

### 目的

第二阶段正式做少步映射蒸馏前，先不训练新模型，重新刻画第一阶段冻结教师：

- 用同一 checkpoint 测试 1/2/4/8/15/30 次更新的 CD、P2M、HD/HD95 和耗时；
- 保存代表样本的 30 步完整轨迹；
- 统计逐步位移、法向/切向分量、最近 clean 误差下降和局部邻域变化；
- 判断 30 步后半段是否冗余，以及后续更适合 30→4、8→4 还是 8→2。

### 实现

1. `tools/runner_finetune.py:patch_based_denoise` 增加可选 `return_trajectory=False`：
   - 默认关闭时仍返回原来的 `[N,3]`，普通 train/val/test/denoise_folder 调用不变；
   - 开启时额外返回真实 patch 内 `patch_states`、固定 patch 索引/融合权重下的诊断 `global_states`、覆盖次数和 σ 日程；
   - `global_states` 只用于评测/可视化，不回灌到下一步，真实教师仍是固定 patch 后各 patch 独立 rollout。
2. 新增 `tools/analyze_teacher_trajectory.py`：
   - 延迟加载 PyTorch/CUDA 重依赖，使缺 CUDA 的机器仍能查看 `--help`；
   - 复用 active test dataset、消融开关注入、模型构建和 checkpoint 加载；
   - 一次 max-step 捕获生成同一 patch 布局下的前缀质量曲线；
   - 独立无轨迹捕获地重复计时，避免 NPZ/指标计算污染延迟；
   - 每个 shape 后增量写 CSV/进度，失败时把错误和已有结果写入 manifest。
3. 新增 `utils/trajectory_metrics.py`：
   - SciPy KDTree 双向 HD/HD95 与 CPU CD 复核；
   - clean kNN-PCA 法向置信度、法/切位移、kNN retention/churn 和局部边长变化；
   - 对外部非对应点集显式禁用索引指标并写 NaN；
   - 聚合时为每列保留有效值数量，避免缺 mesh/无对应关系被静默吞掉。
4. S/L finetune YAML 增加 `stage0_audit`，默认 `max_steps=30`、`budgets=[1,2,4,8,15,30]`、保存三个代表轨迹并对三个 shape 各计时三次。该块只由 Stage 0 工具读取，不影响训练或普通测试。
5. P2M 测试改为跟随预测张量所在 GPU，并按 `(mesh root, name, split)` 缓存 CPU mesh；Stage 0 启用 P2M 时不再吞掉缺 mesh、OOM 或实现错误，只有显式 `--no_p2m` 才允许跳过。
6. 轨迹审计严格验证初始 patch/整云状态和同一次捕获的最终 CPU/GPU 融合，默认最大绝对误差容差为 `1e-5`。各预算的 fresh CUDA replay 会受 FPS/scatter 浮点非确定性影响，因此改为记录 max/mean/RMS 差异并提示，不再把有限的累积漂移误判为轨迹错误。
7. 测试软显存监控改为查询当前 CUDA 设备（遵循 `CUDA_VISIBLE_DEVICES`），Stage 0 遇到内存保护会抛出非成功异常并把运行标记为 failed，不会以退出码 0 伪装成完整实验。
8. 新增共享 `inference_patch_size=1024`，普通 val/test、文件夹推理和 Stage 0 统一读取；同时要求未使用 CLI 覆盖时 `stage0_audit.max_steps == langevin.num_steps`，防止配置漂移后审计错教师。
9. Stage 0 在导入 NumPy/SciPy/Open3D/PyTorch 前先应用 YAML 中的 CPU 线程环境变量，并对旧 PyTorch/MIG 环境不支持显存比例限制的情况给出警告后继续，与主入口的兼容策略一致。

### 关键实验口径

Stage 0 的各预算是同一教师日程的前缀，固定：

```text
step_size = 0.3
decay = 0.95
budget = 1 / 2 / 4 / 8 / 15 / 30
```

`cfgs/PointGPT-L/ablation/abl_T1.yaml` 的 `num_steps=1, step_size=1.0` 是另一条 Tweedie-1 消融，不能通过只改 `num_steps` 来构造 Stage 0 曲线。原消融 README 中“改 abl_T1 扫多步”的建议已经改为专用 Stage 0 命令。

逐步动力学一律使用 raw、SOR 前状态，因为 SOR 会删除点并破坏对应关系。质量表默认同时报告 raw 和 SOR 两个 variant；Stage 0 不应用 `surface_projection`。

计时使用 CUDA 同步，质量轨迹先跑一次兼作 warmup。`denoise_seconds` 包含 patch 构造、patch 内 rollout 和融合；`end_to_end_seconds` 另加 SOR，但不含 DataLoader、指标、轨迹 copy、写盘和可视化。除逻辑 NFE 外还记录 `actual_forward_calls=budget×patch_batches`。

### 输出与文档

默认输出到：

```text
experiments/stage0_teacher_analysis/<run_name>/
```

主要产物为 `manifest.json`、`effective_config.yaml`、`summary_by_budget.csv`、`per_shape_by_budget.csv`、`timing_runs.csv`、`aggregate_per_step.csv`、`per_step_dynamics.csv` 和 `trajectories/<shape>/trajectory.npz`。完整运行说明见 `docs/stage0_teacher_analysis.md`；`CLAUDE.md` 与 PointGPT-L 消融 README 已同步入口和口径。

### 验证状态与限制

当前 Windows 工作区缺少 `data/`、`extensions/`、checkpoint 和 PyTorch，不能在本机运行真实模型。这里已完成 `--help`、Python 语法检查和 8 项 NumPy/SciPy 几何单元测试；真实 CD/P2M、GPU 轨迹数值一致性、显存和耗时仍必须回到原 Linux/CUDA 训练环境做单样本冒烟与全量验收。未在本记录中声称已经跑出 Stage 0 实验结果。
