# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo actually is

A **fork of [PointGPT](https://arxiv.org/abs/2305.11487) repurposed for point-cloud denoising** on the PUNet / ScoreDenoise pipeline. The upstream `README.md` / `DATASET.md` describe classification (ModelNet40, ScanObjectNN), few-shot, and part-segmentation — **treat them as historical reference; those workflows do not run as-is in this fork.** Evidence:

- `cfgs/dataset_configs/` contains only `ScoreDenoise.yaml`. The pre-train config (`cfgs/PointGPT-S/pretrain.yaml`) references `cfgs/dataset_configs/ShapeNet-55.yaml`, **which does not exist** — so pre-training cannot run without restoring that dataset config + data.
- The only runnable configs are `cfgs/PointGPT-{S,L}/finetune_scoredenoise.yaml`.
- `segmentation/` is upstream code, untouched and unused by the denoising path.

Active development lives in `datasets/ScoreDenoiseDataset.py`, `datasets/scoredenoise/transforms.py`, `tools/runner_finetune.py`, `models/PointGPT.py` (`PointTransformer`), and `utils/p2m_loss.py`.

## Entry point and command dispatch

Training and the ordinary validation/test path go through `main.py`. Dispatch (`main.py:117-124`, aliases resolved in `tools/__init__.py`):

| Flags | Runner |
|---|---|
| (default) | `tools.runner_pretrain.run_net` — pre-training, **not runnable** (missing ShapeNet-55 config) |
| `--finetune_model` or `--scratch_model` | `tools.runner_finetune.run_net` — the active training path |
| `--test` (requires `--ckpts <path>`) | `tools.runner_finetune.test_net` |

`main_vis.py` is a separate entry that only supports `--test` and dispatches to `tools.runner.test_net` (the old upstream tester) — not the active denoising tester.

Stage 0 teacher-trajectory analysis is another standalone entry: `tools/analyze_teacher_trajectory.py`. It loads the active test dataset and a frozen fine-tuned checkpoint, captures the current teacher's 30-step patch trajectory, and reports the 1/2/4/8/15/30 prefix quality/runtime curve. It does not train or modify weights. Full usage and metric semantics are documented in `docs/stage0_teacher_analysis.md`.

Typical fine-tune run:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config cfgs/PointGPT-S/finetune_scoredenoise.yaml \
  --finetune_model --exp_name <name> \
  --ckpts pretrainModel/S/pretrained.pth
```
Typical test run (computes CD + P2M on the full test clouds, writes visualizations):
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config cfgs/PointGPT-S/finetune_scoredenoise.yaml \
  --test --exp_name <name> \
  --ckpts experiments/finetune_scoredenoise/PointGPT-S/<name>/ckpt-best.pth
```
Typical Stage 0 run (same checkpoint, fixed `step_size=0.3` / `decay=0.95` teacher prefixes):
```bash
CUDA_VISIBLE_DEVICES=0 python tools/analyze_teacher_trajectory.py \
  --config cfgs/PointGPT-L/finetune_scoredenoise.yaml \
  --ckpt experiments/finetune_scoredenoise/PointGPT-L/<name>/ckpt-best.pth \
  --run_name L_10k_1pct
```
DDP: `--launcher pytorch` via `torchrun`/`torch.distributed.launch`. Single-GPU is `--launcher none` (default).

## The denoising algorithm (the part most worth reading)

The backbone does **ε-prediction (DDPM-style noise prediction)**, *not* score estimation — despite leftover "score" naming (`project_patch_scores_weighted`, `pred_score_global`). Read those identifiers as "ε". The whole flow is in `models/PointGPT.py:PointTransformer.forward` plus `tools/runner_finetune.py`.

**Forward pass** (`PointTransformer.forward`, signature `(noisy_pts, clean_pts=None, type='val', name='', epoch=0, max_epoch=None, noise_std=None)`):
1. `group_divider` (FPS centers + KNN) splits the input into `num_group × group_size` patches; encoder → tokens; GPT `blocks` (encoder) + `generator_blocks` (decoder) produce a `[B, G, M, 3]` per-patch-point vector field.
2. `project_patch_scores_weighted` fuses per-patch vectors to a global `[B, N, 3]` field by Gaussian distance-to-center weighting. **It does NOT add the patch center** (the output is a direction/ε vector, not a coordinate) — the key difference from `project_patch_predictions_weighted`.
3. The fused field is `ε`. Single-step denoise is `x̂ = x + σ·ε`.

**Attention is fully bidirectional during fine-tuning** (`attn_mask=None`) — the pre-trained causal/autoregressive mask is dropped because denoising has no generation order.

**Training loss** (`type='train'`, returns `(loss, denoised)`):
- Target `ε = (clean − noisy) / σ` (per-sample σ from `noise_std`).
- ε-MSE with **EDM 1/σ² weighting**: `((pred_ε − target_ε)² · (σ_ref/σ)²).sum(-1).mean()`, `σ_ref = 0.01`. Small-noise samples (σ≈0.01) keep full gradient; large-noise samples are down-weighted, pushing precision on fine denoising.
- The runner then adds an **optional differentiable P2M term**: `total = ε_MSE + p2m_weight · P2M` (`runner_finetune.py:run_net`). `p2m_weight` from the yaml (S: 0.70, L: 0.30; 0 disables). P2M comes from `utils/p2m_loss.py:compute_p2m_train` — restores the predicted patch to mesh world coords (per-sample `center`/`scale`), normalizes to the mesh's unit sphere, and takes the differentiable one-directional point→face distance (scaled `×1e4` to match test-time P2M magnitude).
- The training log prints only the ε-MSE term (`losses`) and P2M separately (`p2m_meter`); backward is on their weighted sum.
- **Consistency / iterative training** (opt-in via `config.consistency.enable`): instead of one ε step, the runner unrolls `num_steps` Langevin steps at train time — each step feeds the model its own partially-denoised output at the decayed σ (reusing the `'train'` forward: passing the partial point as the "noisy" arg makes it compute the right per-step target `(clean − x_in)/σ_k`). ε-MSE is deep-supervised on every step; P2M only on the final step. Uses **truncated BPTT (`detach` between steps) + per-step `backward`**, so peak memory ≈ single-step while compute is ≈`num_steps×`. This closes the "train 1 step vs test 30 steps" gap (IterativePFN-style). `enable=False` (or block absent) is the original single-step path — keep it for A/B.

**Inference** (`type='val'`, val + test): **multi-step Langevin annealing is implemented** in `tools/runner_finetune.py:patch_based_denoise`:
```
x ← noisy
for t in range(num_steps):
    ε   = (model(x, σ_t) − x) / σ_t     # re-estimate each step
    x   ← x + step_size · σ_t · ε
    σ_t ← σ_t · decay
```
`num_steps=1, step_size=1.0` degenerates to single-step Tweedie. Hyperparameters come from `config.langevin` (calibrated optimum baked into the yamls: `num_steps=30, step_size=0.3, decay=0.95`; `step_size ≥ 0.5` overshoots/diverges).

**Patch coverage is the recurring subtlety** — `group_divider` only covers ~`num_group·group_size / N` of the points, so feeding a full 10k cloud directly leaves ~80% of points with ε≈0 (unmoved). The fix is applied in three places that must stay consistent:
- **Train**: dataset cuts a single 1024-point KNN patch per sample → 64×32/1024 ≈ 200% coverage.
- **Val**: `val_dataloader` supplies full deterministic validation clouds; `validate()` sends them through the same overlapping-patch `patch_based_denoise` path as test, then computes full-cloud CD + P2M.
- **Test**: `patch_based_denoise` tiles the full cloud into overlapping 1024-patches (FPS seeds + KNN), denoises each, and fuses overlapping predictions back with a **distance-to-seed Gaussian weight** (`fuse_tau_ratio` in the yaml, default 0.5): a point's contribution from a patch is weighted `exp(−d²/2τ²)`, τ = ratio·patch-radius, so each patch's boundary points are down-weighted and overlap-fusion doesn't blur curved surfaces (targets P2M). `fuse_tau_ratio ≤ 0` reverts to the old equal-weight averaging. Uncovered points fall back to noisy.

`patch_based_denoise(..., return_trajectory=True)` is an opt-in Stage 0 diagnostic. The default remains the original `[N,3]` return and allocates no trajectory. Diagnostic mode additionally returns exact per-patch `patch_states` and per-step fused `global_states`. The latter are readouts made with fixed patch indices/weights and are **not fed back** into the next step; do not describe them as the teacher's full-cloud Markov states. Step labels use `update_step=0 ↔ teacher_t=30` (noisy) and `update_step=30 ↔ teacher_t=0` (full teacher).

**Checkpoint-load output-head re-init (do not remove):** after `load_model_from_ckpt`, the runner re-initializes `generator_blocks.increase_dim[0]` to `std=0.1` weights / zero bias (`runner_finetune.py:run_net`). The pre-trained head generates *absolute coordinates*; reused directly it makes `x + σ·ε` drift wildly. `std=0.1` puts the initial ε magnitude (~2.0) near the target ε (~√3). The earlier `std=0.01` value left points barely moving — if "the model won't learn amplitude", check this.

**Encoder is intentionally NOT frozen** during fine-tuning (ε-estimation needs encoder adaptation away from the autoregressive pre-training objective).

**Post-processing (test only):** Statistical Outlier Removal (`sor_filter`, gated by `config.sor_enable`, default on) and an optional local-PCA surface projection (`local_surface_projection`, gated by `config.surface_projection.enable`, default off — PCA normals proved unreliable). Stage 0 always analyzes raw/SOR-free states for point-correspondence dynamics and, by default, reports SOR as a separate quality variant; it does not apply `surface_projection`.

## Models

`models/build.py` exposes a `MODELS` registry; `models/PointGPT.py` registers classes via `@MODELS.register_module()` and the yaml `model.NAME` selects one:
- `PointTransformer` — the active fine-tuning backbone (ε-prediction, described above).
- `PointGPT` — the pre-training model (masked + autoregressive patch generation); only used by the non-runnable `pretrain.yaml`.

`models/GPT.py` defines `GPT_extractor` / `GPT_generator` (the transformer encoder/decoder, each ending in an `increase_dim` Conv1d head). `models/z_order.py` provides Morton-code ordering used by `Group.morton_sorting` for the autoregressive patch order in pre-training.

## Metrics & checkpoint ranking

- CD and P2M are both reported `×1e4` (normalized-space). `DenoiseMetrics.better_than` ranks by `cd + 0.3·p2m` — the `0.3` offsets the empirical scale gap (CD ≈ 1.7 vs P2M ≈ 5–7). **Re-scale them together if you re-scale either term.**
- Current `validate()` runs the same full-cloud patch/Langevin/SOR path as test, computes both CD and P2M, and checkpoint ranking uses `cd + 0.3·p2m`. `VAL_NUM=0` uses the benchmark-style test split for validation; `VAL_NUM>0` uses held-out training shapes and their train meshes.
- Stage 0 additionally reports symmetric Euclidean HD/HD95 and their squared `×1e4` forms. Its `cpu_cd_sq_x1e4` is only a SciPy KDTree cross-check; `cd_x1e4` remains the official CUDA Chamfer metric.

## Config system

YAML → `EasyDict` via `utils/config.py:cfg_from_yaml_file`, with recursive `_base_:` includes (the finetune configs pull in `cfgs/dataset_configs/ScoreDenoise.yaml`). Batch sizes are rewritten at runtime from `config.total_bs` divided by world size (`main.py:46-60`), then `tools/builder.py:dataset_builder` propagates `config.others.bs` into `_base_.TRAIN_BATCH_SIZE` — **set `total_bs` in the yaml, never per-rank `bs`.**

Key denoising-specific yaml knobs (read at runtime; no code edit needed to change them):
- `cfgs/dataset_configs/ScoreDenoise.yaml`: `NOISE_MIN/MAX` (0.005/0.020 — keep fixed for fair SOTA comparison), `NOISE_LOG_UNIFORM` (log-uniform σ sampling), `TEST_RESOLUTION` / `TEST_NOISY_DIR` / `TEST_NOISE` (switch 10k/50k and noise level here), `TEST_NOISY_PATH` (optional, generalization: point the noisy input at any folder of `.xyz`, overriding the default `examples/.../<TEST_NOISY_DIR>` — clean PUNet test + meshes stay the reference, so noisy files must match by name/resolution), `PATCH_SIZE` (1024, aligned with pre-train npoints), `TRAIN_OVERSAMPLE` (per-cloud patch resampling per epoch — 120 clouds is far too few steps/epoch without it).
- Finetune yamls: `langevin`, `surface_projection`, `inference_patch_size` (shared by val/test/folder inference/Stage 0), `fuse_tau_ratio`, `p2m_weight`, `consistency` (iterative-training unroll: `enable`/`num_steps`/`step_size`/`decay`), `stage0_audit` (standalone teacher audit only), `test_patch_batch`, `mem_check_interval`, `grad_norm_clip`, `cpu_threads` / `gpu_mem_fraction` (proactive resource caps, see Memory guards).
- `stage0_audit.budgets=[1,2,4,8,15,30]` means prefixes of the same `langevin` schedule (`step_size=0.3`, `decay=0.95`). It is deliberately different from `ablation/abl_T1.yaml`, whose one-step Tweedie baseline uses `step_size=1.0`. Never sweep `abl_T1` to construct the Stage 0 prefix curve.
- Without an explicit Stage 0 CLI override, `stage0_audit.max_steps` must equal `langevin.num_steps`; mismatch is an error, preventing a stale audit block from silently measuring a different teacher.
- `val_interval` / `save_interval` in the yaml are inert — validation cadence uses `args.val_freq` (default 1) and `ckpt-last` is saved every epoch.

## Dataset wiring and on-disk layout

`datasets/build.py` has a `DATASETS` registry but the active path **bypasses it**: `tools/builder.py:dataset_builder` directly instantiates `datasets.ScoreDenoiseDataset.ScoreDenoise` (a `pl.LightningDataModule`) and calls `.train_dataloader()` / `.val_dataloader()` / `.test_dataloader()`. (`datasets/DMRSetDataset.py` is an unrelated, **un-wired** loader for the DMRDenoise benchmark — ignore it for the active pipeline.)

Expected layout (root = `data/ScoreDenoise/`, gitignored — must be supplied):
```
data/ScoreDenoise/PUNet/
    ├── meshes/{train,test}/<name>.off
    └── pointclouds/{train,val,test}/{10000,30000,50000}_poisson/<name>.xyz
data/ScoreDenoise/examples/pointclouds/test/<TEST_NOISY_DIR>/<name>.xyz   # noisy eval clouds
```
- **Train** loads all of `10000/30000/50000_poisson` (≈120 clean clouds) and **adds Gaussian noise on the fly** (`transforms.AddNoise`, σ sampled per sample from `[NOISE_MIN, NOISE_MAX]`). Each clean cloud is oversampled `TRAIN_OVERSAMPLE×` per epoch with a fresh random patch + σ.
- **Val** loads `val/10000_poisson` (falls back to `test/` if `val/` is absent) and adds noise at fixed `VAL_NOISE`.
- **Test** (`PairedEvalDataset`) pairs clean `test/<TEST_RESOLUTION>` clouds with **pre-noised** clouds from `examples/.../<TEST_NOISY_DIR>`; normalization is taken from the clean cloud.
- The data tuple through every collate path is `(noisy, clean, noise_std, center, scale, name)` (`denoise_collate_fn_test`). `noise_std` is `None` for the test set (no per-sample σ) and the runner falls back to `config...TEST_NOISE`.
- Train/val patch alignment: KNN is computed in **noisy space** and the same indices index both clean and noisy, guaranteeing point correspondence (`PairedPatchDataset.__getitem__`).
- **Mesh `.off` files are loaded at runtime** for both the training P2M loss (`utils/p2m_loss.py`, root overridable via the `PUNET_MESH_ROOT` env var, default `data/ScoreDenoise/PUNet/meshes`) and the test P2M (`compute_p2m`). Removing/renaming meshes breaks **fine-tuning**, not just metrics. P2M caches CPU meshes per `(absolute mesh root, name, split)` and follows the prediction tensor's device; clear the module cache by restarting the process.

## Output layout

```
experiments/<config_stem>/<parent_stem>/<exp_name>/
    ├── config.yaml, <timestamp>.log
    ├── ckpt-last.pth, ckpt-best.pth (, ckpt-best_vote.pth)
experiments/<config_stem>/<parent_stem>/TFBoard/<exp_name>/{train,test}/

experiments/stage0_teacher_analysis/<run_name>/
    ├── manifest.json, effective_config.yaml, progress.json
    ├── summary_by_budget.csv, per_shape_by_budget.csv, timing_runs.csv
    ├── aggregate_per_step.csv, per_step_dynamics.csv
    └── trajectories/<shape>/{trajectory.npz, metadata.json, *.xyz}
```
`--test` prefixes `exp_name` with `test_`; `--mode <easy|median|hard>` suffixes it. Ordinary `test()` still uses the hard-coded outer path `experiments/finetune_scoredenoise_L/PointGPT-Change/`, but its inner directory is now separated by `exp_name`, so different test runs no longer overwrite one another. Stage 0 uses its own output root shown above and records the effective config, Git state, checkpoint metadata, schedule and device in `manifest.json`.

## Memory guards (look like "graceful OOM", not bugs)

There are two complementary layers: **reactive monitor-and-exit guards** and **proactive hard caps**.

Reactive (the runner self-terminates rather than risk a server crash):
- `check_memory_and_exit` (train path) saves `ckpt-last` and `sys.exit(0)` when **GPU > 72%** or **CPU > 78%** (defaults). Called every `mem_check_interval` (≈50) batches, before validation, and after each epoch.
- `test()` has its own hard guard (GPU > 88% / CPU > 90%) before each sample and inside `patch_based_denoise`; on trip it prints the running mean and exits 0. If a test run dies "early but clean", lower `test_patch_batch` in the yaml.
- `vote_times` is pinned to 1 in `test()` (multi-vote was the main crash source).

Proactive (`main.py:apply_resource_limits`, applied to both train and test; yaml knobs, `0`/absent = no limit):
- `cpu_threads` → `torch.set_num_threads` + `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS` env (so DataLoader workers inherit it) — caps CPU core usage on a shared box.
- `gpu_mem_fraction` → `torch.cuda.set_per_process_memory_fraction` — caps this process's GPU allocation to a fraction of the card; a spike past it raises a **catchable** CUDA OOM (the train loop's `except OOM` skips the batch) instead of taking the whole card down. Note it's relative to *total* card memory, not free memory, so it doesn't account for other tenants.

## Dependencies pip alone does not install

`requirements.txt` is incomplete. The fine-tune path also needs:
```bash
# CUDA ops (compiled in-tree; extensions/ is gitignored — re-pull from upstream PointGPT if missing)
cd extensions/chamfer_dist && python setup.py install --user
cd extensions/emd          && python setup.py install --user
# PointNet++ ops (source also vendored in Pointnet2_PyTorch-master/)
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
# CUDA kNN
pip install https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
# implicit imports
pip install pytorch3d pytorch_lightning trimesh scikit-learn open3d
# `chamfer` (separate CUDA module imported in models/PointGPT.py, distinct from extensions/chamfer_dist)
```

## Conventions worth keeping

- Comments in this fork are predominantly **Chinese** — match the language of nearby comments when editing.
- Many transforms in `train_dataloader` / `train_transforms` (`RandomRotate`, `CleanScaleTranslate`, …) are **intentionally commented out**: rotating/scaling the noisy cloud breaks its alignment with the fixed mesh used by the P2M loss. Re-enabling any of them requires applying the matching transform to the mesh side too.
- `denoise_collate_fn` (the 5-tuple version without `noise_std`) is dead — all dataloaders use `denoise_collate_fn_test`.
- No test suite, linter, or formatter config exists.

## Session changelog

`CHANGES.md` (not gitignored) is a human-readable per-session log with design rationale, the v4–v9 / PointGPT-L tuning history, and full CD/P2M-vs-SOTA result tables. **Check it first when re-orienting** — it explains *why* the current ε-prediction + EDM-weighting + P2M + Langevin stack looks the way it does, and what was tried and rejected.
