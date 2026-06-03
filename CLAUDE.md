# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo actually is

This is a **fork of [PointGPT](https://arxiv.org/abs/2305.11487) repurposed for point-cloud denoising** on the PUNet / ScoreDenoise pipeline. The upstream `README.md` and `DATASET.md` describe classification (ModelNet40, ScanObjectNN), few-shot, and part-segmentation tasks — **most of those workflows do not run as-is in this fork**:

- `cfgs/dataset_configs/` only contains `ScoreDenoise.yaml` (no ShapeNet-55, ModelNet, ScanObjectNN, etc.).
- `cfgs/PointGPT-S/` has `pretrain.yaml` (ShapeNet-55, references a missing dataset config) and `finetune_scoredenoise.yaml`.
- `cfgs/PointGPT-L/` has only `finetune_scoredenoise.yaml`.
- `cfgs/PointGPT-B/` is absent.

Treat the upstream README as historical / reference material. Active development centers on `datasets/ScoreDenoiseDataset.py`, `tools/runner_finetune.py`, `utils/p2m_loss.py`, and `models/PointGPT.py`.

## Entry point and command dispatch

Everything goes through `main.py`. The dispatch logic (`main.py:85-91`) is:

| Flag combination | Runner invoked |
|---|---|
| (default) | `tools.runner_pretrain.run_net` |
| `--finetune_model` or `--scratch_model` | `tools.runner_finetune.run_net` |
| `--test` (+ `--ckpts <path>` required) | `tools.runner_finetune.test_net` (or `tools.runner.test_net` via `main_vis.py`) |

Typical denoising fine-tune run:
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
  --config cfgs/PointGPT-S/finetune_scoredenoise.yaml \
  --finetune_model \
  --exp_name <name> \
  --ckpts pretrainModel/S/pretrained.pth
```

DDP: pass `--launcher pytorch` (and launch via `torch.distributed.launch`/`torchrun`). Single-GPU uses `--launcher none` (default).

## Output layout (auto-created by `utils/parser.py`)

```
experiments/<config_stem>/<parent_stem>/<exp_name>/
    ├── config.yaml          # copied from --config
    ├── <timestamp>.log
    ├── ckpt-last.pth, ckpt-best.pth, ...
experiments/<config_stem>/<parent_stem>/TFBoard/<exp_name>/{train,test}/
```
`--test` prefixes the exp name with `test_`. `--mode <easy|median|hard>` suffixes it.

## Config system

YAML loaded via `utils/config.py:cfg_from_yaml_file` into `EasyDict`. Supports recursive `_base_:` includes — used by `cfgs/PointGPT-*/finetune_scoredenoise.yaml` to pull in `cfgs/dataset_configs/ScoreDenoise.yaml`. Batch sizes are rewritten at runtime in `main.py:46-60` from `config.total_bs` divided by world size — **set `total_bs` in the yaml, not per-rank `bs`.**

## Dataset wiring (the non-obvious parts)

- `datasets/build.py` provides a `DATASETS` registry, but the active denoising path **does not use it**. `tools/builder.py:dataset_builder` instantiates `datasets.ScoreDenoiseDataset.ScoreDenoise` (a `pl.LightningDataModule`) directly and calls `.train_dataloader()` / `.val_dataloader()` / `.test_dataloader()`.
- Expected on-disk layout (root from `cfgs/dataset_configs/ScoreDenoise.yaml` → `data/ScoreDenoise/`):
  ```
  data/ScoreDenoise/PUNet/
      ├── meshes/{train,test}/<name>.off
      └── pointclouds/{train,test}/{10000,30000,50000}_poisson/<name>.xyz
  data/ScoreDenoise/examples/pointclouds/test/PUNet_10000_poisson_0.01/<name>.xyz   # noisy test pairs
  ```
- Mesh `.off` files are read at runtime by `tools/runner_finetune.py:compute_mesh_normals_for_pcl` (path hard-coded as `data/ScoreDenoise/PUNet/meshes/<split>/<name>.off`) for the P2M loss. **Removing/renaming meshes breaks fine-tuning, not just metric reporting.**
- The noisy test directory `examples/pointclouds/test/PUNet_10000_poisson_0.01` is hard-coded in `datasets/ScoreDenoiseDataset.py:680` — change it there if your eval set differs.

## Models

`models/build.py` provides a `MODELS` registry; `models/PointGPT.py` registers everything via `@MODELS.register_module()`. The yaml `model.NAME` field selects the class:

- `PointGPT` — pre-training (mask + auto-regressive generation).
- `PointTransformer` — fine-tuning backbone used by `finetune_scoredenoise.yaml`.

`models/GPT.py` defines `GPT_extractor` / `GPT_generator` used inside `PointGPT.py`. `models/z_order.py` provides Morton-code ordering for the auto-regressive prediction order.

## Dependencies that pip alone does not install

`requirements.txt` is incomplete. The fine-tune path additionally requires:

```bash
# CUDA ops (must be compiled in-tree; extensions/ is gitignored)
cd extensions/chamfer_dist && python setup.py install --user
cd extensions/emd          && python setup.py install --user

# PointNet++ ops (vendored source also exists in Pointnet2_PyTorch-master/, but installs via pip)
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# CUDA kNN
pip install https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# Implicit imports not in requirements.txt
pip install pytorch3d pytorch_lightning trimesh scikit-learn
# `chamfer` (separate CUDA module, imported in models/PointGPT.py)
```

If `extensions/` is missing it has to be re-pulled from upstream PointGPT — it's listed in `.gitignore`.

## Loss & metric specifics for denoising

- `tools/runner_finetune.py:DenoiseMetrics.better_than` ranks checkpoints by `cd + 0.3 * p2m`. The `0.3` weight is calibrated against the empirical CD ≈ 1.7 vs P2M ≈ 5–7 scale gap — adjust together if you re-scale either term.
- `compute_mesh_normals_for_pcl` caches results in the module-level `_mesh_normal_cache` dict; clearing it requires restarting the process or manually `del`-ing the entry.
- `check_memory_and_exit` (`tools/runner_finetune.py:152`) will save `ckpt-last` and `sys.exit(0)` if GPU memory > 80% or CPU memory > 85% — looks like a "graceful OOM" rather than a bug.

## Conventions worth keeping

- Code comments in this fork are predominantly Chinese — match the language of nearby comments when editing.
- `tools/runner_finetune.py` contains a lot of commented-out transforms (`RandomRotate`, `CleanScaleTranslate`, …); they are intentionally disabled because they break mesh/point-cloud coordinate alignment used by P2M loss. Re-enabling them needs corresponding mesh-side transforms.
- No automated test suite or linter config exists. There is no `pytest`/`unittest` to run.
