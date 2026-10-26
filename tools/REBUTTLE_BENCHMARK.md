# PUNet inference cost and quality sweep

Run on one CUDA GPU from the repository root, with the same fine-tuned
`ckpt-best.pth` for every setting. This script does not train or alter weights.
It uses the existing `patch_based_denoise` implementation and the PUNet test
clouds at 10K/50K and 1%/2%/3% Gaussian noise.

```bash
CKPT=/absolute/path/to/ckpt-best.pth
for res in 10000 50000; do
  for noise in 0.01 0.02 0.03; do
    CUDA_VISIBLE_DEVICES=0 python tools/benchmark_inference.py \
      --config cfgs/PointGPT-L/finetune_scoredenoise.yaml \
      --checkpoint "$CKPT" \
      --data-root data/ScoreDenoise \
      --resolution "$res" --noise "$noise" \
      --steps 1 5 10 20 30 --expected-clouds 20 \
      --output-dir "experiments/rebuttle_cost/${res}_${noise}"
  done
done
```

If interrupted, repeat the corresponding command with `--resume`. The script
writes each completed cloud to `per_cloud.csv` immediately, checks all paired
cloud and mesh names, and writes `summary.csv` only after all T values finish.
`run_info.json` records the checkpoint, hardware, software, and settings.
Adjust `--expected-clouds` if the actual PUNet release contains a different
number of test shapes; do not silently evaluate a subset.

## Measurement definition

- The timer starts after the noisy cloud is on the GPU, and ends after GPU
  synchronization. It includes FPS/KNN tiling, all T network updates, and
  cross-patch fusion. It excludes data loading, CD/P2M, mesh loading, and file
  output. One untimed warm-up per T is the default.
- `peak_allocated_gib` is the per-cloud peak PyTorch GPU allocation during the
  timed call, including model weights and the input/output tensors. Report the
  maximum over test clouds. Other GPU processes should be stopped while
  measuring. The script disables only the `nvidia-smi` subprocess safety check
  inside `patch_based_denoise`; the inference calculation is unchanged.
- `clouds_per_s = n_clouds / sum(latency_s)` and
  `points_per_s = sum(n_points) / sum(latency_s)`. These are inference-only
  throughputs on a single GPU. `model_forwards_per_cloud` counts calls to the
  patch batch model: `ceil(floor(3N/1024) / patch_batch) * T`.
- CD and P2M use the same normalization and metric calls as the existing
  `test()` path, outside the timed region. SOR and surface projection are off,
  matching the no-post-processing Gaussian tables in the paper.
- All rows use the same `step_size` and `decay` from the model config (normally
  0.3 and 0.95). Thus T=1 is one **partial** update. To reproduce the paper's
  separate single-step Tweedie ablation, run `--steps 1 --step-size 1.0` in a
  separate output directory and label that row explicitly.

For the rebuttal, show CD, P2M, mean latency (with variation across clouds),
clouds/s, and peak allocated GiB for each T. Plot CD against latency for 10K
and 50K, with one curve per noise level. State the GPU, patch batch size, and
timing boundary alongside the table. If runtime is costly, the six independent
commands can be scheduled separately; do not substitute a small timing subset
for the full-set quality results without saying so.

## Directional P2M and inference-parameter sensitivity

The script now writes `point_to_face_x1e4`, `face_to_point_x1e4`, and their
sum `p2m_x1e4` alongside `cd_x1e4` in `per_cloud.csv`. `summary.csv` contains
the corresponding per-cloud means. Both directional terms use the same
unit-sphere mesh normalization and the same squared-distance reductions as
PyTorch3D's original bidirectional P2M. The regular `main.py --test` log also
prints these four metrics per cloud and as final means. Use new output
directories: older `per_cloud.csv` files do not have the directional columns
and cannot be resumed with this version. The benchmark bypasses SOR for the
Gaussian protocol; the checked-in L-model yaml has `sor_enable: True`, so
disable that option in a copy of the config if using `main.py --test` to
compare against these benchmark results.

For an inference-only, one-factor-at-a-time sensitivity study, use the same
fine-tuned checkpoint on representative PUNet Gaussian conditions 10K/1%
and 50K/3%. Keep `T=30`, SOR off, and the other two parameters at their
baseline values while varying one of step size `eta`, decay `gamma`, or
patch overlap `r`. These seven settings include one shared baseline:

| Setting | `--step-size` (eta) | `--decay` (gamma) | `--seed-ratio` (r) |
|---|---:|---:|---:|
| baseline | 0.3 | 0.95 | 3 |
| eta_0.2 | 0.2 | 0.95 | 3 |
| eta_0.4 | 0.4 | 0.95 | 3 |
| gamma_0.90 | 0.3 | 0.90 | 3 |
| gamma_0.98 | 0.3 | 0.98 | 3 |
| r_2 | 0.3 | 0.95 | 2 |
| r_4 | 0.3 | 0.95 | 4 |

```bash
CKPT=/absolute/path/to/ckpt-best.pth
DATA_ROOT=/absolute/path/to/ScoreDenoise
for setting in "10000 0.01" "50000 0.03"; do
  read -r res noise <<< "$setting"
  for variant in \
    "baseline 0.3 0.95 3" \
    "eta_0.2 0.2 0.95 3" "eta_0.4 0.4 0.95 3" \
    "gamma_0.90 0.3 0.90 3" "gamma_0.98 0.3 0.98 3" \
    "r_2 0.3 0.95 2" "r_4 0.3 0.95 4"; do
    read -r label eta gamma overlap <<< "$variant"
    CUDA_VISIBLE_DEVICES=0 python tools/benchmark_inference.py \
      --config cfgs/PointGPT-L/finetune_scoredenoise.yaml \
      --checkpoint "$CKPT" --data-root "$DATA_ROOT" \
      --resolution "$res" --noise "$noise" --steps 30 \
      --step-size "$eta" --decay "$gamma" --seed-ratio "$overlap" \
      --expected-clouds 20 \
      --output-dir "experiments/rebuttle_sensitivity/${res}_${noise}/${label}"
  done
done
```

Each directory contains its own `run_info.json`, `per_cloud.csv`, and
`summary.csv`. Compare CD, both directional P2M terms, total P2M, latency,
and peak memory. `r` changes the number of patches and thus inference cost.
If the actual test release has a different number of shapes, set
`--expected-clouds` to that verified count. Re-run an interrupted command
with the same settings/output directory plus `--resume`.
