# PUNet/PCNet inference cost, directional P2M, and sensitivity sweeps

Run on one CUDA GPU from the repository root, with the same fine-tuned
`ckpt-best.pth` for every setting. This script does not train or alter weights.
It uses the existing `patch_based_denoise` implementation. The original sweep
uses PUNet test clouds at 10K/50K and 1%/2%/3% Gaussian noise; the extended
sweep below also covers PUNet covariance/Laplace noise and PCNet Gaussian noise.

Before running on the server, check `git rev-parse --short HEAD` and confirm
that its `tools/benchmark_inference.py` accepts `--seed-ratio` and writes
`point_to_face_x1e4` and `face_to_point_x1e4`. An older checkout fails at
argument parsing. Use a fresh output directory for the directional schema;
older `per_cloud.csv` files cannot be resumed with this version.

```bash
CKPT=/absolute/path/to/ckpt-best.pth
DATA_ROOT=/absolute/path/to/ScoreDenoise
for res in 10000 50000; do
  for noise in 0.01 0.02 0.03; do
    CUDA_VISIBLE_DEVICES=0 python tools/benchmark_inference.py \
      --config cfgs/PointGPT-L/finetune_scoredenoise.yaml \
      --checkpoint "$CKPT" \
      --data-root "$DATA_ROOT" \
      --resolution "$res" --noise "$noise" \
      --steps 1 5 10 20 30 --expected-clouds 20 \
      --output-dir "experiments/rebuttle_directional_baseline/${res}_${noise}"
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
fine-tuned checkpoint on all six PUNet Gaussian conditions. Keep `T=30`, SOR
off, and the other two parameters at their baseline values while varying one
of step size `eta`, decay `gamma`, or patch overlap `r`. These seven settings
include one shared baseline. Its `T=30` row comes from the baseline sweep
above; do not run it again in the sensitivity loop:

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
for res in 10000 50000; do
  for noise in 0.01 0.02 0.03; do
    for variant in \
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
done
```

Each directory contains its own `run_info.json`, `per_cloud.csv`, and
`summary.csv`. The two loops make 42 condition/setting invocations: six
baseline runs with five T values each, and 36 nonbaseline runs at T=30.
Together they yield 66 aggregate rows if every run completes. For the
directional P2M reviewer table, use the baseline `T=30` row in each of the
six conditions and compare both directional terms and total P2M against
competitor outputs evaluated on the same shapes with `tools/eval_external.py`.
For the separate sensitivity reviewer table, use those same six baseline
`T=30` rows alongside the 36 nonbaseline rows. Compare CD, both directional
P2M terms, total P2M, latency, and peak memory; `r` changes the number of
patches and thus inference cost. Do not infer the cause of a P2M difference
from the directional split alone. If the actual test release has a different
number of shapes, set `--expected-clouds` to that verified count. Re-run an
interrupted command with the same settings/output directory plus `--resume`.

Before writing either reviewer response, check that every selected
`summary.csv` row has the expected `n_clouds`, the same checkpoint and data
root in `run_info.json`, and a complete `per_cloud.csv`. For the directional
P2M table, use columns: resolution, noise, method, shape count, CD,
point-to-face, face-to-point, and total P2M (all distance metrics x1e4).
For the sensitivity table, use: resolution, noise, setting, eta, gamma, r,
shape count, CD, both directional terms, total P2M, mean latency, and peak
allocated GiB. The competitor evaluator prints `n_cd` and `n_p2m`; require
both to equal the baseline shape count, and confirm identical shape names
before comparing rows. Numerical reviewer claims must wait for those outputs.

## Extended T sweep: PUNet Gaussian/cov/lap and PCNet Gaussian

The current benchmark accepts `--dataset PUNet|PCNet`, `--steps 15 25 35`,
`--noise-family`, explicit clean/noisy/mesh paths for PCNet, and `--sor on|off`.
It records the two directional P2M terms,
core inference time, separate SOR time, both throughput definitions, PyTorch
allocated-memory peak, and model forward count. `--profile-flops` adds
`profiled_matmul_conv_gflops`: this is a PyTorch profiler estimate for the
model's supported matrix multiplication/2D convolution operators, scaled by
the actual patch-batch count and T. It is **not** complete pipeline FLOPs:
unsupported/custom CUDA operators and FPS/KNN/fusion may be omitted. Profiling
is outside the timing boundary. Do not label that column simply "total FLOPs"
in a paper. See `run_info.json` for its scope and the actual noisy directory.

The following server paths are examples from the previous run and the supplied
cov/lap directory. Check them before starting. These commands deliberately
use new output directories because older CSV files have a different schema.
Copy the locally edited `tools/benchmark_inference.py` to the server first;
these edits are uncommitted, so `git pull` alone will not transfer them.
The filename suffix is used as the nominal inference sigma for cov/lap; if
the dataset generator defines that value differently, set
`--inference-sigma` to the validated model input instead.

```bash
cd /home/sapi/zhy/pointgpt-claude
CKPT=/workspace/experiments/L_consistency_plus/ckpt-best.pth
DATA_ROOT=/workspace/data/ScoreDenoise
OTHER_NOISE_ROOT=/home/sapi/zhy/pointgpt-claude/data/other-noise
```

For all six PUNet Gaussian conditions, run only the three new T values with
SOR off. Existing T=5/10/20/30 quality results remain in the earlier runs;
their FLOPs column is empty, so do not invent FLOPs for those rows.

```bash
for res in 10000 50000; do
  for noise in 0.01 0.02 0.03; do
    CUDA_VISIBLE_DEVICES=0 python tools/benchmark_inference.py \
      --config cfgs/PointGPT-L/finetune_scoredenoise.yaml \
      --checkpoint "$CKPT" --data-root "$DATA_ROOT" \
      --dataset PUNet --noise-family gaussian \
      --resolution "$res" --noise "$noise" \
      --steps 15 25 35 --sor off --profile-flops --expected-clouds 20 \
      --output-dir "experiments/rebuttle_T_sweep_v2/PUNet/gaussian/${res}_${noise}" || exit 1
  done
done
```

If the previously requested Gaussian 50K/3% visual comparison is still needed,
run it separately; it is not part of the three-value cost sweep above. The
image and `.xyz` files are produced outside the timed region.

```bash
CUDA_VISIBLE_DEVICES=0 python tools/benchmark_inference.py \
  --config cfgs/PointGPT-L/finetune_scoredenoise.yaml \
  --checkpoint "$CKPT" --data-root "$DATA_ROOT" \
  --dataset PUNet --noise-family gaussian --resolution 50000 --noise 0.03 \
  --steps 5 10 15 20 25 30 35 --sor off --profile-flops \
  --visualize-clouds all --expected-clouds 20 \
  --output-dir experiments/rebuttle_T_sweep_v2/PUNet/gaussian_visual/50000_0.03
```

For covariance and Laplace noise, use the supplied `data/other-noise`
directory pattern. Run SOR on for the paper-protocol table. The script also
supports SOR off for a separate controlled analysis: change `for sor in on`
to `for sor in on off` below if that additional sweep is needed. The two
settings go to separate directories and have separate core versus
SOR-inclusive latency/throughput columns.

```bash
for family in cov lap; do
  for sor in on; do
    for res in 10000 50000; do
      for noise in 0.01 0.02 0.03; do
        noisy_dir="$OTHER_NOISE_ROOT/PUNet_${res}_poisson_${noise}_${family}"
        CUDA_VISIBLE_DEVICES=0 python tools/benchmark_inference.py \
          --config cfgs/PointGPT-L/finetune_scoredenoise.yaml \
          --checkpoint "$CKPT" --data-root "$DATA_ROOT" \
          --dataset PUNet --noise-family "$family" --noisy-dir "$noisy_dir" \
          --resolution "$res" --noise "$noise" --inference-sigma "$noise" \
          --steps 5 10 15 20 25 30 35 --sor "$sor" \
          --profile-flops --expected-clouds 20 \
          --output-dir "experiments/rebuttle_T_sweep_v2/PUNet/${family}/${sor}/${res}_${noise}" || exit 1
      done
    done
  done
done
```

For PCNet Gaussian noise, point `PCNET_ROOT` at the dataset with
`pointclouds/test/<resolution>_poisson`,
`noisy/PCNet_<resolution>_poisson_<noise>`, and `meshes/test/<shape>.off`.
These paths come from the existing PCNet alignment script and config examples;
verify the actual server layout before running. PCNet is evaluated without
SOR to preserve sharp CAD features. Use `--expected-clouds 10` only if the
server's paired PCNet test set contains the expected 10 shapes. Name checks
cannot verify clean/mesh coordinate alignment or the nominal sigma convention;
inspect those with `tools/check_pcnet_alignment.py` before interpreting P2M.

```bash
PCNET_ROOT=/home/sapi/zhy/compare_methods/PCNet
for res in 10000 50000; do
  for noise in 0.01 0.02 0.03; do
    CUDA_VISIBLE_DEVICES=0 python tools/benchmark_inference.py \
      --config cfgs/PointGPT-L/finetune_scoredenoise.yaml \
      --checkpoint "$CKPT" --data-root "$DATA_ROOT" \
      --dataset PCNet --noise-family gaussian \
      --clean-root "$PCNET_ROOT/pointclouds/test" \
      --noisy-dir "$PCNET_ROOT/noisy/PCNet_${res}_poisson_${noise}" \
      --mesh-root "$PCNET_ROOT/meshes" \
      --resolution "$res" --noise "$noise" \
      --steps 5 10 15 20 25 30 35 --sor off \
      --profile-flops --expected-clouds 10 \
      --output-dir "experiments/rebuttle_T_sweep_v2/PCNet/gaussian/${res}_${noise}" || exit 1
  done
done
```

The script checks exact clean/noisy shape-name pairing and mesh availability
before writing results. Each condition writes `run_info.json`, `per_cloud.csv`,
and `summary.csv`; the latter contains CD, both P2M directions and their sum,
mean latency, core and SOR-inclusive throughput, peak allocated GPU memory,
model forward count, and profiler-covered GFLOPs. If a directory has a verified
shape count other than 20 for PUNet or 10 for PCNet, adjust `--expected-clouds`;
do not silently compare subsets. To resume, repeat the identical command for
that one condition with `--resume`. Do not mix SOR-on and SOR-off rows, or
profiler-covered FLOPs and total FLOPs, in the same comparison column.
