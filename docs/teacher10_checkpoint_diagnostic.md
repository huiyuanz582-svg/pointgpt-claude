# Diagnose existing Teacher10 / Student4 checkpoints

Entry point: `tools/diagnose_teacher10_checkpoint.py`. Use the current endpoint
alignment branch. This is an observational audit with no optimizer, training
loop, parameter updates, or saved model weights. It does perform backward
passes to measure gradients. Do not confuse those measurements with training.

The old `tools/diagnose_distill_trajectory.py` assumes a 16-step Teacher and is
not the entry point for this experiment.

## Compared inputs and checkpoints

The primary `--train_root` is the endpoint-alignment run. The optional
`--reference_train_root` is the earlier Teacher10 control without endpoint
loss. The audit reads each run's manifest and train log, validates the actual
best checkpoint's epoch, schedule, and objective, and records its SHA256.
The control and primary must use the same settings except `teacher_endpoint`.
No hard-coded best5 or best6 selection is used.

Defaults are the six 10k/50k by 1%/2%/3% PUNet conditions, the first four cloud
names in sorted order, and two fixed noisy-KNN 1024-point patches per cloud.
These are 48 diagnostic patches, not a full benchmark evaluation. All models
receive exactly the same cached noisy patches and Teacher states. Patch seeds
are fixed per resolution/name; the same seed-point indices are used across
noise levels, while noisy-KNN membership can change with noise.

The clean/noisy pairing and normalization come from the existing test dataset.
No mesh/P2M or whole-cloud CD is recomputed. Diagnostic MSE uses corresponding
point squared L2 in normalized patch coordinates. It must not replace the
existing benchmark Score. Three-percent input is only a diagnostic stress case.

## Stage errors

For all four stages of each selected patch, `stage_errors.csv` records:

- `teacher_forced_mse`: a Student update from the original Teacher start node
  compared with the next Teacher node. Large values indicate a difficult local
  transition even without an inherited Student state error.
- `free_to_canonical_mse`: the continuously chained Student compared with the
  corresponding original Teacher node. This includes accumulated path error.
- `free_minus_teacher_forced_mse`: the difference between those measurements.
  Stage 1 is computed once on its shared T0 input and has zero gap by construction.
- `free_to_input_matched_target_mse`: the Student update compared with a fresh
  frozen Teacher interval starting at the same actual Student input.
- `matched_target_to_canonical_mse`: that fresh Teacher target's distance to the
  original Teacher node. A small local imitation error alongside a large value
  here indicates a moving target can remain displaced from the original path.

`stage_summary.json` averages each field by checkpoint, condition and stage.
Each selected cloud contributes the same number of patches. These comparisons
locate potential mechanisms; none alone proves causality.

## Gradient probes

By default gradients are measured only for 10k/1% and 50k/3%, on the first eight
cached patches in each condition, matching the original optimizer batch size.
Both checkpoints use the same fixed Teacher mask with probability 0.2.
This standardizes the comparison and is not an exact replay of the different
training batches at best5/best6. Reported norms can therefore differ from the
epoch averages in `train.jsonl`.

Six cases are compared on each batch:

| Case | Purpose |
| --- | --- |
| `local_train` | Original Mixed, input-matched local loss with train mode |
| `local_eval` | Same local objective/mask with eval mode |
| `endpoint_eval_full` | Continuous four-step endpoint gradient with eval mode, matching the added objective |
| `endpoint_eval_last_only` | Same eval forward rollout, with the first three steps disconnected from the gradient |
| `endpoint_bn_batch_stats_full` | Full endpoint gradient with only BatchNorm in train mode; dropout/DropPath stay off |
| `endpoint_train_full` | Full endpoint gradient with all Student modules in train mode |

`gradient_probes.json` records raw gradient norm, group norms, the twelve largest
parameter norms, cosine with `local_train`, and endpoint-output state gradient
norms. Gradient norms and cosine describe the measured batches, not a proven
optimization intervention. In particular, changing BN statistics can change
both predictions and gradients, and full-versus-last-only also changes which
uses of the shared parameters contribute.

Every probe restores modes, running buffers, RNGs, and clears gradients.
Checkpoint parameter version checks and file size/mtime checks guard against
updates. Teacher parameters remain frozen. Any temporary BN running-stat
updates in train-mode probes are discarded.

## Manual server command

Sync the new diagnostic script into the existing server checkout. Run:

```bash
cd /workspace
CUDA_VISIBLE_DEVICES=0 python -u tools/diagnose_teacher10_checkpoint.py \
  --train_root /workspace/experiments/input_ablation/mixed_10to4_endpoint_seed0_20261109_083432_40/train \
  --reference_train_root /workspace/experiments/input_ablation/mixed_10to4_seed0_20261103_091537_40/train \
  --teacher_ckpt /workspace/experiments/L_consistency_plus/ckpt-best.pth \
  --max_shapes 4 --patches_per_shape 2 \
  --gradient_conditions 10k_1pct 50k_3pct \
  --teacher_probability 0.2 --seed 2024 --device 0
```

The default output is a fresh `teacher10_diagnostic_<timestamp>_<pid>` directory
next to the endpoint run's `train` directory. It refuses to overwrite an
existing directory or ZIP. On success, the last line prints `ZIP=<path>`.
The ZIP contains the manifest, full log, patch-level stage errors, aggregated
stage errors, and gradient probes. It contains no checkpoint tensors or clouds.
On failure, the directory and failed manifest are preserved; no success ZIP is
created. Send the error log before changing sample counts or memory settings.

`--gradient_conditions` with no following values disables all backward probes
for a trajectory-only audit. It does not answer the gradient-source question.
Increasing `--max_shapes` broadens the stage-error coverage, while gradient
probes still use the first eight patches to preserve the optimizer-batch size.

## Verification status

Only static review and syntax checks are planned locally. No model loading,
CPU unit test, GPU diagnostic, training, or inference has been executed here.
The script requires the original server CUDA extensions and PyTorch supporting
non-reentrant checkpointing. Runtime, memory consumption and numerical output
remain to be verified when the manual command is executed.
