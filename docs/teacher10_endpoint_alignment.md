# Teacher10 endpoint alignment for the four-step Student

Base branch: `paper2-teacher10-student4`, commit
`7bc43fc4db501392e487dcbb1fa7b7a40c757cb4`.

The control is `cfgs/PointGPT-L/distill_10to4_input_scheduled.yaml`. The new
config, `cfgs/PointGPT-L/distill_10to4_input_endpoint.yaml`, adds only the
`teacher_endpoint` objective block. It keeps fixed nodes `[0,2,5,7,10]`, the
Teacher/Student update rules, Mixed source probabilities, input-matched local
Teacher targets, local input detach, optimizer, learning rate, 20 epochs,
training noise 0.005-0.020, 10k/1% validation, and the original Score selection.

## Objective

The ordinary local pass still performs four stage losses and accumulates their
average gradient `g_local` before any optimizer update. The new pass starts
from cached noisy `T0`, runs all four Student intervals continuously, and
matches cached `T10` from that same noisy patch:

`L_end = mean_points_and_patches(sum_xyz((Student4(T0) - detach(T10))**2))`.

Every Student transition remains connected to the endpoint gradient. No
Teacher states are inserted between the Student updates. The target is the
Teacher10 output, not the paired clean cloud. No extra Teacher rollout is
needed because the local pass has already cached T10.

The auxiliary pass uses Student eval mode with autograd enabled to match
inference and avoid extra BatchNorm running-stat updates. The ordinary local
pass remains in train mode. Module modes and Python/NumPy/Torch RNG states are
restored after the auxiliary pass, including on exceptions. Its checkpoint
recomputation also runs inside that eval context.

`torch.autograd.grad` computes `g_end` separately, then the optimizer receives
`g_local + lambda * g_end`, followed by the existing single clipping and step.
Norms cover all trainable Student parameters and the whole optimizer batch.
Endpoint micro-batches are weighted by their sample counts before summation.

`lambda = min(max_weight, ramped_ratio * ||g_local|| / ||g_end||)`.

If either norm is zero, lambda is zero. Lambda is a detached per-batch
coefficient, not a differentiable objective parameter. The default ratio is
zero at epochs 1-2, then 0.025, 0.05, 0.075, and 0.1 at epochs 3-6. It stays
at 0.1 thereafter, with lambda capped at 1. The weighted endpoint gradient
therefore has norm at most 10% of the local gradient before combined clipping.
This is a conservative starting setting, not a measured optimal weight or a
guarantee of improvement. The ratio controls gradient norms, not loss values.

The endpoint patch batch is 1, with non-reentrant activation checkpointing.
The extra forward/backward pass and temporary endpoint gradient vector add
compute and memory. Runtime and GPU memory have not been measured. A compatible
PyTorch environment supporting `checkpoint(..., use_reentrant=False)` is needed.

## Manual training

Sync the changed runner, new utility, new YAML, and six-condition test script
to the server. Start a fresh Student from the original Teacher weights:

```bash
cd /workspace
RUN_ROOT="experiments/input_ablation/mixed_10to4_endpoint_seed0_$(date +%Y%m%d_%H%M%S)_$$"
CUDA_VISIBLE_DEVICES=0 python -u tools/runner_distill.py --mode train \
  --config cfgs/PointGPT-L/distill_10to4_input_endpoint.yaml \
  --teacher_ckpt experiments/L_consistency_plus/ckpt-best.pth \
  --output_dir "$RUN_ROOT/train" --device 0 --seed 0
```

The existing Mixed resume restriction still applies. This command does not
resume the previous best6 Student. A fixed training plan from the same Teacher
initialization supports comparison with the existing Teacher10 control.

`manifest.json` and checkpoints include the objective protocol. The manifest
also fingerprints the relevant implementation files. In `train.jsonl`:

- `loss_traj` remains the original local average.
- `loss_teacher_endpoint` is the unweighted endpoint MSE, or null in warmup.
- `loss_teacher_endpoint_weighted` averages lambda times endpoint MSE.
- `loss_total` is local loss plus that weighted endpoint loss.
- `endpoint_weight_mean/min/max` expose the effective per-batch weight.
- `endpoint_target_grad_ratio` and `endpoint_actual_grad_ratio_mean` expose the
  requested and achieved norm ratio. The latter can be smaller if the cap binds.
- `endpoint_local_grad_norm_mean` and `endpoint_grad_norm_mean` expose the raw
  gradient norms. Means are weighted by optimizer-batch sample count.

Best still minimizes validation `CD + 0.3 * P2M`. Neither endpoint MSE nor total
training loss selects best. The control config has no `teacher_endpoint` block,
so its training path and checkpoint objective metadata remain unchanged.

## Manual six-condition test and ZIP

After all planned epochs have finished, set `TRAIN_ROOT` to this new run's
`train` directory, not a checkpoint file and not the older Teacher10 run:

```bash
cd /workspace
TRAIN_ROOT="/workspace/experiments/input_ablation/<new_endpoint_run>/train"
CUDA_VISIBLE_DEVICES=0 python -u tools/test_teacher_endpoint_six.py \
  --train_root "$TRAIN_ROOT" --device 0 --seed 0
```

This checks the completed training plan, current implementation hashes, actual
checkpoint epoch/schedule/objective, then evaluates all 20 PUNet clouds for
10k/50k at 1%/2%/3%. The original test pipeline writes denoised point clouds
into fresh output directories. The helper verifies CSV means and full coverage
and generates one ZIP containing the six metrics/manifests, three training
records, and `selected_checkpoint.json` with the actual checkpoint SHA256 and
selected epoch. The checkpoint binary and point-cloud outputs are not in the
ZIP. The final line is `ZIP=<absolute_path>`.

## Verification status

No training, inference, GPU diagnostics, or CPU unit tests have been executed.
The focused tests in `tests/test_teacher_endpoint.py` are provided for future
manual verification of the full four-step gradient, gradient balancing,
micro-batch/checkpoint equivalence, RNG isolation, and unchanged BatchNorm
buffers. Static parsing/review cannot establish CUDA correctness, memory usage,
or numerical benefits on the benchmark.
