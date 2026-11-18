# Teacher10 / Student4: local + endpoint supervision in stage 2

The previous stage-2 objective minimized only the distance between the continuous
Student4 endpoint and the Teacher10 endpoint. This experiment retains that term
and adds input-matched local Teacher supervision on the same Student trajectory.
It tests whether local supervision can limit the quality regression observed at
50k/3%; improvement is not established by this code change.

## Objective and gradient flow

Let the nodes be `t = [0, 2, 5, 7, 10]`, and `s_0` be the noisy training patch.
The Student executes four connected updates `s_{i+1} = S(s_i, t_i, t_{i+1})`.
For each actual Student input, the frozen Teacher produces
`y_i = stopgrad(T_{t_i -> t_{i+1}}(stopgrad(s_i)))`. These Teacher intervals run
2, 3, 2, 3 baseline updates with eta 0.3 and sigma starting at
`sigma0 * 0.95 ** t_i`. Stages 2-4 therefore use their actual Student states,
not cached states from the original Teacher trajectory.

The global endpoint remains `y_end = stopgrad(T_{0 -> 10}(s_0))`.

```
L_local = (1/4) * sum_i mean_batch_points(||s_{i+1} - y_i||_2^2)
L_end   = mean_batch_points(||s_4 - y_end||_2^2)
L_total = local_weight * L_local + endpoint_weight * L_end
```

The combined configuration initially uses both weights equal to 1. Local terms
are averaged over four stages; they are not summed at four times the scale.
These are fixed experimental weights, not an adaptive gradient balance or a
claim that the two terms have equal gradient norms.

Teacher inputs and outputs are detached. The Student trajectory has no
interstage detach: every active loss updates its current and preceding Student
steps. Local Teacher calls run outside activation-checkpoint closures and
preserve RNG states through the existing `teacher_interval_target` helper.
Backward recomputes Student activations, not Teacher targets.

The combined objective uses 20 Teacher updates per training patch (10 global
and 10 local), while each one-term ablation uses 10. This is additional training
cost. Inference still uses only the original four Student steps.

## Controlled settings

- Start each variant from the same masked Student **best6**, with a fresh AdamW.
- LR `1e-6`, weight decay 0, optimizer batch 8, micro-batch 1, clipping 1.
- Train noise remains **0.5%-2%**. Clean patches are not supervision targets.
- Student uses eval-mode autograd, fixed BN buffers, and frozen pre-BN Conv
  biases; BN affine and other parameters remain trainable.
- Full-cloud validation still selects best by `CD + 0.3 * P2M`; epoch 0 is an
  eligible checkpoint. Endpoint validation MSE is only a diagnostic.
- Keep the same training seed and epoch budget across objective ablations.

| Configuration | Local weight | Endpoint weight |
| --- | ---: | ---: |
| `distill_10to4_endpoint_finetune.yaml` (unchanged historical configuration) | 0 | 1 |
| `distill_10to4_local_finetune.yaml` | 1 | 0 |
| `distill_10to4_local_endpoint_finetune.yaml` | 1 | 1 |

The new configurations retain a 3-epoch default. The first comparison below
explicitly declares a **1-epoch** plan to compare with the existing 1-epoch
endpoint result. Do not initialize from its already-fine-tuned checkpoint.
Changing to 3 epochs is a separate, fresh experiment from the same best6.

## Manual train, six-condition evaluation and ZIP

Sync the modified code and both configurations to the server before running.
Run this entire block from the server shell. Nothing is executed automatically
by editing the repository.

```bash
(
set -euo pipefail
cd /workspace
INIT_ROOT="/workspace/experiments/input_ablation/mixed_10to4_endpoint_masked_seed0_20261112_075942_40/train"
CONFIG="cfgs/PointGPT-L/distill_10to4_local_endpoint_finetune.yaml"
RUN_ROOT="experiments/input_ablation/mixed_10to4_local_endpoint_finetune_1epoch_seed0_$(date +%Y%m%d_%H%M%S)_$$"
test -f "$INIT_ROOT/ckpt-best.pth"
printf 'RUN_ROOT=%s\n' "$RUN_ROOT"
CUDA_VISIBLE_DEVICES=0 python -u tools/runner_distill.py --mode train \
  --config "$CONFIG" \
  --teacher_ckpt experiments/L_consistency_plus/ckpt-best.pth \
  --init_student_ckpt "$INIT_ROOT/ckpt-best.pth" \
  --epochs 1 --output_dir "$RUN_ROOT/train" --device 0 --seed 0
CUDA_VISIBLE_DEVICES=0 python -u tools/test_teacher_endpoint_six.py \
  --train_root "$RUN_ROOT/train" --device 0 --seed 0
)
```

The test helper selects the validated `ckpt-best.pth`, evaluates 10k/50k at
1%/2%/3% on all 20 clouds each, and prints `ZIP=<absolute_path>`. The combined
ZIP prefix is `six_condition_metrics_10to4_local_endpoint_finetune_`. It includes
the training log, training/validation manifests, selected checkpoint identity,
and all six quality summaries/CSVs. It does not repeat the efficiency benchmark.

For the local-only ablation, change `CONFIG` to
`cfgs/PointGPT-L/distill_10to4_local_finetune.yaml` and change `local_endpoint`
to `local` in `RUN_ROOT`. For the endpoint-only control, use the unchanged
`distill_10to4_endpoint_finetune.yaml` and `endpoint` in the run name. All three
use the same `INIT_ROOT`, `--epochs 1`, seed and LR. Run them separately; the
block above launches only the combined experiment.

In a later shell, set `--train_root` to the printed run's **train directory**,
not the `.pth` file. The helper requires the source fingerprints to match the
training manifest; do not change implementation between training and testing.
Historical runs retain their historical fingerprints and are not relabeled as
new-objective runs.

## Logs and verification

New configurations record `loss_teacher_local`, its four
`loss_teacher_local_stages`, `loss_teacher_endpoint`, both weighted terms and
`loss_total`. The total is the sum of the two weighted epoch means. Disabled
raw terms and all epoch-0 training losses are null. Weighted disabled terms are
zero after training. `finetune_grad_norm_mean` describes the combined gradient
before clipping. Objective weights, target semantics and gradient scope are
saved in checkpoint/manifest metadata.

The historical configuration retains its endpoint-only metadata and
`endpoint_grad_norm_mean` field. Runtime tests for input-matched targets,
detachment, connected gradients, micro-batches and checkpoint recomputation
are supplied in `tests/test_teacher_endpoint.py` but have **not been run**.
Only static parsing, diff checks and source review are performed locally.
CUDA correctness, memory needs and benchmark improvements remain unverified.
