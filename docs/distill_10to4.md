# Teacher 10 to Student 4

Branch base: `paper2-scheduled-rollout-input`, commit `937dd7855aceeeca160907cb2d069ebdc14f1848`.

The supplied T sensitivity result motivates ending the Teacher trajectory at
step 10. It does not determine the optimal Student budget. Start with four
Student steps to keep the same inference budget as the historical Mixed20
Student. The Teacher-to-Student model evaluation ratio is 10/4 = 2.5; this is
not a measured wall-clock speedup. A two-step Student would make each learned
jump larger and needs its own quality/latency comparison before adoption.

The fixed nodes are `[0, 2, 5, 7, 10]`, computed as `floor(10 * stage / 4)`.
They are a near-uniform starting schedule, not a result of node optimization.
The cached Teacher trajectory runs ten updates at eta 0.3 and decay 0.95. The
Student has four updates at eta 1. Its actual sigma before each update is
`sigma0 * 0.95**start_step`, with start steps `[0, 2, 5, 7]`. The generic patch
loop's nominal decay is overridden by this existing interval-forward adapter.

Use the same denoising Teacher network checkpoint used in the T sensitivity
study. The number of Teacher iterations is an inference schedule; changing
16 to 10 does not require retraining that network. This assumes that the T
sensitivity study used the same eta 0.3, decay 0.95, patching, and postprocessing.
The new Student starts fresh from those original Teacher weights. Do not load
an already distilled 16-to-4 or terminal-loss Student as the Teacher.

The new config preserves the base branch's input-matched Teacher supervision
with `student_target: teacher_from_student_input`. Teacher-source samples use
the cached next Teacher node. Student-source samples feed their actual detached
input to the frozen Teacher, which performs the configured interval updates:

| Stage | Teacher-source target | Student-source target |
| --- | --- | --- |
| 1, 0 to 2 | T2 | Stage 1 always uses T0 |
| 2, 2 to 5 | T5 | Teacher from detached S1, 3 updates, initial sigma = sigma0 * 0.95**2 |
| 3, 5 to 7 | T7 | Teacher from detached S2, 2 updates, initial sigma = sigma0 * 0.95**5 |
| 4, 7 to 10 | T10 | Teacher from detached S3, 3 updates, initial sigma = sigma0 * 0.95**7 |

These extra Teacher calls generate detached training labels and are counted
in `training_target_stages` as `extra_teacher_sample_steps`. The nominal
`teacher_steps: 10` describes the base trajectory, not total training calls
including input-matched labels. Only the Student receives gradients.

Training noise remains 0.005-0.020. Terminal clean loss, dynamic node selection,
and shadow search are disabled. The base branch currently uses 10k/1% validation, and the new config
validates the full Student rollout after every epoch, matching the historical
full20 run's selection cadence. Best still minimizes CD + 0.3 * P2M.

For isolating the change in Teacher endpoint, the matched control is
`distill_16to4_input_matched.yaml`, which uses the same target policy and
per-epoch validation. Historical original Mixed20 used fixed node targets and
therefore also differs in its Student-source supervision policy.

The existing condition embedding encodes absolute iteration indices divided by
16. That fixed feature scale is retained for compatibility; it does not cause
extra Teacher updates. Both training/validation targets and checkpoint schedule
metadata now end at step 10. Legacy 16-to-4 schedules keep their original
metadata and forward paths. Dynamic/shadow search remains limited to 16-to-4.

After syncing this branch's changes into the Linux project, manually train:

```bash
cd /workspace
RUN_ROOT="experiments/input_ablation/mixed_10to4_matched_seed0_$(date +%Y%m%d_%H%M%S)_$$"
CUDA_VISIBLE_DEVICES=0 python -u tools/runner_distill.py --mode train \
  --config cfgs/PointGPT-L/distill_10to4_input_scheduled.yaml \
  --teacher_ckpt experiments/L_consistency_plus/ckpt-best.pth \
  --output_dir "$RUN_ROOT/train" --device 0 --seed 0
```

`TRAIN_ROOT` is `$RUN_ROOT/train`. Inspect `manifest.json`,
`validation_manifest.json`, and `train.jsonl` for `teacher_steps: 10`,
`student_steps: 4`, `[0, 2, 5, 7, 10]`, and
`student_target: teacher_from_student_input` before associating later results with
this experiment. Evaluate the trained Student against Teacher10 and historical
Student4 under the same six-condition test protocol. Neither CPU tests, GPU
diagnostics, training, nor inference has been executed for this change.
