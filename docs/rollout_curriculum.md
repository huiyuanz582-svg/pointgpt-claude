# Rollout-aware scoring + teacher-forced training (Pilot-A)

This implementation changes curriculum scoring only. Training continues to call
the original `backward_stages` with Teacher inputs, the original four losses,
micro-batch scaling, gradient clipping and optimizer update. Model/condition,
Teacher rollout, whole-cloud inference and CD/P2M best-checkpoint selection are unchanged.

Missing configuration, or explicit `curriculum_metric.type: original_pcd` and
`training_input.type: teacher_forced`, dispatches to the existing interval-PCD
cache and selector. Original YAML files are unchanged. New search/calibration
blocks with `original_pcd` are rejected instead of silently ignored.

## Implemented and deferred

Implemented: strict configuration validation, all 455 or explicit candidate paths,
fixed stratified calibration, free-rollout scoring, per-micro-batch DFS prefix reuse,
raw robust-score argmin, configured epoch update interval, complete JSON search
reports, and new checkpoint metadata needed to bind weights to used/next nodes.

**Not enabled:** scheduled-rollout training, EMA, relative/absolute switch thresholds,
minimum hold duration, and rollout-aware training resume. The last four search
settings are parsed/validated and recorded as requested settings; reports and
startup logs explicitly mark smoothing/suppression disabled. Every scheduled
search directly selects the raw `Jrobust` minimum. Ties use lexicographic path order.

`scheduled_rollout` and rollout-aware `--resume` raise before training. Resuming a
new-method checkpoint through the old mode is also rejected. Old checkpoints do
not require new fields, and old-mode resume remains supported. New checkpoint
inference uses `nodes_used_this_epoch`, never `next_teacher_nodes`. Checkpoint
files from existing experiments are never rewritten by this feature.

## Calibration

The formal YAML uses 16 patches at each of `[0.005, 0.01, 0.02]`: 48 patches total.
3% is rejected by configuration and reserved as unseen high-noise testing.

Calibration draws from the actual training dataset only. A shallow copy of the
dataset wrapper uses the existing `NormalizeUnitSphere` and `AddNoise(sigma,sigma)`,
then the unchanged `PairedPatchDataset` noisy-KNN paired extraction. Training
transforms are not mutated. The same fixed dataset indices are used across sigma
groups, with independent noise draws. Actual noise levels must lie inside the
dataset's training range. No clean or test error enters the search objective.

Teacher `T0..T16` is captured once, detached and stored on CPU. Metadata records
the seed, sigma group, dataset index, shape, patch size and paired-input fingerprint.
Python, NumPy and Torch RNG states are restored after calibration and search.

## Exact scoring

For each patch, `d(X,Y) = mean_points(sum_xyz((X-Y)^2))` and
`D = d(T0,T16) + eps`. This is endpoint displacement, not trajectory arc length.

For each candidate `(0,n1,n2,n3,16)`, start at `S0=T0` and recursively call the
existing interval interface on the previous Student output, with
`sigma0 * 0.95**start`, `start_step=start`, `target_step=target`.
No Teacher reset, outer-patch resampling, or fused-cloud feedback is introduced.

```
E[k] = d(S[k], T[n[k]]) / D
components_i = (E[4], mean(E[1:5]), max(E[1:5]))
J_i = alpha * components_i.final + beta * components_i.mean + gamma * components_i.max
J_b = mean_patches_in_group(J_i)
Jrobust = mean_noise_groups(J_b) + lambda_worst * max_noise_groups(J_b)
selected = argmin(Jrobust)
```

All three signs are **plus**. Weights are finite and nonnegative, and alpha/beta/gamma
cannot all be zero. The stage maximum is taken **per patch before patch averaging**.
Groups receive equal weight even when their patch counts differ. Final error is
also part of the mean/max, as specified. Distance reductions use at least float32;
group accumulators use float64. No AMP/TF32 setting is changed.

`denominator_warn_threshold` only counts/warns; it never clamps or discards values.
Nonfinite states/errors abort selection, write a failed report, and propagate the
error. Empty groups or inconsistent sigma labels also fail. A failed search does
not produce a partial winner or update curriculum nodes.

## Prefix reuse and cost

The 455 paths have 13/91/455/455 unique prefixes at depths 1/2/3/4, totaling 1,014.
Only identical full prefixes share states; identical `(start,target)` edges reached
through different histories are recomputed. Student predictions never persist
across micro-batches or curriculum updates. The DFS retains only the active
point-state branch and scalar components for all candidates. Search uses eval/no_grad
and restores every module's prior mode, including mixed train/eval submodules.

Forward calls = unique prefixes times the actual number of calibration micro-batches.
The original 64-patch/batch-8 example is 8,112 calls, versus 14,560 without reuse
and 1,040 for the old local-PCD method. The new 48-patch/batch-8 YAML needs **6,084**
calls per search. The explicit smoke YAML has 10 unique prefixes and three
micro-batches, so it needs **30** calls. These are counts, not measured GPU timings.

The formal search runs after epochs 2,4,...,20; the selected path is used starting
with the next epoch. The existing 47 GiB PyTorch allocator budget remains active;
it is not a Docker-wide hard VRAM quota. There is no automatic OOM retry or batch resizing.

## Reports

Training writes `curriculum_search_epochNNNN.json` for every completed/failed
scheduled search. It contains metric weights; each ranked candidate's final/mean/max
components (equal-weight noise aggregate and per group); each group's `Jb`;
`Jrobust`; first/second gap; group patch counts; denominator min/median/mean/P95
overall and by group; small-denominator warnings; nonfinite counts; actual/theoretical
forward counts; patch-forward evaluations; elapsed search time; and explicit
implementation/deferred-feature flags. A one-candidate runner-up gap is null.

`train.jsonl` records the selected group's scores, gap, forward count and search time.
Legacy search-PCD fields are null in the rollout-aware branch; training-PCD fields
still describe unchanged teacher-forced training. The full candidate table stays
in the separate report. Search times include scoring, checks and transfers; they
exclude one-time Teacher calibration construction and the post-search stage diagnostics below.

## Four-stage curriculum difficulty diagnostics

Every scheduled rollout-aware search is followed by diagnostics on the same
calibration bank and current Student weights. The candidate scorer and ranking
are unchanged. Diagnostics always include `[0,4,8,12,16]` (`fixed_reference`), the
nodes used for the current training epoch (`current`), and the search winner
(`selected`), even when the reference/current path is outside an explicit search
candidate set. Identical paths are computed once with all overlapping roles noted.
The original-PCD branch remains unchanged.

For every unique path, TF uses `F(T_start,start,target)` independently at each
stage; FR starts at `T0` and uses the preceding Student output thereafter. Both
use the original `sigma0 * 0.95**start` and interval condition. Stage 1 has exactly
the same input and condition, so its forward is shared. Each unique path needs
seven diagnostic forwards per micro-batch (four TF plus three additional FR).
No point clouds are fused or resampled in this patch-level diagnostic.

For every noise group `0.005`, `0.01`, `0.02` and pooled `all_noise`, each stage
records distributions of:

|Metric|Definition|Normalization|
|---|---|---|
|`A_TF`|`d(F(T_start), T_target)`|None; corresponding-point squared L2|
|`M`|`d(T_start,T_target)`|None|
|`C_TF`|`A_TF/(M+eps)`|Per-patch local Teacher interval displacement|
|`A_FR`|`d(S_k,T_target)`|None; corresponding-point squared L2|
|`E_FR`|`A_FR/(d(T0,T16)+eps)`|Per-patch Teacher endpoint displacement|
|`A_FR_minus_A_TF`|`A_FR-A_TF`|Paired absolute-error difference|

The recorded distribution statistics are mean, median, P90, P95 and patch count.
`all_noise` pools patch samples; it does not average group quantiles and is distinct
from the equal-noise-group aggregation used by `Jrobust`.

Additional metrics are `first_stage_A_TF_share`, `C_TF_stage_std/range` and
`E_FR_stage_std/range`. Two explicitly named aggregation orders are saved:

- `per_patch_four_stage_summary`: calculate share/std/range inside each patch,
  then report its mean/median/P90/P95 across patches. Population std uses ddof=0.
- `summary_of_four_stage_patch_means`: first average each stage across patches,
  then calculate stage-1 share or four-stage std/range, saved in `value`.

No extra epsilon is added to the share statistic. A zero four-stage TF error sum
makes that patch's share undefined; `undefined_count` records this and the share
distribution counts only defined samples. When all samples have zero TF error,
the share is null. Other nonfinite metrics fail explicitly instead of dropping
samples. Diagnostic failure restores modes/RNG and writes a failed JSON with no
partial CSV; training stops rather than silently skipping diagnostics. No diagnostic
metric replaces or modifies any candidate score or selected nodes.

Files: `curriculum_stage_diagnostics_epoch_XXX.json` and `.csv`. Training uses the
actual epoch; smoke uses `000` to mean a standalone snapshot, not the checkpoint's
training epoch. JSON has model identity, unique paths and role aliases, noise
levels, epsilon, status, nonfinite count, forward counts, elapsed time and flat
statistic `rows`. CSV contains the same rows with these columns:

|Columns|Meaning|
|---|---|
|`epoch`, `model_state_id`, `checkpoint`|Snapshot identity; training ID combines output directory and post-training epoch; smoke includes the loaded checkpoint path when supplied|
|`path_id`, `path`, `path_role`, `overlap_roles`|Actual node list and fixed/current/selected aliases; repeated alias rows must not be counted as extra patches|
|`noise`, `metric`, `normalization`, `aggregation`|Noise level or `all_noise`, metric formula and aggregation order|
|`stage`, `start_step`, `target_step`|Stage 1..4 and interval; null for four-stage summary rows|
|`sample_count`, `total_patch_count`, `undefined_count`|Included samples, total patches, and undefined zero-sum shares|
|`mean`, `median`, `p90`, `p95`, `value`|Distribution statistics; `value` is used only for summaries of stage means|

Std/range cannot be interpreted interchangeably between the two aggregation
orders. Absolute TF error alone is also insufficient to infer local difficulty:
inspect its corresponding `M` and local `C_TF`, as well as the independently
measured FR error accumulation.

Stdout prints one compact mean table per unique path, with role overlap explained:

```text
P0=[0, 4, 8, 12, 16]; roles=fixed_reference/current/selected; computed_once=true
Path | Noise | Metric | Stage1 | Stage2 | Stage3 | Stage4
P0 | 0.005 | C_TF | ... | ... | ... | ...
P0 | 0.005 | A_TF | ... | ... | ... | ...
P0 | 0.005 | E_FR | ... | ... | ... | ...
P0 | 0.005 | A_FR | ... | ... | ... | ...
```

The same rows print for 1%, 2% and all_noise. Complete statistics are in JSON/CSV,
not just the table. `train.jsonl` and search JSON link these artifacts and report
diagnostic forward count/time separately; the original search call count remains
6,084 for the formal configuration. Diagnostics add 42/84/126 forwards for one,
two or three unique paths, respectively. The smoke search remains 30 forwards
and adds 21/42/63 diagnostic forwards. These are additional eval calls with no
backward/optimizer work. Search and diagnostics each restore modes and RNG state.

## Manual server commands (not executed during development)

Run inside the existing Docker environment, from `/workspace`, after switching
to this branch and making its source files available there. Replace checkpoint
paths with the actual files. Use a fresh output directory on every invocation.

Scoring smoke: three explicit paths, two patches per noise, no training or optimizer.
The optional Student checkpoint must contain the existing Step Condition branch.
Omit `--student_ckpt` to test from a Teacher clone with newly initialized condition.

```bash
cd /workspace
CUDA_VISIBLE_DEVICES=0 python -u tools/smoke_rollout_curriculum.py \
  --config cfgs/PointGPT-L/distill_16to4_rollout_smoke.yaml \
  --teacher_ckpt experiments/L_consistency_plus/ckpt-best.pth \
  --student_ckpt /REPLACE/WITH/EPOCH5/ckpt-best.pth \
  --output_dir "diagnostics/rollout_curriculum/smoke_$(date +%Y%m%d_%H%M%S)_$$" \
  --device 0 --seed 0
```

Require `manifest.json` and `search.json` status `completed`, 30 search Student calls,
three groups of two patches, nonfinite count zero, and finite ranked scores.
The two `curriculum_stage_diagnostics_epoch_000` files must also exist, include the
fixed reference's four stages and all roles/groups, and JSON status must be completed.
Diagnostic calls are counted separately. With a Student checkpoint, `current`
means its saved inference nodes; without one, it means the YAML initial nodes.
The smoke program rejects `all_455`, more than eight paths, or more than four
patches per noise group. It never invokes a training loop.

The following Pilot-A command is future reference only: first review the difficulty
diagnostics. No Pilot-A execution is requested as part of this change.
Pilot-A: fresh initialization from the same Teacher, all 455 paths, original TF training.
This preserves the 20-epoch plan and stops after epoch 10's normal save/validation.
**Continuation via resume is not implemented in this phase**; an eventual 20-epoch
run must start fresh unless resume is implemented and validated in a later change.

```bash
cd /workspace
CUDA_VISIBLE_DEVICES=0 python -u tools/runner_distill.py --mode train \
  --config cfgs/PointGPT-L/distill_16to4_rollout_aware.yaml \
  --teacher_ckpt experiments/L_consistency_plus/ckpt-best.pth \
  --output_dir "experiments/rollout_aware/pilotA_$(date +%Y%m%d_%H%M%S)_$$" \
  --device 0 --seed 0 --stop_after_epoch 10
```

Calibration differs from the historical 64-patch log-uniform bank, and cadence is
now two epochs. Comparisons with historical results must report these differences;
they are not a perfectly isolated scoring-only ablation. Final clean CD/P2M Score
is still assessed independently: lower Teacher trajectory error is not guaranteed
to improve it. Validation/checkpoint selection retains the old data protocol.

## Development verification

Only syntax/import checks and the focused CPU suites `test_rollout*.py` and
`test_curriculum_stage_diagnostics.py` are run.
The suite uses synthetic tensors, mock interval forwards, and extracted original
CPU normalization/noise/KNN definitions. It verifies the formula/reduction order,
prefix counts, naive-vs-cached scores, RNG/mode/parameter/gradient preservation,
calibration isolation, raw epoch scheduling, error reports and legacy dispatch.
Stage-diagnostic tests independently check TF/FR formulas, per-stage quantiles,
all_noise pooling, share/std/range aggregation orders, role deduplication, artifact
fields, failed-run behavior and unchanged scores/ranking/model state after diagnostics.
No training, full test suite, trajectory audit, real-model search, server command
or GPU program is run as part of this implementation.
