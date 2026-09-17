# Epoch-level PCD curriculum distillation

`cfgs/PointGPT-L/distill_16to4_dynamic.yaml` adds `curriculum_mode: dynamic_pcd`.
The existing `fixed` and `fixed_nonuniform` configurations keep their behavior.

## Training and selection

Epoch 1 uses YAML `dynamic_pcd.initial_nodes`, by default `[0,4,8,12,16]`.
An epoch holds one shared schedule constant for all training patches. Each of its
four stages trains on `T_start` against `T_target`, using the interval condition
`[start,target,target-start]/16` and `sigma0 * 0.95**start`. The loss remains
`(prediction-target).square().sum(-1).mean()`, averaged over four stages. No clean,
P2M, consistency, feature, or Student-rollout training loss is added.

The order is:

1. Train with this epoch's nodes.
2. Validate teacher-forced loss with the same nodes.
3. On rollout validation epochs, evaluate continuous four-step Student inference
   with those nodes, using the original fusion/postprocessing/CD/P2M protocol.
4. On curriculum update epochs, evaluate the current Student on calibration
   intervals and run the existing exact global search.
5. Save checkpoints and logs with both used and next nodes.
6. Use the selected next nodes starting with the next epoch.

`update_every_epochs: 1` updates after every epoch, including the last one so that
a resumed run can start with the new schedule. Larger values update at integer
multiples of the configured interval. There is no per-batch search, smoothing,
EMA, or movement cap. The old per-batch shadow option must be disabled in dynamic
mode. Search parameters come from `dynamic_pcd`, not the legacy `pcd_dynamic` block.

## Calibration and search cost

The default calibration bank contains 64 patches, selected independently from the
existing training loader's `PairedPatchDataset`, with `calibration_seed: 2025`.
Sampling calls the original dataset directly, with isolated Python/NumPy/Torch
RNG state; it does not consume a training loader batch. The same clean-cloud
normalization, noise distribution, and shared noisy-KNN indices for clean/noisy
pairs are retained. These are fresh fixed patch/noise realizations from the train
split, not disjoint training shapes and not validation/test samples.

The frozen/eval Teacher generates `T0..T16` once for this fixed bank. These detached
CPU trajectories and per-patch sigma values are reused throughout the run. The
calibration manifest records dataset indices, names, sigma, patch count/seed, and
a SHA-256 fingerprint of the paired patch data. Calibration contributes no loss
or optimizer update.

For each update, `build_interval_pcd_cache()` reevaluates the current Student with
the appropriate condition and sigma. Search uses eval/no_grad and restores module
modes and RNG states. For each interval, aggregate the **mean of per-patch ratios**:
`mean_i(E_imit_i / (D_move_i + 1e-12))`; do not divide the averaged numerator by
the averaged denominator. Partial calibration batches are weighted by patch count.

`search_global_teacher_nodes()` scores all 455 complete four-edge paths. Its
objective remains `mean(abs(stage_PCD-target)) + lambda_balance * population_std(stage_PCD)`.
Defaults are target 0.3 and lambda 1.0. Aggregation produces one shared path for
the next epoch, not separate paths per calibration patch.

The existing exact cache has 130 distinct intervals used by the 455 paths. The
six other `(t,u)` pairs cannot be part of any strictly increasing four-edge path
from 0 to 16. No previously evaluated candidate is removed. With 64 patches and
`interval_patch_batch: 8`, an update makes `130 * ceil(64/8) = 1040` Student forward
calls (8320 patch/interval evaluations), followed by 455 cheap cached path scores.
Teacher caching does not cache Student predictions across epochs.

## Validation, checkpoint and resume semantics

Dynamic trajectory validation caches all 17 Teacher states, then selects this
epoch's five nodes. Rollout validation receives this epoch's schedule explicitly.
The best metric remains `CD + 0.3 * P2M`; trajectory loss does not select best.

Both last and best checkpoints bind model weights to their evaluated schedule:

- `distillation.teacher_nodes`, `current_teacher_nodes`, and
  `nodes_used_this_epoch` are the schedule **used for the saved epoch**.
- `next_teacher_nodes` is the schedule to use when continuing training.
- `pcd_target`, `lambda_balance`, `dynamic_update_every_epochs`,
  `dynamic_pcd_config`, `calibration_metadata`, and `curriculum_history` describe
  the selection state. History includes update epoch, old/new nodes, four PCDs,
  population std/range, target error, path score, and forward count.
- `best_epoch` and `best_val_rollout_score` retain the existing ranking state.

Dynamic test mode reads `distillation.teacher_nodes` from the loaded checkpoint
and checks it against `nodes_used_this_epoch`. It never uses `next_teacher_nodes`
or the YAML initial schedule to infer with a dynamic best checkpoint. The test
summary and manifest report the actual loaded schedule.

Resume uses the last checkpoint's optimizer and `next_teacher_nodes`, preserving
history and historical best. The dynamic configuration must match the checkpoint;
the calibration bank is deterministically rebuilt and its metadata/fingerprint
must match. A changed bank or inconsistent history raises an error. This checks
curriculum continuity; it does not add bitwise replay of training minibatch RNG.
When resuming into a new output directory, the historical best file remains in
the original directory until a new best is produced, as in the fixed runner.

`train.jsonl` records `teacher_nodes_used`, `stage_gaps`, `dynamic_update_performed`,
`next_teacher_nodes`, `search_stage_PCD`, `search_PCD_std`, `search_PCD_range`,
`search_mean_target_error`, `search_path_score`, and `search_student_forward_calls`.
Search fields are null (forward count zero) on epochs without an update.

## Verification without formal training

CPU regression tests:

```bash
python -m unittest discover -s tests -p 'test_distill*.py' -v
```

The CUDA smoke entry uses the full PointGPT-L, the production `train()` loop, two
mini epochs with one original batch each, eight calibration patches, and one
complete validation cloud per epoch. Only smoke data limits and validation cadence
are overridden; the effective configuration is written beside its report.

```bash
python tools/smoke_dynamic_distill.py \
  --teacher_ckpt experiments/L_consistency_plus/ckpt-best.pth \
  --output_dir output/dynamic_pcd_gpu_smoke
```

It checks Teacher immutability, finite condition/backbone gradients, actual stage
inputs/conditions/sigmas, search isolation, epoch-2 adoption of epoch-1's search,
best reload, and saved resume state. `smoke_result.json` contains the measured
nodes/PCDs, gradient norms, checkpoint state checks, runtime, and GPU memory.
The output directory must be new. Smoke checkpoints live under its `train/`
subdirectory and are not formal training checkpoints. This entry never launches
20 epochs.

## Five-epoch pilot with the twenty-epoch plan preserved

`--stop_after_epoch 5` stops only after epoch 5 training, scheduled whole-cloud
validation, curriculum update, checkpoint saves, and log flush. Keep `epochs: 20`
in the YAML and omit `--epochs 5`. This boundary does not change the optimizer,
learning rate, total plan, curriculum objective, or checkpoint schema. A resumed
run still starts at `checkpoint.epoch + 1` with `next_teacher_nodes`; the stop
argument is not persisted in the checkpoint. No subsequent epoch starts unless a
new run is explicitly launched.

```bash
python tools/runner_distill.py --mode train \
  --config cfgs/PointGPT-L/distill_16to4_dynamic.yaml \
  --teacher_ckpt experiments/L_consistency_plus/ckpt-best.pth \
  --output_dir experiments/distill_16to4_dynamic/pilot5 \
  --seed 0 --stop_after_epoch 5
```

Epoch logs additionally distinguish `nodes_used_this_epoch` from
`next_teacher_nodes`. `checkpoint_teacher_nodes` describes that epoch's saved
last checkpoint (and best when `is_best` is true); it does not rebind an older
best checkpoint. `mean_PCD`, `train_PCD_std`, and `train_PCD_range` describe
training batches under the used nodes, while `search_stage_PCD`, `search_PCD_std`,
and `search_PCD_range` describe the selected next path on calibration patches.

Timing fields separately record training, trajectory validation, rollout
validation, curriculum search, checkpoint writing, setup, and total elapsed time.
CUDA is synchronized at phase boundaries for accurate timings; no extra model
forward or optimizer step is introduced.
