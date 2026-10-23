# Fixed Mixed20 with clean endpoint supervision

This experiment keeps the teacher checkpoint, fixed nodes `[0, 4, 8, 12, 16]`,
scheduled Student input, 20 epochs, and training noise `0.005-0.020`. The new
objective adds a clean endpoint loss after four continuous Student updates:

`L_total = mean(L_local_1, ..., L_local_4) + 0.25 * mean(||S4 - clean||_2^2)`.

The four local Teacher targets and the scheduled-input mask are unchanged.
The endpoint pass starts from the noisy patch, propagates gradients through all
four Student steps, and uses the aligned clean patch as its target. It runs
with one patch at a time and non-reentrant activation checkpointing to limit
peak memory. Expect higher training cost; this has not been run on the server.

Both configs select `ckpt-best.pth` using the existing 10k/3% validation Score
after every epoch. The 3% noise appears in validation only, never in training.
The Score formula and the six-condition test protocol are unchanged.

From the Linux project directory, after syncing this branch's uncommitted
changes, run the method with a fresh output directory:

```bash
cd /workspace
RUN_ROOT="experiments/input_ablation/mixed20_terminal_val3pct_seed0_$(date +%Y%m%d_%H%M%S)_$$"
CUDA_VISIBLE_DEVICES=0 python -u tools/runner_distill.py --mode train \
  --config cfgs/PointGPT-L/distill_16to4_input_scheduled_terminal.yaml \
  --teacher_ckpt experiments/L_consistency_plus/ckpt-best.pth \
  --output_dir "$RUN_ROOT/train" --device 0 --seed 0
```

For a matched fixed Mixed20 control under the same 3% validation selection,
use a second fresh `RUN_ROOT` and replace the config path with
`cfgs/PointGPT-L/distill_16to4_input_scheduled_val3pct_control.yaml`. The old
Mixed20 result selected its checkpoint with 1% validation, so its six test
scores alone are not a strictly matched model-selection control for this run.

`RUN_ROOT/train/manifest.json`, `validation_manifest.json`, and `train.jsonl`
record the effective protocol and losses. `ckpt-best.pth` is updated only when
the existing validation Score improves. No result should be attributed to the
new objective until the matched six-condition test is complete.
