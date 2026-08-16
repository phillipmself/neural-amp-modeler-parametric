# Knob-move training work — handoff

Implements §6 of `training/forClaude/FINDINGS_training_side_ramp_noise.md`. Nothing has been
trained yet; everything below is built, verified to run, and pushed.

## Branches

Shared code on `feature/reduce-concat-dc`, merged into `feature/parametric-film-wavenet`
(which adds the two FiLM-only commits). Both pushed.

```
677432a  Run both silence anchors in one forward          optimization
b1daf5b  Compile FiLMWaveNet's moving-control forward      optimization (film branch only)
44a0bd3  Train the captures with the knob already in motion  §6.1 landed-move
56a3ea6  Anchor a moving control against the model's own static render  §6.2 quasi-static
9c4e2b2  Add a scoring tool for the two knob-move artifacts
ae4bdf0  Refuse the moving-control terms on a recurrent net
aef2a74  Let FiLMWaveNet run the moving-control terms      (film branch only)
1724032  Let a recurrent net land a move in its loss mask
```

## What was added

**`landed_move`** (top-level config block). Replaces a capture's held control with a gesture
that lands 20-100 ms before the first scored sample. No new loss term and no new target — the
capture is already correct once the move has landed. Room to land = convolutional history
(`nx - 1`) + `loss.mask_first`. Training only. Works on FiLMWaveNet, ConcatWaveNet, ConcatLSTM.

**`loss.quasi_static_anchor`**. Scores a moving-control render against the model's own
frozen-control render, block-quantised (`block: 32`). Zero static error by construction, so the
residual is all dynamic artifact. Training only. **FIR nets only** — see limits.

**Optimizations.** The two silence anchors now share one forward when their `ny` match
(bit-exact; falls back to two otherwise), zero-weight anchors are skipped, and FiLM's
trajectory path is compiled instead of running eager. That last one was the cause of the
added training time: with `torch_compile.mode: reduce-overhead`, `silence_anchor_ramp` was the
only uncompiled forward in an otherwise CUDA-graphed step.

**`tools/score_knob_move.py`**. Reads an exported `.nam` and prints two tables in dB: *during a
move* (trajectory vs per-sample frozen render) and *after a move* (landed vs static). Takes
several models to print them as columns. This is the score — val ESR is measured entirely on
parked controls and cannot see either artifact.

```bash
python tools/score_knob_move.py a.nam b.nam c.nam d.nam --input input_train.wav
```

## Run configs

Eight arms in `training/SD1/`, each copied byte-identical from the corresponding proven run
except for the variable under test. Reuse `learning_film.json` / `learning_lstm.json` unchanged.

FiLM (from `sd1_film_silence`; silence anchors on in all four):
`model_film_a_baseline` · `model_film_b_landed` · `model_film_c_quasi` · `model_film_d_both`

ConcatLSTM (from `SD1_lstm`; 2x2, because silence anchors have never run on it):
`model_lstm_a_baseline` · `model_lstm_b_silence` · `model_lstm_c_landed` · `model_lstm_d_both`

### Running them

`training/SD1/run_arms.sh <gpu>` (lives with the configs, not in this repo) runs one GPU's
half — two FiLM arms, then two LSTM arms. Launch one per GPU. It resolves configs from its own
directory; `DATA_DIR` / `OUT_DIR` / `REPO_DIR` override the defaults.

```bash
./run_arms.sh 0 &
./run_arms.sh 1 &
```

Arms are paired so each GPU takes one cheap and one expensive run of each architecture, since
the quasi-static anchor adds two forwards per step on FiLM and the silence anchors add ~25% on
the LSTM:

```
GPU 0   film a_baseline → film d_both → lstm a_baseline → lstm d_both
GPU 1   film b_landed   → film c_quasi → lstm c_landed   → lstm b_silence
```

## ConcatLSTM speed settings

The `SD1_lstm` run took 2x the FiLM run. An LSTM's cost is sequential cell evaluations per
epoch, which works out to `total_samples / batch_size` — **independent of `ny`**, so batch size
is the only real lever. At `batch_size: 8` a 4090 is idle: 4 layers x 32 hidden is ~140k MACs
per timestep. Applied:

```
learning_lstm.json    train batch_size 8 -> 32      233 -> 58 steps/epoch, 7.6M -> 1.9M
                      val   batch_size 2 -> 6       validation was 3 sequential passes, now 1
                      benchmark false -> true
model_lstm_*.json     train_truncate  null -> 8192  bounds BPTT depth and memory
                      train_burn_in   null -> 8192  no backward through the masked prefix
                      lr 0.01 -> 0.02               sqrt scaling for 4x fewer optimizer steps
                      lr_scheduler.frequency 100 -> 25   decay is per-step; keeps the same
                                                          per-epoch schedule (~10% of initial
                                                          by epoch 200, not ~56%)
```

Expect ~4-5x faster epochs. **Reasoned from the code and the arithmetic above, not measured on
a GPU.**

Things worth knowing here:

- **Truncation does not reduce how much is trained.** Verified: forward output is bit-identical
  (0.000e+00), same 24576 scored samples, same 30369/30369 parameters receiving gradient. Only
  the backward *reach* is capped at 8192 samples (170 ms) — far beyond the pedal's physical
  memory, and the AL work found the long chain was what amplified gradients ~3e17x.
- **Truncation does not cost the cuDNN fast path** in normal training. The
  `_g_opt_cuda_train_mode_safe` caveat lives only in `active_learning.py` (the g-opt member of
  `find_disagreement_settings`). In `_run_conditioned` a set truncate is just several
  `nn.LSTM` calls instead of one, and `_L` subclasses `nn.LSTM`, so each chunk is still fused.
- **`train_burn_in` is silently ignored unless `train_truncate` is set** — see the
  `if not self.training or self._train_truncate is None:` branch in `_concat_lstm.py`.
- Batch size is the one change that genuinely trades something: 4x fewer weight updates per
  epoch. That is what the `lr` and scheduler changes compensate for. If training ESR is worse
  than the old run at matched epochs, drop `lr` to 0.015; a collapse (train ESR toward 1.0) is
  the documented AL failure mode and `lr` is the first thing to pull.

## Weights — calibrate on gradient norm, not loss value

Both anchors are L1, whose gradient magnitude is independent of its own value, so comparing
loss values understates the imbalance. Measured ratios at weight 1.0 against the capture loss:

| | at weight 1.0 | chosen |
|---|---|---|
| FiLM quasi-static, trained ckpt | **335x** capture | 0.006 |
| FiLM silence, at init (weight 1.0 is proven) | 48x | 1.0 |
| FiLM silence, trained with anchors | 4x | — |
| LSTM silence, ckpt that never had anchors | **1240x** | 0.04 |

`0.006` matches the silence anchors' pull on FiLM. `0.04` matches FiLM's *starting* 48x, since
weight 1.0 is proven there. Least certain number in the set: the LSTM 1240x came from a
converged checkpoint (small capture gradient, untouched silence output), so 0.04 probably errs
weak — the safe direction, since L1 keeps pulling. If arm B barely moves, try 0.1.

## Limits

- **The quasi-static anchor cannot work on ConcatLSTM** and is refused. Its reference must
  cold-start each block, and on the trained SD1 LSTM that lead-in error is -25 dB at 50 ms of
  warm-up and -37 dB at 200 ms, decaying ~5 dB per doubling — at or above the -34 to -49 dB
  artifact it exists to measure. Carrying the hidden state instead removes the lead-in error
  but contaminates the reference with the very motion being detected. Both horns fail.
- **`score_knob_move`'s "during a move" table is invalid for recurrent nets** for the same
  reason. Its "after a move" table is fine. For during-move on the LSTM use the silence-flick
  from `training/plumes/Gain_2val/measure_condition_kernel.py`.
- On a freshly initialised net the quasi-static term reads ~0 and the two silence terms read
  equal, because at init the silence output has zero spread across controls. The gradient is
  non-zero, so the term is live. Not a reason to kill an early run.
- Arms with `landed_move` put the *main* forward on the trajectory path, so its graph is built
  fresh on the first step. Don't read early steps as steady-state throughput.

## Open

- `SD1_lstm` has `train_truncate: null` (full BPTT) and `train_burn_in: null`. `mask_first:
  8192` is doing the burn-in job, so this is a cost question, not correctness: roughly a third
  of the backward pass is computed and discarded. Unknown whether the setting was deliberate.
- `test_set_compiled_round_trips_to_the_eager_path` fails under some file orderings. Confirmed
  pre-existing on a clean tree.
- Quasi-static weight for ConcatWaveNet is unmeasured; `0.006` is FiLM-specific.
