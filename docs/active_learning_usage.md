# Active-learning capture selection (PANAMA-style)

> **Attribution.** This workflow is adapted from **PANAMA** (Parametric Active-learning for Neural Amp
> Modeling Assistance), a NAM fork published as arXiv
> [2509.26564v1](https://arxiv.org/html/2509.26564v1). The round-based query-by-committee loop, the
> disagreement (output-variance) acquisition objective, the `sigmoid(latent)` control-vector
> optimization, and the threshold-grouping clustering are all due to the PANAMA authors. We
> re-implement the *approach* here for a different model, single-GPU, and our `data.json` format.

## What this is for

When building a **parametric** capture set (one knob-conditioned amp model trained on many knob
settings) the hard question is *which settings to capture next*. Guessing wastes reamping time.
Active learning answers it: train a small **ensemble** of cheap models on the captures collected so
far, then find the control settings where the ensemble members **disagree most** (query by
committee) and propose those as the next captures.

The ensemble member is a **ConcatLSTM** — a parametric LSTM that tiles the encoded control vector
across time and concatenates it to the audio. It is a **disposable acquisition proxy**: its only job
is to produce a cheap, differentiable disagreement signal over the control space. The model you
actually ship is still the **HyperWaveNet**; the LSTM ensemble is thrown away each round.

Design details and rationale live in
[`docs/panama_active_learning_lstm_plan.md`](panama_active_learning_lstm_plan.md).

## Example configs

Ready-to-edit examples (Gain/Tone continuous 0–10, Boost switch Off/On) live in
[`nam_full_configs/active_learning/`](../nam_full_configs/active_learning/):

- `model.json` — `net.name = "ConcatLSTM"`, with architecture/loss/optimizer mirroring PANAMA's LSTM
  (3 layers, hidden_size 18, `train_burn_in`/`mask_first` 8192, pre-emph MRSTFT loss, lr 0.008). Its
  `params` block **must match** the params of the parametric production model you are capturing for.
- `learning.json` — PANAMA's active-learning regime (`batch_size` 512, 50 epochs). The accelerator is
  rewritten at runtime to `cuda → mps → cpu`, so it is device-agnostic. Unlike PANAMA (which skips
  in-loop validation), our `train_ensemble` keeps validation on to pick the best checkpoint. With a
  small starter set, lower `batch_size`/disable `drop_last` so batches aren't all dropped.
- `data.json` — a 10-setting starter set produced by the script below, used as the round-0 seed.

## Workflow

### Step 1 — Starter set (round 0 seed)

Generate the first ~10 settings with a Latin-hypercube over the continuous knobs and balanced switch
assignment:

```bash
python scripts/make_starter_settings.py \
  --model-config nam_full_configs/active_learning/model.json \
  --n 10 --n-validation 2 --ny 32768 \
  --output data.json
```

Useful flags: `--input-wav` (the reamp input, default `input.wav`), `--full-grid` (LHS-continuous ×
every switch combination), `--seed`, `--y-path-prefix`, `--no-rounding` (skip capture-grid
quantization), the `--start-seconds`/`--stop-seconds`/`--ny` window controls, and their
`--validation-*` variants. Continuous values are snapped to the realizable knob grid (default 0.5)
so the recorded setting equals the setting a human can actually dial.

The script prints a capture checklist. **Reamp `input.wav` at each listed setting**, save each output
to its `y_path` wav, and you have a trainable round-0 `data.json`.

### Step 2 — One active-learning round

One CLI invocation == one round:

```bash
python scripts/active_learn.py \
  --round-idx 0 --output-dir al_runs \
  --data-config data.json \
  --model-config nam_full_configs/active_learning/model.json \
  --learning-config nam_full_configs/active_learning/learning.json \
  --g-opt-input-wav input.wav
```

Each round:

1. Trains a 4-member ConcatLSTM ensemble **serially** on one device (different seed per member).
2. Runs the disagreement g-optimizer: for every switch combination it Adam-**ascends** a latent `z`
   (mapped to in-range continuous knob values) to maximize member-output variance.
3. Clusters the candidates per switch combination, quantizes survivors to the capture grid, dedupes,
   and takes the global top `--max-per-round`.
4. Writes, in `--output-dir`:
   - `proposed_captures_round_{i}.json` — the proposed settings in **user units** (0–10, enum names),
     with suggested `y_path` filenames, plus a printed checklist.
   - `aggregated_data_config_{i}.json` — the previous `data.json` with the proposals appended to
     `train` (placeholder `y_path`s), `common` and `validation` preserved.

Then it **stops**. You reamp `input.wav` at the proposed settings, fill in the `y_path`s in
`aggregated_data_config_{i}.json`, and run the next round:

```bash
python scripts/active_learn.py \
  --round-idx 1 --output-dir al_runs \
  --model-config nam_full_configs/active_learning/model.json \
  --learning-config nam_full_configs/active_learning/learning.json
```

For `--round-idx i > 0` the driver defaults `--data-config` to
`<output-dir>/aggregated_data_config_{i-1}.json`, so you only pass `--data-config` explicitly for
round 0.

Other useful flags: `--ensemble-size` (default 4), `--num-restarts`, `--num-steps`,
`--g-opt-ny`/`--g-opt-batch-size`, `--use-mel` (PANAMA's multi-resolution mel-variance term),
`--seed`, `--ckpts` (reuse member checkpoints instead of retraining), `--no-plot`.

### Step 3 — Train the production model

Once you have grown the capture set, train the shipped **HyperWaveNet** on the aggregated `data.json`
exactly as for any parametric model (see `nam_full_configs/parametric/`). The LSTM ensemble was only
ever an acquisition tool and is not part of the final model.
