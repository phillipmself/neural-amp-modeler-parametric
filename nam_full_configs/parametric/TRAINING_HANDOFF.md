# Parametric Training Handoff

This note summarizes the recent parametric training experiments, what changed between runs, and what the results suggest. zip artifacts are in the ~/Downloads folder.

## Scope

- Dataset/splits discussed below are the low/high parametric runs with:
  - train grid at `low, high ∈ {-6, 0, +6}`
  - held-out verification points at `(-3, +3)` and `(+3, -3)`
- Metrics called out here are primarily:
  - `ESR_seen_audio_seen_param`
  - `ESR_unseen_audio_unseen_param`
- Log summaries were produced with:
  - [scripts/analyze_training_log.py](/Users/phillipself/dev/audio/parametric_nam_trio/nam_parametric/scripts/analyze_training_log.py)

## High-Level Conclusions

1. The original parametric data split appears sufficient to expose the problem.
2. The main failure mode is not basic audio fitting. It is poor generalization to unseen parameter settings.
3. More raw capacity tends to improve seen buckets faster, but often hurts `unseen_audio_unseen_param`.
4. The most reliable recent win has been:
   - sandwiched adapter
   - trunk LR `0.001`
   - adapter LR `1e-4`
   - adapter hidden dim `4`
   - gamma/beta scale `0.02`
5. Lowering adapter LR below `1e-4` was too much and caused under-learning.
6. Global weight decay `1e-5` was too strong.
7. Adapter-specific weight decay `1e-6` smoothed seen-bucket training a bit, but still regressed deployment quality.

## Best Current Reference Run

Current best recent run:

- `sandwiched_4dim_0.02scale_0.0001adapterLr.zip`
- Settings:
  - trunk LR `0.001`
  - adapter LR `1e-4`
  - trunk channels `8`
  - adapter hidden dim `4`
  - gamma scale `0.02`
  - beta scale `0.02`
- Best metrics:
  - `ESR_seen_audio_seen_param = 0.012583232`
  - `ESR_unseen_audio_seen_param = 0.010031974`
  - `ESR_seen_audio_unseen_param = 0.036380906`
  - `ESR_unseen_audio_unseen_param = 0.033325300`
- Best deployment epoch:
  - `62`
- Best primary epoch:
  - `65`

This is the most balanced recent run because the best deployment and primary epochs are close together and the held-out parameter bucket is the strongest among the recent sandwiched/bounded runs.

## Training Acceptance Criteria

These criteria are meant to answer a practical question:

- when is a parametric run good enough to count as an acceptable network configuration?

The comparison point is a known-good standard packed NAM training run:

- best overall `ESR = 0.000933442`
- best packed buckets:
  - `ESR_packed_0 = 0.000525156`
  - `ESR_packed_1 = 0.000408286`

That standard run is much easier than the parametric problem, so the raw ESR values are not expected to match directly. The useful lesson is that a good standard run is both accurate and balanced across validation slices.

For parametric runs, use the following acceptance thresholds:

1. Evaluate acceptance from the best checkpoint, not the final epoch.
2. Best-checkpoint `ESR_unseen_audio_seen_param` must be at or below `0.01`.
3. `ESR_unseen_audio_unseen_param` should be at or below `0.04` to count as minimally acceptable.
4. `ESR_unseen_audio_unseen_param` at or below `0.035` is the current “good enough to stop searching aggressively” target.
5. Reject runs whose best `ESR_unseen_audio_unseen_param` is worse than `0.05`.
6. Prefer runs whose best deployment epoch is not extremely early. Very early peaks usually indicate that seen-bucket fitting is outrunning parameter generalization.
7. Prefer runs whose final deployment metric stays within about `1.3x` of the best deployment checkpoint.

Useful ratio targets:

$$
\frac{ESR_{\text{unseen audio, unseen param}}}{ESR_{\text{unseen audio, seen param}}} \le 2.5
$$

Stretch goal:

$$
\frac{ESR_{\text{unseen audio, unseen param}}}{ESR_{\text{unseen audio, seen param}}} \le 2.0
$$

Interpretation:

- `unseen_audio_seen_param <= 0.01` is non-negotiable because that is the standard NAM acceptance axis for unseen audio behavior.
- `unseen_audio_unseen_param` is the extra parametric burden, so this bucket still matters as the main deployment metric.
- A run is not acceptable if it only improves seen settings while remaining weak on unseen audio or unseen parameter settings.

## Experiment Log

### 1. Early Parametric Baseline Comparison

Files:

- `low_high_full_cov_norm_half_param_lr.zip`
- `low_high_full_coverage_normalized_param_input.zip`

Main observation:

- Both runs used the same train/validation/verification split.
- Changing adapter LR did not fix the core issue.
- Both fit seen settings very well, but plateaued much higher on unseen parameter settings.

Best deployment bucket:

- `same_lr`: `0.008049928`
- `half_adapter_lr`: `0.008334979`

Interpretation:

- The main problem looked like conditioning/generalization, not a lack of data coverage.

### 2. Shared MLP Adapter, Unbounded

Files:

- `shared_mlp_0.0005lr.zip`
- `shared_mlp_0.002lr.zip`

Main observation:

- These runs were unstable.
- ESR spiked into very large values during training.

Best deployment bucket:

- `shared_mlp_0.0005lr`: `0.010148469`
- `shared_mlp_0.002lr`: `0.013642834`

But the important issue was instability:

- `0.0005lr` had spikes near `25`
- `0.002lr` had spikes near `49`

Interpretation:

- Unconstrained per-layer modulation was too strong.

### 3. Shared MLP Adapter, Bounded

Files:

- `shared_mlp_bounded_adptr_0.0005_trunk_0.001.zip`
- `shared_mlp_bounded_adptr_0.0005.zip`
- `shared_mlp_bounded_adptr_0.002.zip`

Common setup:

- bounded gamma/beta
- hidden dim `8`
- gamma scale `0.05`
- beta scale `0.05`

Main observation:

- Bounding removed the catastrophic blow-ups.
- But held-out parameter generalization was still not good.

Best deployment bucket:

- `0.0005_trunk_0.001`: `0.024598654`
- `0.0005`: `0.036218159`
- `0.002`: `0.023996271`

Important behavior:

- `0.0005_trunk_0.001` had very strong seen-bucket behavior but its deployment bucket peaked very early and then lagged.
- Lower trunk LR helped stability and seen fitting more than it helped deployment generalization.

Interpretation:

- Bounded modulation fixed instability, but not the core interpolation/generalization issue.

### 4. Sandwiched Adapter, Hidden Dim 4

Files:

- `sandwiched_4dim_0.02scale_0.0001adapterLr.zip`
- `sandwiched_4dim_0.02scale.zip`
- `sandwiched_4dim_0.05scale.zip`

Common setup:

- trunk LR `0.001`
- hidden dim `4`
- sandwiched adapter variant

Differences:

- adapter LR `1e-4` vs `5e-4`
- scale `0.02` vs `0.05`

Best deployment bucket:

- `0.02scale_0.0001adapterLr`: `0.033325300`
- `0.02scale`: `0.053010799`
- `0.05scale`: `0.051055484`

Interpretation:

- Lowering adapter LR from `5e-4` to `1e-4` helped a lot.
- Changing scale from `0.02` to `0.05` did not help nearly as much.
- `1e-4` looked like a real sweet spot here.

### 5. Sandwiched Adapter, Hidden Dim 4, Adapter LR `5e-5`

File:

- `sandwiched_4dim_0.02scale_5e-5adapterLr.zip`

Best deployment bucket:

- `0.052225292`

Main observation:

- This was clearly worse than the `1e-4` run.
- The adapter looked too weak to learn enough.

Interpretation:

- `5e-5` is too low for the adapter in this setup.
- Lower than `1e-4` is not promising on this axis.

### 6. Global Weight Decay Increase

File:

- `weightDecay1e-5.zip`

Difference from best reference:

- same sandwiched `4`-dim setup
- same trunk LR `0.001`
- same adapter LR `1e-4`
- global `weight_decay = 1e-5`

Best deployment bucket:

- `0.041942202`

Interpretation:

- This was worse across all buckets.
- Global `1e-5` weight decay was too strong and caused underfitting.

### 7. Dedicated Adapter Weight Decay

File:

- `adapterWeightDecay1e-6.zip`

Difference from best reference:

- same best sandwiched setup
- add `adapter_weight_decay = 1e-6`

Best metrics:

- `ESR_seen_audio_seen_param = 0.011900375`
- `ESR_unseen_audio_seen_param = 0.009089713`
- `ESR_seen_audio_unseen_param = 0.040380094`
- `ESR_unseen_audio_unseen_param = 0.037161212`

Interpretation:

- Seen buckets got a little smoother/better.
- Deployment bucket regressed versus the current best.
- Adapter-specific decay is a meaningful knob, but `1e-6` did not beat the current best overall.

### 8. More Trunk Channels

File:

- `10channels.zip`

Difference from best reference:

- trunk channels `10` instead of `8`
- hidden dim `4`
- scale `0.02`
- trunk LR `0.001`
- adapter LR `1e-4`

Best metrics:

- `ESR_seen_audio_seen_param = 0.010654652`
- `ESR_unseen_audio_seen_param = 0.008835251`
- `ESR_seen_audio_unseen_param = 0.044526182`
- `ESR_unseen_audio_unseen_param = 0.045763362`

Interpretation:

- More channels helped seen buckets fit faster and lower.
- It hurt deployment generalization.
- This looks like more memorization capacity, not better interpolation over parameters.

### 9. Smaller Adapter Hidden Dim

File:

- `dim2.zip`

Difference from best reference:

- hidden dim `2` instead of `4`

Best metrics:

- `ESR_seen_audio_seen_param = 0.067919306`
- `ESR_unseen_audio_seen_param = 0.052428655`
- `ESR_seen_audio_unseen_param = 0.054364879`
- `ESR_unseen_audio_unseen_param = 0.050206147`

Interpretation:

- Hidden dim `2` underfit badly.
- This was too little adapter capacity.

## What Seems Less Promising Now

- Lower adapter LR than `1e-4`
- Larger global weight decay
- Much smaller adapter hidden dim
- More trunk channels, if the goal is held-out parameter generalization
- Adapting only the last `7` layers
- Faster trunk LR `0.002` with partial-layer adapters

## Current Working Hypothesis

The remaining problem does not look like a simple “more capacity” or “less capacity” issue.

What the experiments suggest:

- Too much distributed power across the network helps seen settings more than unseen parameter interpolation.
- More trunk capacity makes the model fit known settings faster, but degrades held-out parameter behavior.
- Too little adapter capacity or too little adapter LR causes under-learning.
- Early-layer conditioning appears much more important than late-layer-only conditioning.

That points toward improving early and mid-network conditioning quality, not simply adding capacity or moving the adapter toward the output.

## Suggested Next Experiments

Highest-signal next experiment:

1. Keep the current best recipe:
   - channels `8`
   - hidden dim `4`
   - gamma/beta scale `0.02`
   - trunk LR `0.001`
   - adapter LR `1e-4`
2. Sweep early-layer coverage instead of late-layer-only coverage.
3. First try:
   - adapt the first `10` layers
   - adapt the first `12` layers
   - adapt the first `14` layers

Why this is the leading next idea:

- `first_14` was the only partial-layer variant that stayed close to the full-adapter baseline.
- `last_7` was clearly worse, which argues against shifting modulation toward the output only.
- `first_7` found a decent early deployment minimum but drifted badly later, which suggests the boundary is probably not as small as `7`.

Follow-up experiments after that:

1. Add stronger early stopping or checkpoint selection based on both `ESR_unseen_audio_seen_param` and `ESR_unseen_audio_unseen_param`
2. Try a slightly larger early-layer slice if `first_14` remains close but still misses the unseen-audio target
3. Revisit small adapter-only decay changes only if a structurally better adapter placement is found first

## Files to Keep in Mind

- Current best recent run:
  - `sandwiched_4dim_0.02scale_0.0001adapterLr.zip`
- Smoother but worse deployment:
  - `adapterWeightDecay1e-6.zip`
- “More capacity helps seen buckets, hurts deployment” example:
  - `10channels.zip`
- “Too little adapter capacity” example:
  - `dim2.zip`
