# FiLMWaveNet

A parametric WaveNet whose controls act only through FiLM. Branch: `feature/parametric-film-wavenet`.

Companion docs: [`parametric_conditioning_handoff.md`](parametric_conditioning_handoff.md) for the
conditioning investigation this came out of, [`parametric_control_warp.md`](parametric_control_warp.md)
for the knob-taper work the param encoder overlaps with.

---

## 1. Why

`ConcatWaveNet` makes the encoded controls input channels. They enter the residual stream and are
then convolved by every dilated kernel downstream. In every training example those channels are
constant in time, so a kernel spanning a knob move sees a tap pattern it was never trained on. On
the shipped 5-knob topology that window is the full receptive field: 6347 samples, **132 ms**.

Ramping the control channels does not fix this. The runtime already walks continuous controls
across the block (`ConcatConditioner`, core #5), and ConcatWaveNet still produces prominent,
audible low-end disturbances on a knob move. That is the expected result if the problem is the
kernel straddle rather than the step: slewing changes how fast the network crosses the
invalid-configuration window, not whether it is in one.

FiLM's condition-to-scale/shift map is a 1x1 convolution — no time extent. If the FiLM condition is
the control vector alone, the control value at a sample affects that sample's modulation and
nothing else. No weight is ever exposed to a non-constant control pattern, so there is no
invalid-configuration window at any rate of change, including instantaneous. The controls never
generate audio; they only scale audio the network already produced.

Audio state still washes out over the receptive field after a move. That is unavoidable and is what
a real amp does; the difference is that during the washout the network computes the exact trained
function for the *current* setting, applied to state left over from the old one.

## 2. What falls out of it

**The inner WaveNet is shape-identical to a stock, non-parametric one** (`input_size` and
`condition_size` are 1). It does not need the widened channels `ConcatWaveNet` uses to carry control
information:

| model | params | MACs/sample |
|---|---|---|
| stock, channels 8, cond 1 | 12,145 | 11,832 |
| shipped ConcatWaveNet, channels 16, cond 6 (5 knobs) | 49,121 | 48,496 |
| FiLMWaveNet, channels 8, one cached FiLM site + encoder (5 knobs) | 14,033 | ~12,000 |

Roughly **3.5x cheaper than the shipped concat model** at the same kernel/dilation schedule, and the
per-sample figure does not move with the knob count or the encoder width — only the model size does.

**A runtime computes each FiLM's scale/shift once per control change, not once per frame**, because
the condition is constant between moves. Per-frame cost is the fused multiply-add alone. Training
passes the control with a length-1 time axis and lets it broadcast, which is the same arithmetic.

**Identity init means the model starts as a plain WaveNet**, so an existing single-setting capture
can be loaded straight into it and training only learns the modulation.

## 3. Config

```json
{
  "net": {
    "name": "FiLMWaveNet",
    "config": {
      "layers": [{ "channels": 8, "kernel_sizes": [...], "dilations": [...],
                   "activation": "LeakyReLU", "head": {...},
                   "activation_post_film": {"active": true, "shift": false, "groups": 1} }],
      "head_scale": 0.01,
      "param_encoder": {"hidden_sizes": [16], "out_features": 8, "activation": "ReLU"},
      "params": [ ...ParamSpec... ],
      "identity_init": true
    }
  }
}
```

`nam.capture.export.build_film_model_config` generates this from a capture project;
`write_film_training_configs` writes `model_film.json` / `learning_film.json` next to the
ConcatWaveNet and HyperWaveNet pairs.

- `input_size`, `condition_size` and `film_condition_size` are **derived** — omit them, or a
  mismatched declared value is rejected.
- At least one FiLM site must be active, or the controls have no way into the network.
- `param_encoder` may be `null`, in which case the FiLM condition is the encoded control vector
  itself. FiLM is linear in its condition, so without an encoder a knob maps linearly to a channel
  gain, which does not match measured knob tapers.
- `param_encoder.out_features` is a free choice, **not** a function of the knob count — see
  [Sizing the FiLM condition](#sizing-the-film-condition).
- `identity_init` defaults to true.
- `condition_dsp`, packed and slimmable layer arrays are rejected.

### Sizing the FiLM condition

`out_features` is the width of the vector every FiLM in the network reads. Each layer computes its
own gains from it as `W_i · e(k) + b_i`, so with `m = out_features` the network's *entire* knob
response — 23 layers x 8 channels = 184 gains on the shipped topology — is 184 fixed linear
combinations of the same `m` functions of the knobs.

The consequence is that gains cannot disagree about what a knob does. At `m = 1` every gain in the
network is the same curve, scaled and possibly inverted: if one channel's gain rises monotonically,
none can rise and then fall. The encoder decides how fast you travel along a fixed path in gain
space; `m` decides how much that path can bend.

Setting it to the encoded control dim is therefore a *lower* bound dressed up as a default: it gives
the whole network one shape per knob. `nam.capture.export` uses `_FILM_ENCODER_OUT_FEATURES = 8`
instead, decoupled from the knob count. Widening is cheap — the encoder runs once per control change
and the FiLM 1x1 is cached with it, so per-sample audio cost does not move at all:

| `out_features` | FiLM weights | encoder weights | total (5 knobs) |
|---|---|---|---|
| 5 (= encoded control dim) | 1,104 | 181 | 13,430 |
| 8 (default) | 1,656 | 232 | 14,033 |
| 16 | 3,128 | 368 | 15,641 |
| 32 | 6,072 | 640 | 18,857 |

Still well under ConcatWaveNet's 49,121 at every width. If FiLM underperforms on tone-stack knobs,
this is the first thing to sweep.

### Choosing FiLM sites

`activation_post_film` with `shift=false` is the default: modulating the bottleneck after the
nonlinearity is the most expressive single site, and scale-only avoids injecting a control-derived
DC into the residual stream — the one part of FiLM that a downstream dilated kernel still straddles
when a knob moves. `input_mixin_pre_film` is the cheapest alternative (a per-layer input gain).
`head1x1_post_film` alone confines control response to the head's kernel span if very short control
latency matters more than expressivity.

## 4. Export contract

Architecture name `FiLMWaveNet`, version namespace `1.0.0` (`PARAMETRIC_MODEL_VERSION`).

`config` carries the stock WaveNet keys (`layers`, `head`, `head_scale`) plus `params` and
`param_encoder`. Every layer array declares `film_condition_size`, including when it happens to equal
`condition_size`; a reader should derive the width from `param_encoder.out_features` (or the encoded
control dim) and treat the declared value as a check.

Weights, in order:

1. **Param encoder**, if present: per `nn.Linear` layer, the weight matrix (out x in, row-major)
   then the bias. Layer sizes are `[encoded_param_dim, *hidden_sizes, out_features]`.
2. **The inner WaveNet blob**, byte-identical in layout to a stock WaveNet, ending with `head_scale`.

The encoder comes first so a reader can size it from the config before walking the WaveNet's
variable-length blob.

## 5. Evaluating a control move

`FiLMWaveNet.forward_control_sequence(x, params)` takes a per-sample control sequence, which is what
a knob turn looks like and costs nothing extra here. It is also the shape training augmentation on
knob moves would use.

The architectural invariant, pinned by
`tests/parametric/test_film_wavenet.py::test_a_control_move_settles_within_exactly_the_receptive_field`:
outside one receptive field of a move the output is *exactly* the corresponding steady state, and the
transition is confined to that window.

Note that `ConcatWaveNet` satisfies that same invariant — causality gives it for free. The
difference between the two is what the network computes *inside* the window, which is a property of
the trained weights, and that is where concat's audible disturbance lives. Sizing the improvement
needs a trained A/B: same capture set, concat-16 (ramped, as shipped) versus FiLM-8 initialized from
a nominal-setting static capture, comparing knob-step transient energy — particularly below ~200 Hz,
where the concat artifact is most audible — against the steady-state reference.

## 6. Runtime smoothing

The runtime ramps a committed control over `kDefaultFiLMRampSeconds` (10 ms), **in scale/shift
space** rather than on the control vector. `GetParams()` still reflects a committed value
immediately; what is smoothed is when it arrives at the audio.

Without this, the control only updates at block boundaries, so the per-channel gain FiLM applies
steps once per block — zipper noise, scaled by the signal level and by the block size (a 512-sample
block gives steps 8x larger than a 64-sample one). Interpolating per sample makes that gain
piecewise-linear instead, which removes the discontinuity outright; the ramp length then only
bounds the slew rate.

Ramping downstream of the encoder rather than on the control has three consequences:

- **The 1x1 still runs once per control change, not per frame.** Each FiLM caches a current and a
  target scale/shift column and interpolates between them; the per-sample cost is one extra add per
  channel while a ramp is in flight, and nothing once it lands.
- **Switches keep their exact one-hot.** The encoder is never fed a blended index — the blend
  happens after it, between two gain vectors the model was trained at. `ConcatWaveNet` has to
  exclude switches from its ramp (`ParamRamp`'s mask) for exactly the reason this avoids.
- **The settled state is untouched.** Once the ramp lands, the code path is the one that existed
  before smoothing, and the current column is snapped to the committed target rather than left on
  the accumulated value, so no residual survives. This is why the Python-parity fixtures still
  match byte-for-byte.

10 ms is much shorter than `ConcatWaveNet`'s 200 ms, deliberately. Concat needs a long ramp to
cross its receptive-field-long window of control patterns it was never trained on *slowly*;
FiLMWaveNet has no such window, so every instant of the ramp is a configuration the model was
trained at and the ramp only has to outrun the step. The value was picked from that reasoning, not
by ear — no FiLMWaveNet has been trained on a real capture yet. Tune it with
`FiLMWaveNet::SetParamRampSeconds()`; 0 disables smoothing entirely, which is how a stream restart
settles and how the architecture tests isolate propagation from settling.

## 7. The knob-move artifact that remains

Measured on a trained SD1 capture (2 knobs, `_FILM_ENCODER_OUT_FEATURES = 8`), turning TONE at
DRIVE=10 leaves an audible thump. It is *not* zippering, and the runtime ramp does not reach it.

Feed the model digital silence and the output is a constant, not zero — std `4.7e-10`, so pure DC —
and its level depends on the controls (`-0.00213` at TONE=0, `-0.00417` at TONE=10). Zeroing the
model's 72 bias tensors makes that output **exactly** 0.000 at every setting, which locates the
source: the conv biases manufacture a DC pedestal, and scale-only FiLM happily modulates it. Moving
a control moves the pedestal, and it re-settles through the 132 ms receptive field, overshooting by
~9x the endpoint difference at DRIVE=10 versus ~3x at DRIVE=0 — hence the drive dependence.

For scale, the same measurement on the shipped ConcatWaveNet for the same pedal:

| | 10 ms ramp | 50 ms | 200 ms |
|---|---|---|---|
| FiLMWaveNet | −34.8 dBFS | −45.8 | −47.1 |
| ConcatWaveNet | −15.1 dBFS | −18.8 | −24.1 |

FiLM is ~20 dB better and responds far more to ramping, because it has no invalid-configuration
window for a ramp to cross slowly — only the DC re-settling. What does **not** work, all measured:

- **A DC blocker.** 3.7 dB at a 10 ms ramp. Half the peak excursion is above 100 Hz — the energy is
  at DC but the audible edge is not, and a 20 Hz high-pass is unity gain there.
- **Subtracting the static `DC(knob)`.** Worse than leaving it alone: the subtraction is instant
  while the model's DC lags a receptive field behind it.
- **Subtracting a parallel instance fed silence.** Cancels perfectly in exact digital silence and is
  8–13 dB *worse* with any signal present, even at −34 dBFS. Output DC is a function of the signal
  as well as the controls, so the silent instance computes a different, uncorrelated artifact.

What works is removing the pedestal in training: `loss.silence_anchor` scores the model's output on
silence against zero, and `supports_param_trajectory` lets `silence_anchor_ramp` do the same while
the controls move. The anchor is scored with mean absolute error, so its logged value is directly
the artifact amplitude in linear units — on this model it starts around `2.6e-3`.

## 8. Status

Trainability confirmed on a synthetic gain task (`y = tanh(g(knob) x)`): FiLMWaveNet reached eval ESR
0.0014 at 2,357 params against 0.0033 for a comparably sized ConcatWaveNet. That task maximally
favours multiplicative modulation, so it establishes that the architecture trains and learns knob
dependence — not that it wins on real captures.

The expressivity risk is real and untested on captures: scale-only FiLM on 8 channels is a much
smaller control surface than concat-16 or the hypernetwork, and tone-stack knobs change filtering
rather than gain. Mitigations in increasing cost: a wider `out_features` (see
[Sizing the FiLM condition](#sizing-the-film-condition) — nearly free, sweep this first), more FiLM
sites, `shift=true`, a modest channel bump — all still well under the concat model's 4x.

Exposed in the capture app behind `nam-capture --filmwavenet`, matching how HyperWaveNet is gated,
until a real capture run says whether it holds up.
