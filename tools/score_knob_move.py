"""
Score a parametric model on the two things a knob move goes wrong in.

Both numbers come from rendering the same model twice and differencing, so they need no
ground truth and have no measurement noise floor to fight -- unlike val ESR, which is
measured entirely on parked controls and cannot see either effect.

  moving-vs-quasi-static   the artifact heard *during* a move. The model renders the
                           window once with a per-sample control trajectory and once
                           quasi-statically (every output sample computed with the control
                           frozen at its own value across the whole receptive field). The
                           difference is control motion and nothing else. This is what the
                           quasi-static anchor drives down.

  settling                 the artifact heard *after* a move has landed. A gesture lands a
                           given margin before the scored window, and the render is
                           compared against a pure static render at the destination. The
                           hardware settles in milliseconds, so anything left at 20 ms is
                           the model's own. This is what landed-move augmentation drives
                           down.

Both are reported in dB relative to the reference render's RMS, so more negative is
better. Run it on each training arm and compare columns.

Usage:
    python tools/score_knob_move.py MODEL.nam --input input_train.wav
    python tools/score_knob_move.py a.nam b.nam c.nam --input input_train.wav
"""

import argparse as _argparse
import json as _json
import math as _math
from pathlib import Path as _Path
from typing import List as _List
from typing import Optional as _Optional
from typing import Sequence as _Sequence
from typing import Tuple as _Tuple

import numpy as _np
import torch as _torch

from nam.models import factory as _factory
from nam.models.parametric import ParametricNet as _ParametricNet

# Moves worth scoring: a big gain sweep, a tone-only move, and a small nudge. Endpoints are
# fractions of each control's range so the same set works whatever the model's controls are.
_DEFAULT_MOVES: _Tuple[_Tuple[_Tuple[float, ...], _Tuple[float, ...]], ...] = (
    ((0.95, 0.10), (0.50, 0.45)),
    ((0.60, 0.60), (0.55, 0.65)),
    ((0.90, 0.10), (0.80, 0.05)),
    ((0.05, 0.50), (0.95, 0.50)),
)
_DEFAULT_RAMPS_MS = (50.0, 200.0)
_DEFAULT_MARGINS_MS = (0.0, 5.0, 20.0, 50.0)


def load_model(path: _Path) -> _ParametricNet:
    """Rebuild the PyTorch net from an exported .nam, weights and all."""
    blob = _json.loads(path.read_text())
    net = _factory.init(blob["architecture"], args=(blob["config"],))
    if not isinstance(net, _ParametricNet):
        raise SystemExit(f"{path} holds a {blob['architecture']}, which is not parametric")
    consumed = net.import_weights(_torch.tensor(blob["weights"], dtype=_torch.float32))
    if consumed != len(blob["weights"]):
        raise SystemExit(
            f"{path}: consumed {consumed} of {len(blob['weights'])} weights"
        )
    net.sample_rate = blob["sample_rate"]
    net.eval()
    return net


def _settings(net: _ParametricNet, fractions: _Sequence[float]) -> _torch.Tensor:
    """Turn per-control fractions of range into raw control values."""
    specs = net.param_specs
    if len(fractions) != len(specs):
        fractions = tuple(fractions) + (0.5,) * (len(specs) - len(fractions))
    values = []
    for spec, fraction in zip(specs, fractions):
        value = spec.min + (spec.max - spec.min) * float(fraction)
        values.append(round(value) if spec.type == "switch" else value)
    return _torch.tensor([values], dtype=_torch.float32)


def _db(numerator: _torch.Tensor, reference: _torch.Tensor) -> float:
    """RMS of `numerator` in dB relative to the RMS of `reference`."""
    den = reference.pow(2).mean().sqrt()
    if den <= 0.0:
        return float("nan")
    num = numerator.pow(2).mean().sqrt()
    if num <= 0.0:
        return float("-inf")
    return 20.0 * _math.log10(float(num / den))


def _ramp(
    start: _torch.Tensor,
    end: _torch.Tensor,
    length: int,
    commit: float,
    ramp_samples: float,
    specs,
) -> _torch.Tensor:
    """One linear gesture as (1, length, P); switches step at the commit instant."""
    t = _torch.arange(length, dtype=_torch.float32)[None, :]
    fraction = ((t - commit) / max(ramp_samples, 1.0)).clamp(0.0, 1.0)[:, :, None]
    stepped = (t >= commit)[:, :, None]
    is_switch = _torch.tensor([s.type == "switch" for s in specs])[None, None, :]
    return _torch.where(
        is_switch,
        _torch.where(stepped, end[:, None, :], start[:, None, :]),
        start[:, None, :] + fraction * (end - start)[:, None, :],
    )


@_torch.no_grad()
def _quasi_static(
    net: _ParametricNet,
    x: _torch.Tensor,
    trajectory: _torch.Tensor,
    ny: int,
    chunk: int,
) -> _torch.Tensor:
    """Per-sample frozen-control render: output sample j uses trajectory[j] throughout.

    Exact rather than block-quantised -- this is a measurement, not a training term, so it
    should not inherit the block floor the anchor trades cost against.
    """
    history = net.receptive_field - 1
    # Sample j is scored from x[j : j + history + 1], i.e. a stride-1 unfold.
    windows = x.unfold(1, history + 1, 1)[0]
    frozen = trajectory[0, history : history + ny]
    out = []
    for i in range(0, ny, chunk):
        out.append(net(windows[i : i + chunk], frozen[i : i + chunk], pad_start=False))
    return _torch.cat(out, dim=0).reshape(1, ny)


@_torch.no_grad()
def moving_vs_quasi_static(
    net: _ParametricNet,
    audio: _torch.Tensor,
    moves,
    ramps_ms: _Sequence[float],
    scored: int,
    chunk: int,
) -> _List[dict]:
    """The artifact during a move: trajectory render minus quasi-static render."""
    rate = float(net.sample_rate)
    history = net.receptive_field - 1
    length = history + scored
    x = audio[None, :length]
    rows = []
    for a_fraction, b_fraction in moves:
        a, b = _settings(net, a_fraction), _settings(net, b_fraction)
        for ramp_ms in ramps_ms:
            ramp_samples = ramp_ms * rate / 1000.0
            # The move begins as the scored window opens, so the whole ramp is in flight.
            trajectory = _ramp(a, b, length, history, ramp_samples, net.param_specs)
            moving = net(x, trajectory, pad_start=False)
            reference = _quasi_static(net, x, trajectory, scored, chunk)
            rows.append(
                {
                    "move": f"{_fmt(a)} -> {_fmt(b)}",
                    "ramp_ms": ramp_ms,
                    "db": _db(moving - reference, reference),
                }
            )
    return rows


@_torch.no_grad()
def settling(
    net: _ParametricNet,
    audio: _torch.Tensor,
    moves,
    ramps_ms: _Sequence[float],
    margins_ms: _Sequence[float],
    scored: int,
) -> _List[dict]:
    """The artifact after a move: landed render minus a pure static render at the target."""
    rate = float(net.sample_rate)
    history = net.receptive_field - 1
    length = history + scored
    x = audio[None, :length]
    rows = []
    for a_fraction, b_fraction in moves:
        a, b = _settings(net, a_fraction), _settings(net, b_fraction)
        static = net(x, b, pad_start=False)
        for ramp_ms in ramps_ms:
            ramp_samples = ramp_ms * rate / 1000.0
            for margin_ms in margins_ms:
                margin = margin_ms * rate / 1000.0
                # Land `margin` before the first scored sample.
                commit = (history - margin) - ramp_samples
                trajectory = _ramp(a, b, length, commit, ramp_samples, net.param_specs)
                landed = net(x, trajectory, pad_start=False)
                rows.append(
                    {
                        "move": f"{_fmt(a)} -> {_fmt(b)}",
                        "ramp_ms": ramp_ms,
                        "margin_ms": margin_ms,
                        "db": _db(landed - static, static),
                    }
                )
    return rows


def _fmt(setting: _torch.Tensor) -> str:
    return "/".join(f"{v:g}" for v in setting[0].tolist())


def _load_audio(path: _Path, rate: float, needed: int, offset: int) -> _torch.Tensor:
    import wave

    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != int(rate):
            raise SystemExit(
                f"{path} is {handle.getframerate()} Hz but the model is {int(rate)} Hz"
            )
        handle.setpos(min(offset, max(handle.getnframes() - needed, 0)))
        frames = handle.readframes(needed)
        width, channels = handle.getsampwidth(), handle.getnchannels()
    if width == 2:
        data = _np.frombuffer(frames, dtype="<i2").astype(_np.float32) / (1 << 15)
    elif width == 3:
        # 24-bit has no numpy dtype; sign-extend each triple into int32.
        raw = _np.frombuffer(frames, dtype=_np.uint8).reshape(-1, 3).astype(_np.int32)
        packed = raw[:, 0] | (raw[:, 1] << 8) | (raw[:, 2] << 16)
        data = _np.where(packed & 0x800000, packed - (1 << 24), packed).astype(
            _np.float32
        ) / (1 << 23)
    elif width == 4:
        data = _np.frombuffer(frames, dtype="<i4").astype(_np.float32) / (1 << 31)
    else:
        raise SystemExit(f"{path}: unsupported sample width {width * 8}-bit")
    if channels > 1:
        data = data.reshape(-1, channels)[:, 0]
    if len(data) < needed:
        raise SystemExit(f"{path} has {len(data)} samples; {needed} needed")
    return _torch.from_numpy(data.copy())


def _print(title: str, rows: _List[dict], keys: _Sequence[str], models: _List[str]):
    print(f"\n{title}")
    width = max(len(str(row[keys[0]])) for row in rows)
    header = "  ".join(f"{k:>9}" for k in keys[1:])
    print(f"  {'':{width}}  {header}  " + "  ".join(f"{m:>9}" for m in models))
    seen = set()
    for row in rows:
        key = tuple(row[k] for k in keys)
        if key in seen:
            continue
        seen.add(key)
        label = f"{row[keys[0]]:{width}}"
        rest = "  ".join(f"{row[k]:>9g}" for k in keys[1:])
        values = "  ".join(f"{v:>9.1f}" for v in row["db_by_model"])
        print(f"  {label}  {rest}  {values}")


def main(argv: _Optional[_Sequence[str]] = None) -> None:
    parser = _argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", type=_Path, help="exported .nam file(s)")
    parser.add_argument("--input", type=_Path, required=True, help="16-bit PCM wav")
    parser.add_argument("--offset", type=int, default=48000 * 10)
    parser.add_argument(
        "--scored", type=int, default=4096, help="output samples scored per render"
    )
    parser.add_argument(
        "--chunk", type=int, default=256, help="rows per quasi-static forward"
    )
    args = parser.parse_args(argv)

    nets = [load_model(path) for path in args.models]
    names = [path.stem[:9] for path in args.models]
    rate = float(nets[0].sample_rate)
    needed = max(net.receptive_field for net in nets) - 1 + args.scored
    audio = _load_audio(args.input, rate, needed, args.offset)

    moving = [
        moving_vs_quasi_static(
            net, audio, _DEFAULT_MOVES, _DEFAULT_RAMPS_MS, args.scored, args.chunk
        )
        for net in nets
    ]
    settle = [
        settling(
            net,
            audio,
            _DEFAULT_MOVES,
            _DEFAULT_RAMPS_MS,
            _DEFAULT_MARGINS_MS,
            args.scored,
        )
        for net in nets
    ]
    for group in (moving, settle):
        for i, row in enumerate(group[0]):
            row["db_by_model"] = [g[i]["db"] for g in group]

    print(f"input {args.input.name} @ {int(rate)} Hz, {args.scored} samples scored")
    print("dB relative to the reference render -- more negative is better")
    _print("during a move (trajectory vs quasi-static)", moving[0], ("move", "ramp_ms"), names)
    _print("after a move (landed vs static)", settle[0], ("move", "ramp_ms", "margin_ms"), names)


if __name__ == "__main__":
    main()
