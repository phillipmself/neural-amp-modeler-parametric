"""
Silence-in / silence-out training anchors.

A captured device produces no output when its input is silent, whatever its controls are
doing. That holds unconditionally, so it can be trained on directly without any capture
data: feed the net digital silence, hold or move the controls, and score the output
against zero.

The anchors exist because the concat architectures carry the encoded controls as input
channels, so those channels pass through the whole receptive field. Training only ever
presented a control setting that was constant across a window, which constrains the sum of
the model's per-lag control sensitivity but not how that sum is distributed across lags.
Anchoring silence pins the distribution as well.
"""

import math as _math
from collections.abc import Sequence as _Sequence
from typing import Optional as _Optional

import torch as _torch

from ._base import ParametricNet as _ParametricNet
from ._spec import ParamSpec as _ParamSpec


def sample_raw_params(
    param_specs: _Sequence[_ParamSpec],
    n: int,
    *,
    device: _Optional[_torch.device] = None,
    generator: _Optional[_torch.Generator] = None,
) -> _torch.Tensor:
    """
    Draw ``n`` raw (un-encoded) control vectors uniformly over the declared ranges.

    Continuous params are uniform on ``[min, max]``; switches are uniform over their enum
    indices. Returns ``(n, P)`` in declared order, ready for ``ParametricNet.forward``.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1; got {n}")
    columns = []
    for spec in param_specs:
        if spec.type == "switch":
            column = _torch.randint(
                spec.num_inputs,
                (n, 1),
                device=device,
                generator=generator,
                dtype=_torch.float32,
            )
        else:
            column = spec.min + (spec.max - spec.min) * _torch.rand(
                (n, 1), device=device, generator=generator, dtype=_torch.float32
            )
        columns.append(column)
    return _torch.cat(columns, dim=1)


def anchor_output(
    net: _ParametricNet,
    params: _torch.Tensor,
    ny: int,
) -> _torch.Tensor:
    """
    Run ``net`` on digital silence at the given control settings and return the ``(B, ny)``
    output the device would answer with silence.

    ``params`` is ``(B, P)`` for a held setting or ``(B, T, P)`` for a trajectory, where
    ``T`` must be the full ``receptive_field - 1 + ny`` window: the input is built here and
    fed with ``pad_start=False``, so the receptive-field history is part of the window
    rather than something padded on afterwards.
    """
    if ny < 1:
        raise ValueError(f"ny must be at least 1; got {ny}")
    if params.ndim not in (2, 3):
        raise ValueError(
            f"Expected params to have shape (B, P) or (B, T, P); got {tuple(params.shape)}"
        )
    batch_size = params.shape[0]
    length = net.receptive_field - 1 + ny
    if params.ndim == 3 and params.shape[1] != length:
        raise ValueError(
            f"Expected a param trajectory of length {length} for ny={ny}; "
            f"got {params.shape[1]}"
        )
    device = next(net.parameters()).device
    x = _torch.zeros((batch_size, length), device=device)
    if net.requires_uniform_batch_params:
        if params.ndim == 3:
            raise ValueError(
                f"{type(net).__name__} cannot evaluate a param trajectory"
            )
        # This net runs a whole batch under one generated weight set, so a batch of
        # distinct settings would be silently evaluated at the first row's setting. The
        # per-row call passes a 1-D setting, which takes the net's unbatched path and so
        # leaves its batched-uniformity check untouched for the real training batches.
        return _torch.cat(
            [net(x[i : i + 1], params[i], pad_start=False) for i in range(batch_size)],
            dim=0,
        )
    return net(x, params, pad_start=False)


def as_held_trajectory(params: _torch.Tensor, length: int) -> _torch.Tensor:
    """
    Present a held ``(B, P)`` setting as the constant ``(B, length, P)`` trajectory.

    A control that never moves is a trajectory like any other, which is what lets a held
    anchor share a forward with a moving one. The result is a stride-0 view, so the
    expansion costs nothing until something materializes it.
    """
    if params.ndim != 2:
        raise ValueError(
            f"Expected a held setting of shape (B, P); got {tuple(params.shape)}"
        )
    if length < 1:
        raise ValueError(f"length must be at least 1; got {length}")
    return params[:, None, :].expand(-1, length, -1)


def sample_param_trajectories(
    param_specs: _Sequence[_ParamSpec],
    n: int,
    length: int,
    sample_rate: float,
    *,
    min_ramp_seconds: float,
    max_ramp_seconds: float,
    rail_probability: float = 0.25,
    device: _Optional[_torch.device] = None,
    generator: _Optional[_torch.Generator] = None,
) -> _torch.Tensor:
    """
    Draw ``n`` raw control trajectories of ``length`` samples, shaped ``(n, length, P)``.

    Each trajectory models one control gesture as the runtime renders it: a single commit
    instant at which every switch steps to its new index and every moving continuous
    control begins a linear ramp of a shared duration. Switches step rather than blend
    because a blended one-hot is a conditioning vector no model was trained on, which is
    why the runtime excludes switch channels from its smoothing.

    Trajectories are built in raw units and encoded per frame downstream. For a continuous
    control the encoding is affine, so a raw-linear ramp is an encoded-linear one; for a
    switch a raw index step encodes to a one-hot step at the same instant. Both match the
    runtime, which ramps in the encoded domain.

    The commit instant is drawn from ``[-ramp, length)`` rather than ``[0, length)``. A
    ramp is longer than a short anchor window, so only a gesture that began before the
    window opens can finish inside it -- which is the case where the settling tail, and
    the slope discontinuity that ends the ramp, are scored.
    """
    if length < 1:
        raise ValueError(f"length must be at least 1; got {length}")
    if sample_rate <= 0.0:
        raise ValueError(f"sample_rate must be positive; got {sample_rate}")
    if not 0.0 < min_ramp_seconds <= max_ramp_seconds:
        raise ValueError(
            "Ramp durations must satisfy 0 < min_ramp_seconds <= max_ramp_seconds; got "
            f"{min_ramp_seconds} and {max_ramp_seconds}"
        )
    if not 0.0 <= rail_probability <= 1.0:
        raise ValueError(
            f"rail_probability must be within [0, 1]; got {rail_probability}"
        )

    param_specs = tuple(param_specs)
    start = _sample_endpoints(param_specs, n, rail_probability, device, generator)
    end = _sample_endpoints(param_specs, n, rail_probability, device, generator)

    def rand(*shape: int) -> _torch.Tensor:
        return _torch.rand(shape, device=device, generator=generator)

    # Hold a random non-empty subset of the controls still, so single-control moves are
    # covered alongside the everything-at-once case.
    held = rand(n, len(param_specs)) < 0.5
    moved_index = _torch.randint(
        len(param_specs), (n, 1), device=device, generator=generator
    )
    held.scatter_(1, moved_index, False)
    end = _torch.where(held, start, end)

    # Log-uniform: a gesture's slope spans orders of magnitude, and sampling the duration
    # uniformly would put almost every draw at the slow end.
    log_min, log_max = _math.log(min_ramp_seconds), _math.log(max_ramp_seconds)
    ramp = _torch.exp(log_min + (log_max - log_min) * rand(n, 1)) * sample_rate
    ramp = ramp.clamp(min=1.0)
    commit = -ramp + (ramp + length) * rand(n, 1)

    t = _torch.arange(length, device=device, dtype=ramp.dtype)[None, :]
    fraction = ((t - commit) / ramp).clamp(0.0, 1.0)[:, :, None]
    stepped = (t >= commit)[:, :, None]

    is_switch = _torch.tensor(
        [spec.type == "switch" for spec in param_specs], device=device
    )[None, None, :]
    return _torch.where(
        is_switch,
        _torch.where(stepped, end[:, None, :], start[:, None, :]),
        start[:, None, :] + fraction * (end - start)[:, None, :],
    )


def _sample_endpoints(
    param_specs: _Sequence[_ParamSpec],
    n: int,
    rail_probability: float,
    device: _Optional[_torch.device],
    generator: _Optional[_torch.Generator],
) -> _torch.Tensor:
    """Uniform draws, with continuous controls pulled onto a rail some of the time.

    Users park knobs fully off and fully up far more often than uniform sampling would,
    and the rails are where a model has the least capture data on both sides.
    """
    values = sample_raw_params(param_specs, n, device=device, generator=generator)
    if rail_probability == 0.0:
        return values
    for i, spec in enumerate(param_specs):
        if spec.type == "switch":
            continue
        on_rail = _torch.rand(n, device=device, generator=generator) < rail_probability
        high = _torch.rand(n, device=device, generator=generator) < 0.5
        rail = _torch.where(
            high,
            _torch.full((n,), spec.max, device=device),
            _torch.full((n,), spec.min, device=device),
        )
        values[:, i] = _torch.where(on_rail, rail, values[:, i])
    return values
