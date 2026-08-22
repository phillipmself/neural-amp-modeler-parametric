"""
Silence-in / silence-out training anchors.

A captured device produces no output when its input is silent, whatever its controls are
doing. That holds unconditionally, so it can be trained on directly without any capture
data: feed the net digital silence, hold the controls somewhere, and score the output
against zero.

The anchors exist because the concat architectures carry the encoded controls as input
channels, so those channels pass through the whole receptive field. Training only ever
presented a control setting that was constant across a window, which constrains the sum of
the model's per-lag control sensitivity but not how that sum is distributed across lags.
Anchoring silence pins the distribution as well. Holding the controls still pins the
model's DC-vs-control map; moving them pins how that sensitivity is spread across lags,
which is what a listener hears as a click or a smear when a knob is turned.

The moving anchor covers the still case too (see ``hold_probability``), so a net that
accepts a param trajectory needs only that one. A net that generates its weights from the
control vector as a whole cannot be handed a trajectory at all and is left with the static
anchor.
"""

import math as _math
from collections.abc import Sequence as _Sequence
from functools import lru_cache as _lru_cache
from typing import NamedTuple as _NamedTuple
from typing import Optional as _Optional

import torch as _torch

from ._base import ParametricNet as _ParametricNet
from ._spec import ParamSpec as _ParamSpec


class AnchorWindow(_NamedTuple):
    """The silent window one anchor batch is run on; see :func:`anchor_window`."""


    #: Samples the net is run on, history and warmup included.
    length: int
    #: Earliest sample whose value can reach the scored tail.
    first_influencing: int
    #: Scored output samples, the trailing slice of the window.
    ny: int


def sample_raw_params(
    param_specs: _Sequence[_ParamSpec],
    n: int,
    *,
    rail_probability: float = 0.0,
    device: _Optional[_torch.device] = None,
    generator: _Optional[_torch.Generator] = None,
) -> _torch.Tensor:
    """
    Draw ``n`` raw (un-encoded) control vectors uniformly over the declared ranges.

    Continuous params are uniform on ``[min, max]``; switches are uniform over their enum
    indices. Returns ``(n, P)`` in declared order, ready for ``ParametricNet.forward``.

    ``rail_probability`` pulls each continuous control onto one of its two extremes that
    often instead. Users park knobs fully off and fully up far more often than uniform
    sampling would, and the rails are where a model has the least capture data on both
    sides.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1; got {n}")
    if not 0.0 <= rail_probability <= 1.0:
        raise ValueError(f"rail_probability must be within [0, 1]; got {rail_probability}")
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
            if rail_probability > 0.0:
                on_rail = (
                    _torch.rand((n, 1), device=device, generator=generator)
                    < rail_probability
                )
                high = _torch.rand((n, 1), device=device, generator=generator) < 0.5
                column = _torch.where(
                    on_rail,
                    _torch.where(
                        high,
                        _torch.full_like(column, spec.max),
                        _torch.full_like(column, spec.min),
                    ),
                    column,
                )
        columns.append(column)
    return _torch.cat(columns, dim=1)


def anchor_window(net: _ParametricNet, ny: int) -> AnchorWindow:
    """
    The silent window a net has to run to get ``ny`` scored output samples out of it.

    ``first_influencing`` is the earliest sample in that window whose value can reach the
    scored tail, which is where a moving-control gesture is aimed. It sits at the end of
    the warmup rather than at the start of the window because the receptive field ahead
    of the scored tail is what carries an earlier sample into it. For a net whose history
    lives in a convolution the boundary is hard; for a recurrent one it is soft, since an
    earlier sample still arrives through hidden state.

    ``training_warmup`` can depend on whether the net is in training mode, so a window is
    only valid for the mode it was computed in. :func:`anchor_output` re-derives it and
    will reject a trajectory sampled against the other one.
    """
    if ny < 1:
        raise ValueError(f"ny must be at least 1; got {ny}")
    warmup = net.training_warmup
    return AnchorWindow(
        length=net.receptive_field - 1 + warmup + ny, first_influencing=warmup, ny=ny
    )


def anchor_output(
    net: _ParametricNet,
    params: _torch.Tensor,
    ny: int,
) -> _torch.Tensor:
    """
    Run ``net`` on digital silence at the given control settings and return the
    ``(B, ny)`` scored tail of the output the device would answer with silence.

    ``params`` is ``(B, P)`` for a held setting or ``(B, T, P)`` for a trajectory, where
    ``T`` must be the full :func:`anchor_window` length: the input is built here, so a
    trajectory has to have been sampled against the same window.

    The input is built here and fed with ``pad_start=False``, so the model's history is
    part of the window rather than something padded on afterwards. Two kinds of history
    have to be budgeted for, and only one of them is the receptive field: a recurrent net
    also runs a detached burn-in whose outputs carry no gradient, so ``ny`` samples of
    window would leave nothing scored at all. ``training_warmup`` covers that, and the
    warmup samples are dropped from the return so the score is taken only where the
    gradient actually flows.
    """
    if params.ndim not in (2, 3):
        raise ValueError(
            f"Expected params to have shape (B, P) or (B, T, P); got {tuple(params.shape)}"
        )
    batch_size = params.shape[0]
    length = anchor_window(net, ny).length
    if params.ndim == 3 and params.shape[1] != length:
        raise ValueError(
            f"Expected a param trajectory of length {length} for ny={ny}; "
            f"got {params.shape[1]}"
        )
    parameter = next(net.parameters())
    x = _torch.zeros(
        (batch_size, length), device=parameter.device, dtype=parameter.dtype
    )
    if net.requires_uniform_batch_params:
        if params.ndim == 3:
            raise ValueError(f"{type(net).__name__} cannot evaluate a param trajectory")
        # This net runs a whole batch under one generated weight set, so a batch of
        # distinct settings would be silently evaluated at the first row's setting. The
        # per-row call passes a 1-D setting, which takes the net's unbatched path and so
        # leaves its batched-uniformity check untouched for the real training batches.
        y = _torch.cat(
            [net(x[i : i + 1], params[i], pad_start=False) for i in range(batch_size)],
            dim=0,
        )
    else:
        y = net(x, params, pad_start=False)
    if y.shape[-1] < ny:
        raise RuntimeError(
            f"{type(net).__name__} returned {y.shape[-1]} samples for an anchor window "
            f"sized to score {ny}; its history requirements are larger than "
            "receptive_field and training_warmup report"
        )
    return y[..., -ny:]


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
    window: AnchorWindow,
    *,
    min_window_travel: float,
    max_window_travel: float,
    rail_probability: float = 0.25,
    hold_probability: float = 0.25,
    device: _Optional[_torch.device] = None,
    generator: _Optional[_torch.Generator] = None,
) -> _torch.Tensor:
    """
    Draw ``n`` raw control trajectories over ``window``, shaped ``(n, window.length, P)``.

    Each trajectory models one control gesture as the runtime renders it: a single commit
    instant at which every switch steps to its new index and every moving continuous
    control begins a linear ramp of a shared duration. Switches step rather than blend
    because a blended one-hot is a conditioning vector no model was trained on, which is
    why the runtime excludes switch channels from its smoothing.

    Trajectories are built in raw units and encoded per frame downstream. For a continuous
    control the encoding is affine, so a raw-linear ramp is an encoded-linear one; for a
    switch a raw index step encodes to a one-hot step at the same instant. Both match the
    runtime, which ramps in the encoded domain.

    Gesture speed is drawn as ``window travel``: the fraction of a control's full range a
    ramp covers over the scored tail, log-uniform between the two bounds. Sizing it
    against the window rather than in seconds is what keeps the term meaningful, because
    the window is short -- a few tens of milliseconds for a recurrent net -- and a
    duration drawn in seconds would put most gestures far too slow to move a control
    measurably before the scored tail closed, leaving the term paying for a trajectory
    and scoring a held setting. Travel above 1 is a gesture that finishes inside the
    scored tail, which is where the settling tail and the slope discontinuity that ends
    the ramp get scored.

    The tail is ``ny`` samples for every architecture, which is what makes one config
    mean one gesture. Sizing against the whole influencing span instead would hand a
    convolutional net a reach of ``receptive_field - 1 + ny`` against a recurrent net's
    ``ny`` -- a thirteenfold difference at the shipping sizes -- so the same numbers
    would train the two on gestures an order of magnitude apart in speed.

    ``hold_probability`` of the rows do not move at all. Those rows are the static anchor:
    they pin the model's DC-vs-control map, which a moving gesture on its own does not.

    The commit instant is drawn from ``[first_influencing - ramp, length)``. A ramp that
    resolves before the window's scored reach opens leaves little for the loss to see;
    reaching back by one ramp is what lets a gesture longer than the scored tail finish
    inside it. (For a recurrent net the reach is soft rather than hard -- an early gesture
    still arrives through hidden state -- but aiming at the tail is what makes it count.)
    """
    if not 0.0 < min_window_travel <= max_window_travel:
        raise ValueError(
            "Window travel must satisfy 0 < min_window_travel <= max_window_travel; got "
            f"{min_window_travel} and {max_window_travel}"
        )
    if not 0.0 <= hold_probability <= 1.0:
        raise ValueError(f"hold_probability must be within [0, 1]; got {hold_probability}")

    param_specs = tuple(param_specs)
    start = sample_raw_params(
        param_specs,
        n,
        rail_probability=rail_probability,
        device=device,
        generator=generator,
    )
    end = sample_raw_params(
        param_specs,
        n,
        rail_probability=rail_probability,
        device=device,
        generator=generator,
    )

    # Hold a random subset of the controls still, so single-control moves are covered
    # alongside the everything-at-once case. At least one always moves.
    held = _torch.rand((n, len(param_specs)), device=device, generator=generator) < 0.5
    moved = _torch.randint(len(param_specs), (n, 1), device=device, generator=generator)
    held.scatter_(1, moved, False)
    end = _torch.where(held, start, end)
    end = _force_movement(param_specs, start, end, moved, device, generator)
    if hold_probability > 0.0:
        hold = _torch.rand((n, 1), device=device, generator=generator) < hold_probability
        end = _torch.where(hold, start, end)

    ramp = _sample_ramp_samples(
        n,
        window.ny,
        min_window_travel=min_window_travel,
        max_window_travel=max_window_travel,
        device=device,
        generator=generator,
    )
    # Only a row with a continuous control in motion reaches back: a ramp caught partway
    # through at the window's first influencing sample is a gesture the loss can still
    # see. A switch steps instantaneously, so a commit before that sample leaves it
    # sitting at its destination for the whole window -- a row that pays for a trajectory
    # forward and shows the net nothing moving.
    moves_continuously = (
        ((start != end) & ~_switch_mask(param_specs, device)[None, :])
        .any(dim=1, keepdim=True)
        .to(ramp.dtype)
    )
    earliest = window.first_influencing - ramp * moves_continuously
    commit = earliest + (window.length - earliest) * _torch.rand(
        (n, 1), device=device, generator=generator
    )
    return _build_trajectory(param_specs, start, end, commit, ramp, window.length, device)


def _sample_ramp_samples(
    n: int,
    ny: int,
    *,
    min_window_travel: float,
    max_window_travel: float,
    device: _Optional[_torch.device] = None,
    generator: _Optional[_torch.Generator] = None,
) -> _torch.Tensor:
    """Draw ``n`` ramp durations in samples, shaped ``(n, 1)``.

    Log-uniform in travel: a gesture's slope spans orders of magnitude, and sampling it
    uniformly would put almost every draw at one end.
    """
    log_min, log_max = _math.log(min_window_travel), _math.log(max_window_travel)
    span = _torch.rand((n, 1), device=device, generator=generator)
    travel = _torch.exp(log_min + (log_max - log_min) * span)
    return (ny / travel).clamp(min=1.0)


@_lru_cache(maxsize=None)
def _switch_mask(
    param_specs: tuple[_ParamSpec, ...], device: _Optional[_torch.device]
) -> _torch.Tensor:
    # Cached because building it from a Python list is a host-to-device copy, and this
    # runs once per training step.
    return _torch.tensor([spec.type == "switch" for spec in param_specs], device=device)


def _build_trajectory(
    param_specs: _Sequence[_ParamSpec],
    start: _torch.Tensor,
    end: _torch.Tensor,
    commit: _torch.Tensor,
    ramp: _torch.Tensor,
    length: int,
    device: _Optional[_torch.device] = None,
) -> _torch.Tensor:
    """
    One control gesture per row as ``(n, length, P)``, in raw units.

    ``commit`` is the sample index at which the gesture begins and ``ramp`` its duration,
    both ``(n, 1)`` and both free to fall outside ``[0, length)`` so that a gesture can
    begin before the window opens or land after it closes.
    """
    t = _torch.arange(length, device=device, dtype=ramp.dtype)[None, :]
    fraction = ((t - commit) / ramp).clamp(0.0, 1.0)[:, :, None]
    stepped = (t >= commit)[:, :, None]
    is_switch = _switch_mask(tuple(param_specs), device)[None, None, :]
    return _torch.where(
        is_switch,
        _torch.where(stepped, end[:, None, :], start[:, None, :]),
        start[:, None, :] + fraction * (end - start)[:, None, :],
    )


def _force_movement(
    param_specs: _Sequence[_ParamSpec],
    start: _torch.Tensor,
    end: _torch.Tensor,
    moved: _torch.Tensor,
    device: _Optional[_torch.device],
    generator: _Optional[_torch.Generator],
) -> _torch.Tensor:
    """Make the control singled out to move actually land somewhere else.

    Its endpoints are drawn independently, so it can be handed its own starting value: a
    three-way switch draws its own index one time in three, and a continuous control
    stalls whenever both draws are pulled onto the same rail. Such a row pays for a
    trajectory forward and scores a held setting, which ``hold_probability`` already
    covers deliberately and in a known proportion.

    A stuck switch steps to another index; a stuck continuous control sweeps to the rail
    furthest from where it started, which is the largest gesture available to it.
    """
    n = start.shape[0]
    end = end.clone()
    for i, spec in enumerate(param_specs):
        column = start[:, i]
        stuck = (moved[:, 0] == i) & (end[:, i] == column)
        if spec.type == "switch":
            step = 1 + _torch.randint(
                spec.num_inputs - 1, (n,), device=device, generator=generator
            )
            unstuck = (column + step) % spec.num_inputs
        else:
            midpoint = 0.5 * (spec.min + spec.max)
            unstuck = _torch.where(
                column > midpoint,
                _torch.full_like(column, spec.min),
                _torch.full_like(column, spec.max),
            )
        end[:, i] = _torch.where(stuck, unstuck, end[:, i])
    return end
