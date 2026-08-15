"""
Landed-move augmentation for the capture examples.

A capture is recorded with its controls parked, so training only ever presents a control
that is constant across the window. That constrains the sum of the model's per-lag
control sensitivity but not how that sum is distributed across lags, and the model ends up
with sensitivity to control values tens of milliseconds old -- a term the hardware has no
counterpart for, since a knob is a resistance and the circuit's response to changing it is
bounded by time constants in the milliseconds.

The fix needs no new ground truth. The device's output at a setting does not depend on
where the knob was 20 ms ago, so the capture is already the correct target for a window
whose move has *landed* before the first scored sample. Replacing the constant control
with a gesture that arrives in time therefore trains the settling behaviour against real
hardware, using the captures the loss already owns.

This penalizes only the tail after a move, never the noise during one; the quasi-static
anchor covers that case.
"""

from collections.abc import Sequence as _Sequence
from typing import Optional as _Optional

import torch as _torch

from ._anchors import _sample_endpoints
from ._anchors import build_trajectory as _build_trajectory
from ._anchors import sample_ramp_samples as _sample_ramp_samples
from ._spec import ParamSpec as _ParamSpec


def sample_landed_trajectories(
    param_specs: _Sequence[_ParamSpec],
    destination: _torch.Tensor,
    length: int,
    land_by: int,
    sample_rate: float,
    *,
    min_ramp_seconds: float,
    max_ramp_seconds: float,
    min_margin_seconds: float,
    max_margin_seconds: float,
    rail_probability: float = 0.25,
    probability: float = 1.0,
    generator: _Optional[_torch.Generator] = None,
) -> _torch.Tensor:
    """
    Control trajectories that arrive at ``destination`` before the scored window opens.

    ``destination`` is the capture's own ``(B, P)`` setting and the value every row holds
    from its landing instant onwards, so the capture stays the correct target. ``land_by``
    is the index of the first scored sample; each row lands a margin earlier than that,
    drawn per row rather than annealed on a schedule so that one run covers the range.

    Rows not selected by ``probability`` come back as the constant ``destination``
    trajectory, which is bit-identical to passing the held setting -- the plain
    constant-control path stays intact, and the whole batch can still share one forward.
    """
    if destination.ndim != 2:
        raise ValueError(
            f"Expected destination of shape (B, P); got {tuple(destination.shape)}"
        )
    if length < 1:
        raise ValueError(f"length must be at least 1; got {length}")
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"probability must be within [0, 1]; got {probability}")
    if not 0.0 <= min_margin_seconds <= max_margin_seconds:
        raise ValueError(
            "Margins must satisfy 0 <= min_margin_seconds <= max_margin_seconds; got "
            f"{min_margin_seconds} and {max_margin_seconds}"
        )
    if sample_rate <= 0.0:
        raise ValueError(f"sample_rate must be positive; got {sample_rate}")

    param_specs = tuple(param_specs)
    n = destination.shape[0]
    device = destination.device

    def rand(*shape: int) -> _torch.Tensor:
        return _torch.rand(shape, device=device, generator=generator)

    origin = _sample_endpoints(param_specs, n, rail_probability, device, generator)
    # Hold a random subset of the controls at the destination, so single-control moves are
    # covered alongside the everything-at-once case.
    held = rand(n, len(param_specs)) < 0.5
    moved_index = _torch.randint(
        len(param_specs), (n, 1), device=device, generator=generator
    )
    held.scatter_(1, moved_index, False)
    origin = _torch.where(held, destination, origin)

    ramp = _sample_ramp_samples(
        n,
        sample_rate,
        min_ramp_seconds=min_ramp_seconds,
        max_ramp_seconds=max_ramp_seconds,
        device=device,
        generator=generator,
    )
    margin = (
        min_margin_seconds
        + (max_margin_seconds - min_margin_seconds) * rand(n, 1)
    ) * sample_rate
    # The gesture ends `margin` before the first scored sample, so it starts a whole ramp
    # earlier again. A commit before the window opens is fine: the row simply enters
    # part-way along its ramp, which is what a move already in flight looks like.
    commit = (land_by - margin) - ramp

    trajectory = _build_trajectory(
        param_specs, origin, destination, commit, ramp, length, device
    )
    if probability >= 1.0:
        return trajectory
    selected = (rand(n, 1) < probability)[:, :, None]
    return _torch.where(
        selected, trajectory, destination[:, None, :].expand(-1, length, -1)
    )
