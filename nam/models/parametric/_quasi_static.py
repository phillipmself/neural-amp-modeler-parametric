"""
Quasi-static consistency anchor: the noise a control makes *while* it moves.

The landed-move augmentation trains what happens once a gesture has arrived, and the
silence anchors pin the control's lag extent where there is no signal to modulate. Neither
sees the artifact during the move itself, on real audio.

Scoring that needs a target for an in-flight control, and the obvious one -- crossfading
the two nearest captures -- does not work. Measured against real in-between captures a
crossfade sits about 22 dB below signal, which is roughly the model's own static fit
error, while the moving-control artifact is 34 to 49 dB down. The artifact is buried 15 to
31 dB inside the target's own error, so training on it would spend the gradient
re-litigating static response from a target no better than the captures already in the
loss.

The model's own frozen-control render has no such problem. Its static error is zero by
construction -- it *is* the model's static response, which the capture loss already
supervises -- so the residual is all dynamic artifact and no static mismatch, the exact
inverse of the crossfade. It also needs no capture pairing, no nearest-neighbour map and
no crossfade law, and never sums two captures, so comb filtering cannot arise.

The term cannot be gamed by killing knob authority: the reference still varies with the
controls, and the capture loss pins that variation. It is zero exactly when the model's
control sensitivity has no lag extent, which is the target behaviour with full authority
intact.

Freezing the reference per sub-block rather than per sample is what makes this affordable,
and the block width sets the floor the loss can drive the artifact down to. Measured
against a per-8-sample reference, a block of 32 sits 30-43 dB below the artifact and a
block of 128 sits 20-26 dB below; 512 is too coarse for a fast ramp. The error scales at
about 6 dB per doubling, so narrow the block if the artifact ever approaches it.
"""

from typing import Optional as _Optional

import torch as _torch

from ._base import ParametricNet as _ParametricNet


def quasi_static_reference(
    net: _ParametricNet,
    x: _torch.Tensor,
    trajectory: _torch.Tensor,
    ny: int,
    block: int,
) -> _torch.Tensor:
    """
    The model's own frozen-control render of the same window, as ``(B, ny)``.

    Every output sample is computed with the control held at one value across the whole
    receptive field, so the render carries no control motion at all. The control is frozen
    per block of ``block`` output samples, taken from the trajectory at each block's centre
    to halve the worst-case offset against a per-sample freeze.

    Runs under ``no_grad``: this is the target the moving render is scored against, and the
    gradient belongs to the moving side alone.
    """
    if not net.receptive_field_bounds_memory:
        # Each block is evaluated from a cold start, which for a recurrent net discards
        # the state the moving render carried in. The residual would then be dominated by
        # that state discontinuity rather than by control motion, and driving it down
        # would teach the model to ignore its own memory.
        raise ValueError(
            f"The quasi-static anchor needs a net whose receptive field bounds its "
            f"memory; {type(net).__name__} is recurrent"
        )
    if x.ndim != 2:
        raise ValueError(f"Expected a batched (B, L) input; got {tuple(x.shape)}")
    if ny < 1:
        raise ValueError(f"ny must be at least 1; got {ny}")
    if block < 1 or ny % block:
        raise ValueError(f"block must be at least 1 and divide ny={ny}; got {block}")
    history = net.receptive_field - 1
    if x.shape[1] != history + ny:
        raise ValueError(
            f"Expected an input of {history + ny} samples for ny={ny}; got {x.shape[1]}"
        )
    if trajectory.shape[:2] != x.shape:
        raise ValueError(
            f"Trajectory {tuple(trajectory.shape)} must align with the input "
            f"{tuple(x.shape)}"
        )

    batch = x.shape[0]
    blocks = ny // block
    # Each block's own receptive-field window: block k is scored from x[k*block :
    # k*block + history + block], which is exactly a stride-`block` unfold.
    windows = x.unfold(1, history + block, block)
    # The control in force at each block's centre scored sample.
    centres = history + _torch.arange(blocks, device=x.device) * block + block // 2
    frozen = trajectory[:, centres, :]

    with _torch.no_grad():
        y = net(
            windows.reshape(batch * blocks, history + block),
            frozen.reshape(batch * blocks, -1),
            pad_start=False,
        )
    return y.reshape(batch, ny)


def quasi_static_loss(
    net: _ParametricNet,
    x: _torch.Tensor,
    trajectory: _torch.Tensor,
    ny: int,
    block: int,
) -> _torch.Tensor:
    """Score the moving-control render against the model's own frozen-control one.

    Mean absolute error rather than mean squared, for the reason the silence anchors use
    it: the quantity being driven out is small, and a squared term goes quieter the closer
    the model gets, which is the opposite of what an anchor needs.
    """
    reference = quasi_static_reference(net, x, trajectory, ny, block)
    moving = net(x, trajectory, pad_start=False)
    return (moving - reference).abs().mean()
