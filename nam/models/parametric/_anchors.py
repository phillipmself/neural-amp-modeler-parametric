"""
Silence-in / silence-out training anchors.

A captured device produces no output when its input is silent, whatever its controls are
doing. That holds unconditionally, so it can be trained on directly without any capture
data: feed the net digital silence, hold the controls somewhere, and score the output
against zero.

The anchor exists because the concat architectures carry the encoded controls as input
channels, so those channels pass through the whole receptive field. Training only ever
presented a control setting that was constant across a window, which constrains the sum of
the model's per-lag control sensitivity but not how that sum is distributed across lags.
Anchoring silence pins the distribution as well.
"""

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
    Run ``net`` on digital silence at the given ``(B, P)`` control settings and return the
    ``(B, ny)`` scored tail of the output the device would answer with silence.

    The input is built here and fed with ``pad_start=False``, so the model's history is
    part of the window rather than something padded on afterwards. Two kinds of history
    have to be budgeted for, and only one of them is the receptive field: a recurrent net
    also runs a detached burn-in whose outputs carry no gradient, so ``ny`` samples of
    window would leave nothing scored at all. ``training_warmup`` covers that, and the
    warmup samples are dropped from the return so the score is taken only where the
    gradient actually flows.
    """
    if ny < 1:
        raise ValueError(f"ny must be at least 1; got {ny}")
    if params.ndim != 2:
        raise ValueError(
            f"Expected params to have shape (B, P); got {tuple(params.shape)}"
        )
    batch_size = params.shape[0]
    length = net.receptive_field - 1 + net.training_warmup + ny
    parameter = next(net.parameters())
    x = _torch.zeros(
        (batch_size, length), device=parameter.device, dtype=parameter.dtype
    )
    if net.requires_uniform_batch_params:
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
