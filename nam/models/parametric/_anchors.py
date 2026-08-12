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
