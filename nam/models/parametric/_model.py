"""
ParametricWaveNet: a WaveNet wrapper with a residual affine adapter for parametric
conditioning. The adapter implements:

    z1' = (1 + gamma(p)) * z1 + beta(p)

where gamma and beta are zero-initialized linear maps from parameter vector p.
Zero-init guarantees z1' == z1 for all p at construction, so importing ordinary
A2 weights into self._net gives exact parity before any fine-tuning.
"""

from typing import Dict as _Dict
from typing import List as _List
from typing import Optional as _Optional
from typing import Set as _Set
from typing import Tuple as _Tuple

import numpy as _np
import torch as _torch
import torch.nn as _nn

from ..base import BaseNet as _BaseNet
from ..wavenet._wavenet import WaveNet as _InnerWaveNet
from ..wavenet._wavenet_wrapper import WaveNet as _WaveNetWrapper

# Re-export the core symbol used by the package __init__
__all__ = ["ResidualAffineAdapter", "ParametricWaveNet"]


class _ChannelAdapter(_nn.Module):
    """
    Residual affine adapter for a single channel dimension C.

    gamma_map and beta_map are zero-init so that at construction the adapter
    is an identity transformation: z1' = z1 for all p.
    """

    def __init__(self, param_dim: int, channels: int):
        super().__init__()
        self.gamma_map = _nn.Linear(param_dim, channels, bias=True)
        self.beta_map = _nn.Linear(param_dim, channels, bias=True)
        # Zero-init: both weight and bias → gamma(p) = beta(p) = 0 for all p
        _nn.init.zeros_(self.gamma_map.weight)
        _nn.init.zeros_(self.gamma_map.bias)
        _nn.init.zeros_(self.beta_map.weight)
        _nn.init.zeros_(self.beta_map.bias)

    def forward(
        self, z1: _torch.Tensor, p: _torch.Tensor
    ) -> _torch.Tensor:
        # z1: (B, C, L), p: (B, P) or (P,)
        gamma = self.gamma_map(p)  # (B, C) or (C,)
        beta = self.beta_map(p)    # (B, C) or (C,)
        # Broadcast channel dim over length axis: (..., C) → (..., C, 1)
        return (1.0 + gamma).unsqueeze(-1) * z1 + beta.unsqueeze(-1)


class ResidualAffineAdapter(_nn.Module):
    """
    Dispatches z1 to the per-channel-size _ChannelAdapter based on z1.shape[1].

    Standard NAM A2 WaveNet can have multiple LayerArrays with DIFFERENT channel
    counts (e.g. 16 and 8). The adapter is called inside every _Layer.forward, so
    it must handle whichever C it receives. We keep one sub-adapter per distinct C
    keyed by str(C) in a ModuleDict so all are registered as parameters.
    """

    def __init__(self, param_dim: int, channel_sizes: _Set[int]):
        super().__init__()
        self._param_dim = param_dim
        # ModuleDict requires string keys
        self._adapters = _nn.ModuleDict(
            {
                str(c): _ChannelAdapter(param_dim, c)
                for c in sorted(channel_sizes)
            }
        )

    def forward(self, z1: _torch.Tensor, p: _torch.Tensor) -> _torch.Tensor:
        c = z1.shape[1]
        key = str(c)
        if key not in self._adapters:
            raise KeyError(
                f"ResidualAffineAdapter has no sub-adapter for C={c}. "
                f"Registered sizes: {sorted(int(k) for k in self._adapters.keys())}"
            )
        return self._adapters[key](z1, p)


def _collect_channel_sizes(inner_net: _InnerWaveNet) -> _Set[int]:
    """
    Walk the inner WaveNet's layer arrays and collect the distinct z1 channel
    sizes across all layers.  z1 = zconv + mix_out, so z1.shape[1] equals
    the dilated conv's OUTPUT channel count (conv.out_channels), which equals
    mid_channels (== 2*bottleneck for gated activations, == channels for Tanh).
    Using conv.out_channels is the only safe way to determine C at adapter time;
    reading the 'channels' config key would be wrong for gated activations.
    """
    sizes: _Set[int] = set()
    for layer_array in inner_net._layer_arrays:
        for layer in layer_array._layers:
            # conv is a LayerConv; .out_channels is the number of output features
            sizes.add(layer.conv.out_channels)
    return sizes


class ParametricWaveNet(_BaseNet):
    """
    WaveNet wrapper that adds parametric conditioning via a ResidualAffineAdapter.

    Architecture: ordinary inner WaveNet + residual affine adapter. The adapter
    is kept on the wrapper (NOT inside the inner net) so that A2 weights import
    unchanged and adapter weights are appended in a separate block.

    param_names: ordered list of parameter names (e.g. ["gain", "treble"])
    param_dim:   length of the parameter vector P (must match len(param_names))
    """

    def __init__(
        self,
        net: _InnerWaveNet,
        param_names: _List[str],
        param_dim: int,
        sample_rate: _Optional[float] = None,
    ):
        super().__init__(sample_rate=sample_rate)
        if param_dim != len(param_names):
            raise ValueError(
                f"param_dim={param_dim} does not match len(param_names)={len(param_names)}"
            )
        self._net = net
        self._param_names = list(param_names)
        self._param_dim = param_dim
        channel_sizes = _collect_channel_sizes(net)
        self._adapter = ResidualAffineAdapter(param_dim, channel_sizes)

    @classmethod
    def parse_config(cls, config: _Dict) -> _Dict:
        config = dict(config)
        sample_rate = config.pop("sample_rate", None)
        param_names = config.pop("param_names")
        param_dim = config.pop("param_dim")
        # Remaining keys are forwarded to the inner WaveNet
        net = _InnerWaveNet.init_from_config(config)
        return {
            "net": net,
            "param_names": param_names,
            "param_dim": param_dim,
            "sample_rate": sample_rate,
        }

    # ------------------------------------------------------------------
    # BaseNet contract
    # ------------------------------------------------------------------

    @property
    def pad_start_default(self) -> bool:
        return True

    @property
    def receptive_field(self) -> int:
        return self._net.receptive_field

    def _forward(self, x: _torch.Tensor, params: _torch.Tensor) -> _torch.Tensor:
        """
        Core forward. x: (B, L), params: (B, P) or (P,).
        Returns (B, L') where L' = L - receptive_field + 1.
        """
        if x.ndim == 2:
            x = x[:, None, :]  # (B, 1, L)
        # Validate parameter dim
        p = params
        if p.ndim == 1:
            if p.shape[0] != self._param_dim:
                raise ValueError(
                    f"params has dim {p.shape[0]} but model was configured for "
                    f"param_dim={self._param_dim}"
                )
            # Broadcast (P,) across the batch by keeping it 1-D; the linear maps
            # handle this via standard PyTorch broadcasting.
        elif p.ndim == 2:
            if p.shape[1] != self._param_dim:
                raise ValueError(
                    f"params has shape {tuple(p.shape)} but model was configured for "
                    f"param_dim={self._param_dim}"
                )
        else:
            raise ValueError(
                f"params must be 1-D (P,) or 2-D (B, P), got shape {tuple(p.shape)}"
            )
        y = self._net(x, adapter=self._adapter, p=p)
        assert y.shape[1] == 1
        return y[:, 0, :]

    def forward(
        self,
        x: _torch.Tensor,
        params: _torch.Tensor,
        pad_start: _Optional[bool] = None,
    ) -> _torch.Tensor:
        """
        Public forward. Overrides BaseNet.forward to add required `params` arg.

        x:      (B, L) or (L,) audio input
        params: (B, P) or (P,) parameter vector
        """
        pad_start = self.pad_start_default if pad_start is None else pad_start
        scalar = x.ndim == 1
        if scalar:
            x = x[None]
        if pad_start:
            x = _torch.cat(
                (
                    _torch.zeros((len(x), self.receptive_field - 1)).to(x.device),
                    x,
                ),
                dim=1,
            )
        if x.shape[1] < self.receptive_field:
            raise ValueError(
                f"Input has {x.shape[1]} samples, which is too few for this model with "
                f"receptive field {self.receptive_field}!"
            )
        y = self._forward(x, params)
        if scalar:
            y = y[0]
        return y

    # ------------------------------------------------------------------
    # Export contract — minimal for C1.1b; fuller metadata is C1.2
    # ------------------------------------------------------------------

    def _export_config(self) -> dict:
        cfg = self._net.export_config(sample_rate=self.sample_rate)
        cfg["param_names"] = self._param_names
        cfg["param_dim"] = self._param_dim
        return cfg

    def _export_weights(self) -> _np.ndarray:
        """
        Inner WaveNet weights first (unchanged blob), then adapter weights appended.
        Inner blob is preserved exactly so existing A2 consumers can slice it.
        """
        inner_weights = self._net.export_weights()  # numpy 1-D
        adapter_weights = _torch.cat(
            [
                p.detach().cpu().flatten()
                for p in self._adapter.parameters()
            ]
        ).numpy()
        return _np.concatenate([inner_weights, adapter_weights])

    def _export_input_output_args(self):
        """
        Provide a zero parameter vector so _Base._export_input_output can call
        self(*args, x, pad_start=True) without binding params→x.

        Note: _Base._export_input_output does:
            self(*args, x, pad_start=True)
        which expands to self(zeros_p, x, pad_start=True) = forward(zeros_p, x, ...),
        but our signature is forward(x, params, ...) so positional order would
        bind the first extra arg as x and the audio x as params — WRONG.

        We fully override _export_input_output instead to be safe.
        """
        pass  # overridden below; this method is not used

    def _export_input_output(self):
        import math as _math

        rate = self.sample_rate
        if rate is None:
            raise RuntimeError(
                "Cannot export model's input and output without a sample rate."
            )
        x = _torch.cat(
            [
                _torch.zeros((rate,)),
                0.5
                * _torch.sin(
                    2.0
                    * _math.pi
                    * 220.0
                    * _torch.linspace(0.0, 1.0, rate + 1)[:-1]
                ),
                _torch.zeros((rate,)),
            ]
        )
        # Use zero params for the snapshot (nominal embedding = no adaptation)
        p = _torch.zeros(self._param_dim)
        return (
            x.detach().cpu().numpy(),
            self(x, p, pad_start=True).detach().cpu().numpy(),
        )

    def _at_nominal_settings(self, x: _torch.Tensor) -> _torch.Tensor:
        """Run at zero params (no parametric offset from the A2 embedding)."""
        p = _torch.zeros(self._param_dim, device=x.device)
        return self(x, p)
