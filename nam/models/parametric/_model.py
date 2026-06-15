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
        for layer in layer_array._layers:  # type: ignore[union-attr]
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
        nominal_params: _List[float],
        sample_rate: _Optional[float] = None,
    ):
        super().__init__(sample_rate=sample_rate)
        if param_dim != len(param_names):
            raise ValueError(
                f"param_dim={param_dim} does not match len(param_names)={len(param_names)}"
            )
        # nominal_params is a required config field (AD-5): it defines the parameter
        # setting used for export snapshots and loudness normalization. Requiring it at
        # construction prevents silent zeros that may not reflect the intended baseline.
        if len(nominal_params) != param_dim:
            raise ValueError(
                f"nominal_params has length {len(nominal_params)} but param_dim={param_dim}. "
                f"nominal_params must have exactly one value per parameter name in param_names."
            )
        self._net = net
        self._param_names = list(param_names)
        self._param_dim = param_dim
        self._nominal_params = _torch.tensor(
            [float(v) for v in nominal_params], dtype=_torch.float32
        )
        channel_sizes = _collect_channel_sizes(net)
        self._adapter = ResidualAffineAdapter(param_dim, channel_sizes)

    @classmethod
    def parse_config(cls, config: _Dict) -> _Dict:
        config = dict(config)
        sample_rate = config.pop("sample_rate", None)
        param_names = config.pop("param_names")
        param_dim = config.pop("param_dim")
        # nominal_params is required (AD-5): fail early with a clear message so the
        # user knows exactly which field is missing rather than getting a cryptic KeyError.
        if "nominal_params" not in config:
            raise ValueError(
                "nominal_params is required in ParametricWaveNet config but was not found. "
                "Provide a list of floats with one value per parameter name."
            )
        nominal_params = config.pop("nominal_params")
        # Remaining keys are forwarded to the inner WaveNet
        net = _InnerWaveNet.init_from_config(config)
        return {
            "net": net,
            "param_names": param_names,
            "param_dim": param_dim,
            "nominal_params": nominal_params,
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

    def _forward(self, x: _torch.Tensor, **kwargs) -> _torch.Tensor:
        """
        Core forward. x: (B, L), kwargs must contain 'params': (B, P) or (P,).
        Returns (B, L') where L' = L - receptive_field + 1.
        """
        params: _torch.Tensor = kwargs["params"]
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

    def forward(  # type: ignore[override]
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
        y = self._forward(x, params=params)
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
        # nominal_params serialized as a JSON-friendly list of Python floats so
        # downstream loaders can reconstruct the tensor without numpy dependency.
        cfg["nominal_params"] = self._nominal_params.tolist()
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

    def _export_input_output_args(self):  # type: ignore[override]
        """
        Not used: _export_input_output is fully overridden below.
        Returns empty tuple to satisfy base return type.
        """
        return ()  # type: ignore[return-value]

    def _export_input_output(self):
        import math as _math

        rate = self.sample_rate
        if rate is None:
            raise RuntimeError(
                "Cannot export model's input and output without a sample rate."
            )
        n = int(rate)
        x = _torch.cat(
            [
                _torch.zeros((n,)),
                0.5
                * _torch.sin(
                    2.0
                    * _math.pi
                    * 220.0
                    * _torch.linspace(0.0, 1.0, n + 1)[:-1]
                ),
                _torch.zeros((n,)),
            ]
        )
        # Use zero params for the snapshot (nominal embedding = no adaptation)
        p = _torch.zeros(self._param_dim)
        return (
            x.detach().cpu().numpy(),
            self(x, p, pad_start=True).detach().cpu().numpy(),
        )

    def import_weights(  # type: ignore[override]
        self,
        weights: _torch.Tensor,
        i: int = 0,
    ) -> int:
        """
        Load the flat weight vector produced by _export_weights().

        The blob is laid out as: [inner WaveNet weights] ++ [adapter weights].
        Delegating to self._net.import_weights returns the index where the inner
        net stopped, then we overwrite the adapter parameters in the same order
        they were flattened during export (adapter.parameters() iteration order).

        The type: ignore[override] suppresses the Sequence[float] vs Tensor mismatch
        with Exportable.import_weights — callers pass a Tensor (matching _from_nam.py
        usage) and internal wavenet methods also accept Tensor. Keeping Tensor here
        avoids an unnecessary re-conversion on every call.
        """
        weights_t = (
            weights
            if isinstance(weights, _torch.Tensor)
            else _torch.tensor(weights, dtype=_torch.float32)
        )
        i = self._net.import_weights(weights_t, i)
        # Load adapter weights in the same order as _export_weights: iterate
        # self._adapter.parameters() and fill each parameter from the flat blob.
        for param in self._adapter.parameters():
            n = param.numel()
            param.data.copy_(weights_t[i : i + n].reshape(param.shape))
            i += n
        return i

    def _at_nominal_settings(self, x: _torch.Tensor) -> _torch.Tensor:
        """Run at the configured nominal params rather than zeros.

        The base BaseNet._at_nominal_settings calls self(x) with no params arg,
        which would fail for ParametricWaveNet's required `params` argument. This
        override injects self._nominal_params so export-time loudness normalization
        and snapshot generation use the user-specified baseline parameter setting.
        """
        p = self._nominal_params.to(x.device)
        return self(x, p)
