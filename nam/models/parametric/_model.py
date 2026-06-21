"""
ParametricWaveNet: a WaveNet wrapper with a residual affine adapter for parametric
conditioning. The adapter implements:

    z1' = (1 + gamma(p)) * z1 + beta(p)

where gamma and beta are zero-initialized per-layer heads on top of a shared
nonlinear parameter encoder. Zero-init guarantees z1' == z1 for all p at
construction, so importing ordinary A2 weights into self._net gives exact parity
before any fine-tuning.
"""

from copy import deepcopy as _deepcopy
from typing import Any as _Any
from typing import Dict as _Dict
from typing import List as _List
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import Union as _Union

import numpy as _np
import torch as _torch
import torch.nn as _nn

from .._activations import get_activation as _get_activation
from ..base import BaseNet as _BaseNet
from ..wavenet._wavenet import WaveNet as _InnerWaveNet
from ..wavenet._wavenet_wrapper import WaveNet as _WaveNetWrapper
from ._spec import ParamSpec as _ParamSpec

# Re-export the core symbol used by the package __init__
__all__ = ["ResidualAffineAdapter", "ParametricWaveNet"]


_DEFAULT_ADAPTER_HIDDEN_DIM = 8
_DEFAULT_ADAPTER_ACTIVATION = "SiLU"
_AdapterActivation = _Union[str, _Dict[str, _Any]]


class _SharedParamEncoder(_nn.Module):
    """
    Shared nonlinear encoder from normalized params to a hidden modulation state.
    """

    def __init__(
        self,
        param_dim: int,
        hidden_dim: int,
        activation: _AdapterActivation,
    ):
        super().__init__()
        self.fc = _nn.Linear(param_dim, hidden_dim, bias=True)
        self.activation = _get_activation(activation)

    def forward(self, params: _torch.Tensor) -> _torch.Tensor:
        return self.activation(self.fc(params))


class _LayerAdapter(_nn.Module):
    """
    Residual affine adapter heads for one specific inner WaveNet layer.

    Each layer keeps its own zero-init gamma and beta heads while sharing the
    upstream nonlinear parameter encoder across the whole WaveNet.
    """

    def __init__(
        self,
        hidden_dim: int,
        channels: int,
    ):
        super().__init__()
        self.gamma_head = _nn.Linear(hidden_dim, channels, bias=True)
        self.beta_head = _nn.Linear(hidden_dim, channels, bias=True)
        # Zero-init: both weight and bias → gamma(h) = beta(h) = 0 for all h
        _nn.init.zeros_(self.gamma_head.weight)
        _nn.init.zeros_(self.gamma_head.bias)
        _nn.init.zeros_(self.beta_head.weight)
        _nn.init.zeros_(self.beta_head.bias)

    def forward(
        self, z1: _torch.Tensor, hidden: _torch.Tensor
    ) -> _torch.Tensor:
        # z1: (B, C, L), hidden: (B, H) or (H,)
        gamma = self.gamma_head(hidden)  # (B, C) or (C,)
        beta = self.beta_head(hidden)  # (B, C) or (C,)
        # Broadcast channel dim over length axis: (..., C) → (..., C, 1)
        return (1.0 + gamma).unsqueeze(-1) * z1 + beta.unsqueeze(-1)


class ResidualAffineAdapter(_nn.Module):
    """
    Dispatches z1 to a dedicated per-layer adapter based on a stable layer index.

    Standard NAM A2 WaveNet invokes the adapter once at each inner layer's
    pre-activation hook. This module keeps one shared nonlinear parameter encoder
    and one registered affine-head adapter per inner layer, preserving the
    existing hook traversal and deterministic export/import ordering.
    """

    uses_layer_index = True

    def __init__(
        self,
        param_dim: int,
        layer_channels: _List[int],
        hidden_dim: int = _DEFAULT_ADAPTER_HIDDEN_DIM,
        activation: _AdapterActivation = _DEFAULT_ADAPTER_ACTIVATION,
    ):
        super().__init__()
        if hidden_dim < 1:
            raise ValueError(
                f"adapter_hidden_dim must be positive, got {hidden_dim}."
            )
        self._param_dim = param_dim
        self._hidden_dim = hidden_dim
        self._layer_channels = list(layer_channels)
        self._shared_encoder = _SharedParamEncoder(
            param_dim,
            hidden_dim,
            activation,
        )
        self._adapters = _nn.ModuleList(
            [
                _LayerAdapter(hidden_dim, channels)
                for channels in self._layer_channels
            ]
        )

    def forward(
        self,
        z1: _torch.Tensor,
        params: _torch.Tensor,
        *,
        layer_index: int,
        hidden: _Optional[_torch.Tensor] = None,
    ) -> _torch.Tensor:
        if not (0 <= layer_index < len(self._adapters)):
            raise IndexError(
                f"ResidualAffineAdapter got layer_index={layer_index}, but has "
                f"{len(self._adapters)} per-layer adapters."
            )
        adapter = self._adapters[layer_index]
        if z1.shape[1] != self._layer_channels[layer_index]:
            raise ValueError(
                f"Per-layer adapter {layer_index} expects C="
                f"{self._layer_channels[layer_index]}, got C={z1.shape[1]}."
            )
        hidden = self._shared_encoder(params) if hidden is None else hidden
        return adapter(z1, hidden)


def _collect_layer_channels(inner_net: _InnerWaveNet) -> _List[int]:
    """
    Walk the inner WaveNet's layer arrays and collect z1 channel sizes in the
    exact order layers are visited during forward. z1 = zconv + mix_out, so
    z1.shape[1] equals
    the dilated conv's OUTPUT channel count (conv.out_channels), which equals
    mid_channels (== 2*bottleneck for gated activations, == channels for Tanh).
    Using conv.out_channels is the only safe way to determine C at adapter time.
    """
    channels: _List[int] = []
    for layer_array in inner_net._layer_arrays:
        for layer in layer_array._layers:  # type: ignore[union-attr]
            channels.append(layer.conv.out_channels)
    return channels


def _assign_adapter_indices(inner_net: _InnerWaveNet) -> int:
    """
    Tag each inner WaveNet layer with a stable traversal-order adapter index.
    """
    layer_index = 0
    for layer_array in inner_net._layer_arrays:
        for layer in layer_array._layers:  # type: ignore[union-attr]
            layer._parametric_adapter_index = layer_index
            layer_index += 1
    return layer_index


class ParametricWaveNet(_BaseNet):
    """
    WaveNet wrapper that adds parametric conditioning via a ResidualAffineAdapter.

    Architecture: ordinary inner WaveNet + residual affine adapter. The adapter
    is kept on the wrapper (NOT inside the inner net) so that A2 weights import
    unchanged and adapter weights are appended in a separate block.

    param_specs: ordered list of ParamSpec objects (name/min/max/default per param).
                 List order is significant — it determines positional index in the
                 params tensor fed to the net.
    """

    def __init__(
        self,
        net: _InnerWaveNet,
        param_specs: _List[_ParamSpec],
        sample_rate: _Optional[float] = None,
        adapter_hidden_dim: int = _DEFAULT_ADAPTER_HIDDEN_DIM,
        adapter_activation: _AdapterActivation = _DEFAULT_ADAPTER_ACTIVATION,
    ):
        super().__init__(sample_rate=sample_rate)
        if len(param_specs) == 0:
            raise ValueError("param_specs must contain at least one ParamSpec.")
        self._net = net
        self._param_specs = list(param_specs)
        self._adapter_hidden_dim = int(adapter_hidden_dim)
        self._adapter_activation = (
            _deepcopy(adapter_activation)
            if isinstance(adapter_activation, dict)
            else adapter_activation
        )
        param_mins = _torch.tensor([s.min for s in param_specs], dtype=_torch.float32)
        param_maxs = _torch.tensor([s.max for s in param_specs], dtype=_torch.float32)
        param_centers = _torch.tensor(
            [s.center for s in param_specs], dtype=_torch.float32
        )
        param_half_ranges = _torch.tensor(
            [s.half_range for s in param_specs], dtype=_torch.float32
        )
        self.register_buffer("_param_mins", param_mins)
        self.register_buffer("_param_maxs", param_maxs)
        self.register_buffer("_param_centers", param_centers)
        self.register_buffer("_param_half_ranges", param_half_ranges)
        # Derive internal state from specs so the forward pass is unchanged: the net
        # still receives default values positionally, exactly as nominal_params did.
        self._nominal_params = _torch.tensor(
            [s.default for s in param_specs], dtype=_torch.float32
        )
        layer_channels = _collect_layer_channels(net)
        _assign_adapter_indices(net)
        self._adapter = ResidualAffineAdapter(
            len(param_specs),
            layer_channels,
            hidden_dim=self._adapter_hidden_dim,
            activation=self._adapter_activation,
        )

    # ------------------------------------------------------------------
    # Derived read-only properties (backward-compat for callers that read
    # the old flat attributes; no storage duplication)
    # ------------------------------------------------------------------

    @property
    def _param_names(self) -> _List[str]:
        return [s.name for s in self._param_specs]

    @property
    def _param_dim(self) -> int:
        return len(self._param_specs)

    @classmethod
    def parse_config(cls, config: _Dict) -> _Dict:
        config = dict(config)
        sample_rate = config.pop("sample_rate", None)
        # Parse the self-describing params array (new schema).
        # Each entry is {"name": ..., "min": ..., "max": ..., "default": ...}.
        if "params" not in config:
            raise ValueError(
                "'params' is required in ParametricWaveNet config but was not found. "
                "Provide a list of {name, min, max, default} objects, one per parameter."
            )
        raw_specs = config.pop("params")
        param_specs = [_ParamSpec.from_dict(d) for d in raw_specs]
        adapter_hidden_dim = int(
            config.pop("adapter_hidden_dim", _DEFAULT_ADAPTER_HIDDEN_DIM)
        )
        adapter_activation = _deepcopy(
            config.pop("adapter_activation", _DEFAULT_ADAPTER_ACTIVATION)
        )
        # Remaining keys are forwarded to the inner WaveNet
        net = _InnerWaveNet.init_from_config(config)
        return {
            "net": net,
            "param_specs": param_specs,
            "sample_rate": sample_rate,
            "adapter_hidden_dim": adapter_hidden_dim,
            "adapter_activation": adapter_activation,
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
        p = self._normalize_params(p, dtype=x.dtype)
        adapter_hidden = self._adapter._shared_encoder(p)
        y = self._net(x, adapter=self._adapter, p=p, adapter_hidden=adapter_hidden)
        assert y.shape[1] == 1
        return y[:, 0, :]

    def _normalize_params(
        self,
        params: _torch.Tensor,
        dtype: _Optional[_torch.dtype] = None,
    ) -> _torch.Tensor:
        """
        Map raw user-space knob values to the adapter's signed unit range.

        Raw values are first clamped to the declared [min, max] bounds, then
        affine-mapped into [-1, 1] so midpoints land at 0.0.
        """
        target_dtype = dtype
        if target_dtype is None:
            target_dtype = (
                params.dtype if params.is_floating_point() else self._param_mins.dtype
            )
        params = params.to(dtype=target_dtype)
        mins = self._param_mins.to(device=params.device, dtype=target_dtype)
        maxs = self._param_maxs.to(device=params.device, dtype=target_dtype)
        centers = self._param_centers.to(device=params.device, dtype=target_dtype)
        half_ranges = self._param_half_ranges.to(
            device=params.device, dtype=target_dtype
        )
        clipped = _torch.maximum(_torch.minimum(params, maxs), mins)
        return (clipped - centers) / half_ranges

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
        # Self-describing params array: order is significant (positional net input).
        # Raw host/UI values are normalized to [-1, 1] from these bounds before
        # reaching the adapter, so downstream runtimes should apply the same rule.
        cfg["params"] = [s.to_dict() for s in self._param_specs]
        cfg["adapter_hidden_dim"] = self._adapter_hidden_dim
        cfg["adapter_activation"] = _deepcopy(self._adapter_activation)
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
        p = self._nominal_params
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
