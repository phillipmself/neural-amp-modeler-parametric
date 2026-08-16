"""
Parametric WaveNet conditioned by FiLM on the control vector alone.

The point of this architecture is *when* the controls act, not just how. Under
:class:`ConcatWaveNet` the encoded controls are input channels, so they enter the residual
stream and are then convolved by every dilated kernel downstream. During training those
channels are constant in time, so a kernel spanning a control change sees a tap pattern it
was never trained on -- which is the ~receptive-field-long disturbance a knob move produces.

Here the controls reach the network only through FiLM's 1x1 convolutions, which have no
time extent. The control value at a given sample affects that sample's modulation and
nothing else, so no weight is ever exposed to a non-constant control pattern and there is
no invalid-configuration window at any rate of change. Audio state still washes out over
the receptive field after a move, which is both unavoidable and the physically right
behaviour; what goes away is the model computing a function it was never trained for.

Two consequences fall out of the control condition being constant in time:

* The inner WaveNet is shape-identical to a stock, non-parametric one (``input_size`` and
  ``condition_size`` are 1). It does not need the widened channels ConcatWaveNet uses to
  carry control information, and it can be initialized from an ordinary single-setting
  capture -- with ``identity_init`` the model *is* that capture at construction.
* A runtime can compute each FiLM's scale/shift once per control change instead of once per
  frame, so the modulation is nearly free. Training passes the control as a length-1 time
  axis and lets it broadcast, which is the same arithmetic.
"""

from collections.abc import Sequence as _Sequence
from typing import Any as _Any
from typing import Optional as _Optional
from typing import cast as _cast

import numpy as _np
import torch as _torch
import torch.nn as _nn

from .._activations import get_activation as _get_activation
from .._from_nam import convert_nam_wavenet_config as _convert_nam_wavenet_config
from ..wavenet._film import FiLM as _FiLM
from ..wavenet._wavenet import WaveNet as _WaveNet
from ._base import ParametricNet as _ParametricNet
from ._spec import ParamSpec as _ParamSpec

_WeightsLike = _Sequence[float] | _np.ndarray | _torch.Tensor


# FiLM exposes its condition-to-scale/shift conv only privately. Reaching for it here keeps
# the shape queries and the identity init that this architecture needs out of the stock
# WaveNet code, which nothing else needs them in.
def _film_condition_size(film: _FiLM) -> int:
    return film._film.in_channels


def _film_input_dim(film: _FiLM) -> int:
    out_channels = film._film.out_channels
    return out_channels // 2 if film.shift else out_channels


def _init_film_identity(film: _FiLM) -> None:
    """Zero weights and unit-scale/zero-shift bias, so the FiLM passes its input through.

    Zero weights also mean the condition has no effect on the output *yet*, so a param
    encoder upstream sees no gradient on the very first step. It clears itself once these
    weights move, which happens on that same step.
    """
    conv = film._film
    conv.weight.data.zero_()
    if conv.bias is not None:
        conv.bias.data.zero_()
        conv.bias.data[: _film_input_dim(film)] = 1.0


class ParamEncoder(_nn.Module):
    """
    Optional MLP between the encoded control vector and the FiLM condition.

    FiLM's 1x1 is linear in its condition, so without this the map from a knob to a
    channel gain is linear -- which does not match measured knob tapers (Gain is far more
    sensitive at the low end than the high). One hidden layer is enough to give the taper.

    Weights are exported in ``nn.Linear`` order, layer by layer: the full weight matrix
    (out_features x in_features, row-major) then the bias.
    """

    def __init__(
        self,
        in_features: int,
        hidden_sizes: _Sequence[int],
        out_features: int,
        activation: str = "ReLU",
    ):
        super().__init__()
        self._activation_name = activation
        sizes = [in_features, *hidden_sizes, out_features]
        layers: list[_nn.Module] = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            if i > 0:
                layers.append(_get_activation(activation))
            layers.append(_nn.Linear(a, b))
        self._layers = _nn.Sequential(*layers)
        self._in_features = in_features
        self._out_features = out_features
        self._hidden_sizes = tuple(hidden_sizes)

    @property
    def in_features(self) -> int:
        return self._in_features

    @property
    def out_features(self) -> int:
        return self._out_features

    def export_config(self) -> dict[str, _Any]:
        return {
            "hidden_sizes": list(self._hidden_sizes),
            "out_features": self._out_features,
            "activation": self._activation_name,
        }

    def export_weights(self) -> _torch.Tensor:
        tensors = []
        for module in self._layers:
            if isinstance(module, _nn.Linear):
                tensors.append(module.weight.data.flatten())
                tensors.append(module.bias.data.flatten())
        return _torch.cat(tensors)

    def import_weights(self, weights: _torch.Tensor, i: int) -> int:
        for module in self._layers:
            if not isinstance(module, _nn.Linear):
                continue
            n = module.weight.numel()
            module.weight.data = (
                weights[i : i + n]
                .reshape(module.weight.shape)
                .to(module.weight.device)
            )
            i += n
            n = module.bias.numel()
            module.bias.data = (
                weights[i : i + n].reshape(module.bias.shape).to(module.bias.device)
            )
            i += n
        return i

    def forward(self, p: _torch.Tensor) -> _torch.Tensor:
        """
        :param p: (B, in_features) encoded controls
        :return: (B, out_features)
        """
        return self._layers(p)


class FiLMWaveNet(_ParametricNet):
    def __init__(
        self,
        *,
        wavenet: _WaveNet,
        param_specs: _Sequence[_ParamSpec],
        param_encoder: _Optional[ParamEncoder] = None,
        identity_init: bool = True,
        sample_rate: _Optional[float] = None,
    ):
        super().__init__(param_specs=param_specs, sample_rate=sample_rate)
        self._validate_supported_wavenet(wavenet)
        self._wavenet = wavenet
        self._param_encoder = param_encoder
        self._films = tuple(
            module for module in wavenet.modules() if isinstance(module, _FiLM)
        )
        if len(self._films) == 0:
            raise ValueError(
                "FiLMWaveNet requires at least one active FiLM module; the control vector "
                "has no other way into the network. Activate one of the layer array's "
                "*_film options."
            )
        self._film_condition_size = (
            self.encoded_param_dim
            if param_encoder is None
            else param_encoder.out_features
        )
        for film in self._films:
            condition_size = _film_condition_size(film)
            if condition_size != self._film_condition_size:
                raise ValueError(
                    f"FiLM condition size {condition_size} does not match the control "
                    f"condition width {self._film_condition_size}"
                )
        if identity_init:
            for film in self._films:
                _init_film_identity(film)

    @classmethod
    def parse_config(cls, config: dict[str, _Any]) -> dict[str, _Any]:
        config = super().parse_config(config)
        sample_rate = config.pop("sample_rate", None)
        identity_init = config.pop("identity_init", True)
        raw_params = config.pop("params", None)
        if raw_params is None:
            raise ValueError("FiLMWaveNet config must define a params array")
        param_specs = tuple(_ParamSpec.from_dict(spec) for spec in raw_params)
        if len(param_specs) == 0:
            raise ValueError("FiLMWaveNet config must define at least one ParamSpec")
        if "condition_dsp" in config:
            # Same rationale as ConcatWaveNet: rebuilding the inner net from config would
            # silently drop the nested condition_dsp weights.
            raise NotImplementedError("FiLMWaveNet does not support condition_dsp")

        layers_configs = config.get("layers")
        if not layers_configs:
            raise ValueError("FiLMWaveNet config must define at least one layer array")

        encoded_param_dim = sum(spec.num_inputs for spec in param_specs)
        encoder_config = config.pop("param_encoder", None)
        param_encoder = (
            None
            if encoder_config is None
            else ParamEncoder(
                in_features=encoded_param_dim,
                hidden_sizes=encoder_config.get("hidden_sizes", []),
                out_features=encoder_config.get("out_features", encoded_param_dim),
                activation=encoder_config.get("activation", "ReLU"),
            )
        )
        film_condition_size = (
            encoded_param_dim if param_encoder is None else param_encoder.out_features
        )

        for i, layer_config in enumerate(layers_configs):
            if layer_config.get("packing"):
                raise NotImplementedError(
                    "FiLMWaveNet does not support packed layer arrays"
                )
            if layer_config.get("slimmable"):
                raise NotImplementedError(
                    "FiLMWaveNet does not support slimmable layer arrays"
                )
            if i == 0:
                cls._require_derived(layer_config, "input_size", 1, i)
            cls._require_derived(layer_config, "condition_size", 1, i)
            cls._require_derived(
                layer_config, "film_condition_size", film_condition_size, i
            )

        wavenet_config = _convert_nam_wavenet_config(config, sample_rate=sample_rate)
        wavenet = _WaveNet.init_from_config(wavenet_config)
        return {
            "wavenet": wavenet,
            "param_specs": param_specs,
            "param_encoder": param_encoder,
            "identity_init": identity_init,
            "sample_rate": sample_rate,
        }

    @staticmethod
    def _require_derived(
        layer_config: dict[str, _Any], key: str, value: int, index: int
    ) -> None:
        """Set a field FiLMWaveNet derives, or reject a config that declares it wrong."""
        declared = layer_config.get(key)
        if declared is None:
            layer_config[key] = value
        elif declared != value:
            raise ValueError(
                f"FiLMWaveNet derives layer array {index}'s {key}: expected {value}, got "
                f"{declared}. Omit the field or set it to the derived value."
            )

    @staticmethod
    def _validate_supported_wavenet(wavenet: _WaveNet) -> None:
        # Slimmable is not checked here: WaveNet already refuses to build a slimmable
        # layer array with FiLM active, and a FiLMWaveNet without FiLM is rejected below.
        if wavenet._condition_dsp is not None:
            raise NotImplementedError(
                "FiLMWaveNet does not support inner WaveNets with condition_dsp"
            )

    @property
    def pad_start_default(self) -> bool:
        return True

    @property
    def receptive_field(self) -> int:
        return self._wavenet.receptive_field

    @property
    def param_encoder(self) -> _Optional[ParamEncoder]:
        return self._param_encoder

    @property
    def film_condition_size(self) -> int:
        return self._film_condition_size

    @property
    def supports_compiled_step(self) -> bool:
        # Straight-line tensor work: encode the controls, then one inner forward.
        return True

    @property
    def supports_param_trajectory(self) -> bool:
        # The controls are consumed by 1x1 convolutions, which have no time extent, so a
        # control that moves within the window is just a condition that is not constant --
        # no weight sees a tap pattern it was not trained on at any rate of change. This is
        # also what the silence anchors need in order to score a knob being turned.
        return True

    @property
    def receptive_field_bounds_memory(self) -> bool:
        # Dilated convolutions and 1x1 conditioning, nothing recurrent, so the receptive
        # field is exact.
        return True

    def _encode(self, p: _torch.Tensor) -> _torch.Tensor:
        """Encoded controls (B, encoded_param_dim) -> (B, film_condition_size)."""
        return p if self._param_encoder is None else self._param_encoder(p)

    def _film_condition(self, p: _torch.Tensor) -> _torch.Tensor:
        """Encoded controls -> (B, film_condition_size, 1), constant over the buffer."""
        return self._encode(p)[:, :, None]

    def _compilable_step(self, x: _torch.Tensor, p: _torch.Tensor) -> _torch.Tensor:
        # Both condition shapes go through here so both get compiled. Dynamo specializes
        # on the rank, so this is two graphs built once each, not a branch re-traced per
        # call -- and an uncompiled forward is expensive on a launch-bound model.
        condition = (
            self._film_condition_sequence(p) if p.ndim == 3 else self._film_condition(p)
        )
        return self._wavenet(x[:, None, :], condition)

    def _film_condition_sequence(self, p: _torch.Tensor) -> _torch.Tensor:
        """Per-sample encoded controls (B, T, D) -> (B, film_condition_size, T)."""
        batch, length, dim = p.shape
        encoded = self._encode(p.reshape(batch * length, dim))
        return encoded.reshape(batch, length, -1).permute(0, 2, 1)

    def _run_conditioned(self, x: _torch.Tensor, p: _torch.Tensor) -> _torch.Tensor:
        if p.ndim == 3:
            # A moving control. The condition is the only thing that changes; the inner
            # forward is the same one the held-setting path runs, with a condition that has
            # a real time axis instead of a broadcast one.
            if p.shape[0] != x.shape[0]:
                raise ValueError(
                    f"Input batch size {x.shape[0]} must match encoded params batch size "
                    f"{p.shape[0]}"
                )
            if p.shape[1] != x.shape[1]:
                raise ValueError(
                    f"Control trajectory length {p.shape[1]} must match the input's "
                    f"{x.shape[1]} samples"
                )
            y = self._run_step(x, p)
            if y.shape[1] != 1:
                raise RuntimeError(
                    f"Expected inner WaveNet to return one channel; got {tuple(y.shape)}"
                )
            return y[:, 0, :]
        if p.ndim == 1:
            p = p[None].expand(x.shape[0], -1)
        elif p.shape[0] != x.shape[0]:
            raise ValueError(
                f"Input batch size {x.shape[0]} must match encoded params batch size "
                f"{p.shape[0]}"
            )
        y = self._run_step(x, p)
        if y.shape[1] != 1:
            raise RuntimeError(
                f"Expected inner WaveNet to return one channel; got shape {tuple(y.shape)}"
            )
        return y[:, 0, :]

    def forward_control_sequence(
        self,
        x: _torch.Tensor,
        params: _torch.Tensor,
        pad_start: _Optional[bool] = None,
    ) -> _torch.Tensor:
        """
        Run with controls that move during the buffer -- what a knob turn looks like.

        A channels-first convenience wrapper over the ``(B, T, P)`` trajectory form of
        :meth:`forward`, so this and the silence anchors drive exactly the same path.

        :param x: (L,) or (B, L) audio
        :param params: (P, L) or (B, P, L) raw controls, sample-aligned with ``x``
        :return: (L',) or (B, L') as in :meth:`forward`
        """
        scalar_input = x.ndim == 1
        if scalar_input:
            x = x[None]
        if params.ndim == 2:
            params = params[None]
        if params.ndim != 3:
            raise ValueError(
                f"Expected params to have shape (P, L) or (B, P, L); got {tuple(params.shape)}"
            )
        # (B, P, L) -> the (B, T, P) trajectory `forward` takes; it owns the length and
        # batch checks, the pad_start history, and holding the opening setting across it.
        y = self(x, params.permute(0, 2, 1), pad_start=pad_start)
        return y[0] if scalar_input else y

    def _export_inner_config(self) -> dict[str, _Any]:
        config = self._wavenet.export_config(sample_rate=self.sample_rate)
        # Self-describing for a reader that would otherwise have to derive it: the stock
        # layer-array export has no notion of a FiLM condition other than its own.
        for layer_config in config["layers"]:
            layer_config["film_condition_size"] = self._film_condition_size
        config["param_encoder"] = (
            None if self._param_encoder is None else self._param_encoder.export_config()
        )
        return config

    def _export_weights(self) -> _np.ndarray:
        # Encoder first, so a reader can size it from the config before walking the
        # WaveNet's variable-length blob.
        wavenet_weights = _torch.tensor(self._wavenet.export_weights())
        if self._param_encoder is None:
            return wavenet_weights.numpy()
        encoder_weights = self._param_encoder.export_weights().detach().cpu()
        return _torch.cat([encoder_weights, wavenet_weights]).numpy()

    def import_weights(self, weights: _WeightsLike, i: int = 0) -> int:
        weights_tensor = _cast(
            _torch.Tensor,
            weights if isinstance(weights, _torch.Tensor) else _torch.tensor(weights),
        )
        if weights_tensor.ndim != 1:
            raise ValueError(
                "FiLMWaveNet weights must be a flat 1-D sequence; got shape "
                f"{tuple(weights_tensor.shape)}"
            )
        if self._param_encoder is not None:
            i = self._param_encoder.import_weights(weights_tensor, i)
        return self._wavenet.import_weights(weights_tensor, i)
