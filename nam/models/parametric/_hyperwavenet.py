"""
Parametric WaveNet wrapper built from a frozen template plus generated weight deltas.

Exported weight layout:
1. Inner WaveNet weights in the stock WaveNet export order.
2. Hypernetwork parameters in ``named_parameters()`` order.
"""

from collections.abc import Mapping as _Mapping
from collections.abc import Sequence as _Sequence
from contextlib import nullcontext as _nullcontext
from copy import deepcopy as _deepcopy
from typing import Any as _Any
from typing import Optional as _Optional
from typing import cast as _cast

import numpy as _np
import torch as _torch
from torch.func import functional_call as _functional_call

from ..wavenet._wavenet import WaveNet as _WaveNet
from ._base import ParametricNet as _ParametricNet
from ._hypernet import Hypernetwork as _Hypernetwork
from ._spec import ParamSpec as _ParamSpec

_DEFAULT_HYPERNET_SELECTOR = {"exclude_suffixes": ["_conv.weight"]}
_WeightsLike = _Sequence[float] | _np.ndarray | _torch.Tensor


class HyperWaveNet(_ParametricNet):
    def __init__(
        self,
        *,
        template: _WaveNet,
        hypernet: _Hypernetwork,
        param_specs: _Sequence[_ParamSpec],
        wavenet_config: _Mapping[str, _Any],
        sample_rate: _Optional[float] = None,
    ):
        super().__init__(param_specs=param_specs, sample_rate=sample_rate)
        self._template = template
        self._hypernet = hypernet
        self._wavenet_config = _deepcopy(dict(wavenet_config))
        self._sync_wavenet_config_state()

    @classmethod
    def parse_config(cls, config: dict[str, _Any]) -> dict[str, _Any]:
        config = super().parse_config(config)
        sample_rate = config.pop("sample_rate", None)
        raw_params = config.pop("params", None)
        if raw_params is None:
            raise ValueError("HyperWaveNet config must define a params array")
        param_specs = tuple(_ParamSpec.from_dict(spec) for spec in raw_params)
        if len(param_specs) == 0:
            raise ValueError("HyperWaveNet config must define at least one ParamSpec")

        hypernet_config = dict(config.pop("hypernet", {}))
        if "selector" not in hypernet_config:
            hypernet_config["selector"] = _deepcopy(_DEFAULT_HYPERNET_SELECTOR)

        wavenet_config = _deepcopy(config)
        template = _WaveNet.init_from_config(config)
        hypernet = _Hypernetwork.from_config(
            input_dim=sum(spec.num_inputs for spec in param_specs),
            named_shapes={
                name: parameter.shape
                for name, parameter in template.named_parameters()
            },
            config=hypernet_config,
        )
        return {
            "template": template,
            "hypernet": hypernet,
            "param_specs": param_specs,
            "sample_rate": sample_rate,
            "wavenet_config": wavenet_config,
        }

    @property
    def pad_start_default(self) -> bool:
        return True

    @property
    def receptive_field(self) -> int:
        return self._template.receptive_field

    def _forward_mps_safe(self, x: _torch.Tensor, **kwargs) -> _torch.Tensor:
        # Slimmable templates draw ONE random channel width per forward (stock
        # `context_adjust_to_random` semantics). Set it here -- ABOVE BaseNet's MPS >65,536
        # stitching loop -- so every temporal segment of a single logical forward shares one
        # width, matching single-width inference. (Drawing it in `_run_conditioned` instead
        # would let each MPS segment redraw a width.) Only meaningful while training; eval
        # pins the full width, so non-slimmable models hit `_nullcontext` with no overhead.
        context = (
            self._template.context_adjust_to_random()
            if self.training and self._template.is_slimmable()
            else _nullcontext()
        )
        with context:
            return super()._forward_mps_safe(x, **kwargs)

    def _run_conditioned(self, x: _torch.Tensor, p: _torch.Tensor) -> _torch.Tensor:
        if p.ndim == 1:
            return self._apply_conditioned_weights(
                x,
                self._assemble_weight_dict(self._hypernet.generate(p)),
            )

        # Per-sample weights => one functional_call per sample. Deliberately a Python loop,
        # not vmap over functional_call: (1) the slimmable path mutates module state
        # (`_slimming_value`) once per forward (set in `_forward_mps_safe`) and a single width
        # must span the batch, and (2) vmap over the dilated Conv1d is problematic. The cost
        # is bounded -- `generate` is batched (called once) and the shared dilated convs (~82%
        # of the params) are passed by reference, so the loop only re-adds the cheap subset.
        deltas = self._hypernet.generate(p)
        if x.shape[0] != p.shape[0]:
            raise ValueError(
                f"Input batch size {x.shape[0]} must match encoded params batch size {p.shape[0]}"
            )
        outputs = []
        for i in range(x.shape[0]):
            outputs.append(
                self._apply_conditioned_weights(
                    x[i : i + 1].contiguous(),
                    self._assemble_weight_dict(
                        {name: delta[i] for name, delta in deltas.items()}
                    ),
                )[0]
            )
        return _torch.stack(outputs, dim=0)

    def _assemble_weight_dict(
        self, deltas: _Mapping[str, _torch.Tensor]
    ) -> dict[str, _torch.Tensor]:
        weight_dict = {}
        for name, parameter in self._template.named_parameters():
            delta = deltas.get(name)
            weight_dict[name] = parameter if delta is None else parameter + delta
        return weight_dict

    def _apply_conditioned_weights(
        self,
        x: _torch.Tensor,
        weight_dict: dict[str, _torch.Tensor],
    ) -> _torch.Tensor:
        y = _functional_call(self._template, weight_dict, (x[:, None, :],))
        if y.shape[1] != 1:
            raise RuntimeError(
                f"Expected template WaveNet to return one channel; got shape {tuple(y.shape)}"
            )
        return y[:, 0, :]

    def _sync_wavenet_config_state(self) -> None:
        self._wavenet_config["head_scale"] = float(self._template._head_scale)

    def _export_inner_config(self) -> dict[str, _Any]:
        self._sync_wavenet_config_state()
        config = _deepcopy(self._wavenet_config)
        config["hypernet"] = _deepcopy(self._hypernet.config)
        return config

    def _export_weights(self) -> _np.ndarray:
        tensors = [
            _torch.from_numpy(self._template.export_weights()).to(dtype=_torch.float32)
        ]
        tensors.extend(
            parameter.detach().reshape(-1).cpu()
            for _, parameter in self._hypernet.named_parameters()
        )
        return _torch.cat(tensors).numpy()

    def import_weights(self, weights: _WeightsLike, i: int = 0) -> int:
        weights_tensor = _cast(
            _torch.Tensor,
            weights if isinstance(weights, _torch.Tensor) else _torch.tensor(weights),
        )
        if weights_tensor.ndim != 1:
            raise ValueError(
                f"HyperWaveNet weights must be a flat 1-D sequence; got shape {tuple(weights_tensor.shape)}"
            )

        i = self._template.import_weights(weights_tensor, i)
        self._sync_wavenet_config_state()
        hypernet_parameters = tuple(self._hypernet.named_parameters())
        # Tail-ownership assumption: the hypernet weights are expected to run to the END of
        # the blob, so `remaining` is everything after the base. `remaining == 0` seeds the
        # base only (e.g. from a stock WaveNet checkpoint) and leaves the hypernet as-is;
        # `remaining == expected` is a full parametric restore. Unlike the stock
        # `import_weights(weights, i) -> next_i` contract this rejects a blob with trailing
        # weights owned by another module, so HyperWaveNet can't yet be embedded as a
        # sub-component of a larger (e.g. packed) blob.
        remaining = len(weights_tensor) - i
        expected = sum(parameter.numel() for _, parameter in hypernet_parameters)
        if remaining == 0:
            return i
        if remaining != expected:
            raise ValueError(
                f"Expected either 0 or {expected} hypernetwork weights after the base "
                f"WaveNet blob, but found {remaining}"
            )

        for _, parameter in hypernet_parameters:
            n = parameter.numel()
            parameter.data.copy_(
                weights_tensor[i : i + n]
                .to(device=parameter.device, dtype=parameter.dtype)
                .reshape(parameter.shape)
            )
            i += n
        return i
