"""
Net-agnostic hypernetwork utilities for parametric NAM models.

A ``Hypernetwork`` maps an *encoded* parameter vector to additive **deltas** for a
configurable subset of some template network's weight tensors. It knows nothing about
WaveNet/LSTM internals -- it operates purely on the ``{name: shape}`` mapping passed in --
so the same class is reused across architectures. The net-specific knowledge (e.g. "share
the dilated convs") lives in the *caller*: the model wrapper (``HyperWaveNet``, Task 5) is
responsible for passing a ``selector`` that excludes the tensors it wants to keep shared.
Because of that, the generic default here is intentionally "target every parameter"; it is
NOT WaveNet-aware. Callers that want the cheap-subset behaviour must say so explicitly via
``selector={"exclude_suffixes": ["_conv.weight"]}``.

Defaults favour *starting small* (per project direction: scale capacity up only if
validation shows it's needed). The default mode is ``low_rank`` with a small rank and no
hidden trunk, so the generated deltas are low-rank and the param->delta map is simple. The
two scale-up levers are ``hidden_sizes`` (add param-space nonlinearity) and ``rank`` (raise
the per-tensor delta rank); ``mode="full"`` removes the rank constraint entirely.

Init contract: the readout is zero-initialized so every delta == 0 at construction (the
model equals the bare template for all params). For ``low_rank`` targets we also register a
tiny factor-anchor seed from a *local* generator: alternating rank components get a
nonzero U anchor or a nonzero V anchor, never both, so the anchored product is still
exactly zero while the first backward pass reaches both factor slices. That seed is
deterministic and independent of the ambient global ``torch`` RNG state. (The trunk's
``Linear`` layers still use standard PyTorch init, which draws from the global RNG like any
``nn.Module`` -- only the extra low-rank anchoring is made global-RNG-neutral.)
"""

import math as _math
from collections.abc import Mapping as _Mapping
from collections.abc import Sequence as _Sequence
from copy import deepcopy as _deepcopy
from dataclasses import dataclass as _dataclass
from typing import Any as _Any
from typing import Optional as _Optional
from typing import cast as _cast

import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F

from .._activations import get_activation as _get_activation

_FULL = "full"
_LOW_RANK = "low_rank"
_VALID_MODES = (_FULL, _LOW_RANK)
_ALLOWED_CONFIG_KEYS = frozenset(
    {"activation", "hidden_sizes", "mode", "rank", "selector"}
)
_ALLOWED_SELECTOR_KEYS = frozenset(
    {"exclude_names", "exclude_suffixes", "include_names", "include_suffixes"}
)
# "Start small" defaults: low-rank deltas, smallest real rank, linear (no-hidden) trunk.
_DEFAULT_MODE = _LOW_RANK
_DEFAULT_RANK = 2
_DEFAULT_HIDDEN_SIZES: tuple[int, ...] = ()
_DEFAULT_ACTIVATION = "ReLU"
# Std of the local symmetry-breaking seed on the low-rank factor anchors.
_LOW_RANK_SEED_STD = 1.0e-2
# Fixed seed for the local generator so init is deterministic without touching global RNG.
_DEFAULT_LOW_RANK_SEED = 0


@_dataclass(frozen=True)
class _TargetLayout:
    name: str
    shape: _torch.Size
    output_width: int
    mode: str
    rank: _Optional[int] = None
    out_features: _Optional[int] = None
    rest_features: _Optional[int] = None

    def decode(
        self, flat: _torch.Tensor, *, anchor: _Optional[_torch.Tensor] = None
    ) -> _torch.Tensor:
        leading_shape = tuple(flat.shape[:-1])
        if self.mode == _FULL:
            return flat.reshape(*leading_shape, *self.shape)

        if self.rank is None or self.out_features is None or self.rest_features is None:
            raise RuntimeError(
                f"Low-rank target layout for {self.name!r} is incomplete"
            )

        u_width = self.out_features * self.rank
        v_width = self.rank * self.rest_features
        u_flat, v_flat = _torch.split(flat, [u_width, v_width], dim=-1)
        u = u_flat.reshape(*leading_shape, self.out_features, self.rank)
        v = v_flat.reshape(*leading_shape, self.rank, self.rest_features)
        if anchor is not None:
            if tuple(anchor.shape) != (self.output_width,):
                raise RuntimeError(
                    f"Low-rank anchor for {self.name!r} has shape {tuple(anchor.shape)}; "
                    f"expected {(self.output_width,)!r}"
                )
            anchor_u_flat, anchor_v_flat = _torch.split(
                anchor, [u_width, v_width], dim=-1
            )
            u = u + anchor_u_flat.reshape(self.out_features, self.rank)
            v = v + anchor_v_flat.reshape(self.rank, self.rest_features)
        return _torch.matmul(u, v).reshape(*leading_shape, *self.shape)

    @property
    def is_low_rank(self) -> bool:
        return self.mode == _LOW_RANK

    @property
    def u_width(self) -> int:
        if self.rank is None or self.out_features is None:
            raise RuntimeError(
                f"Low-rank target layout for {self.name!r} is incomplete"
            )
        return self.out_features * self.rank

    @property
    def v_width(self) -> int:
        if self.rank is None or self.rest_features is None:
            raise RuntimeError(
                f"Low-rank target layout for {self.name!r} is incomplete"
            )
        return self.rank * self.rest_features


class _ZeroLinear(_nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = _nn.Parameter(_torch.zeros((out_features, in_features)))
        self.bias = _nn.Parameter(_torch.zeros((out_features,)))

    def forward(self, x: _torch.Tensor) -> _torch.Tensor:
        return _F.linear(x, self.weight, self.bias)


def _validate_positive_int(value: _Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer; got {value!r}")
    return value


def _validate_hidden_sizes(hidden_sizes: _Sequence[int]) -> tuple[int, ...]:
    if isinstance(hidden_sizes, (str, bytes)):
        raise ValueError("hidden_sizes must be a sequence of positive integers")
    return tuple(
        _validate_positive_int(hidden_size, name="hidden_sizes entries")
        for hidden_size in hidden_sizes
    )


def _coerce_string_sequence(value: _Any, *, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence, not a string")
    return tuple(str(item) for item in value)


def _validate_named_shapes(
    named_shapes: _Mapping[str, _torch.Size],
) -> tuple[tuple[str, _torch.Size], ...]:
    items = []
    for name, shape in named_shapes.items():
        if not name:
            raise ValueError("named_shapes keys must be non-empty strings")
        items.append((str(name), _torch.Size(shape)))
    if len(items) == 0:
        raise ValueError("named_shapes must contain at least one parameter tensor")
    return tuple(items)


def _resolve_target_names(
    named_shape_items: tuple[tuple[str, _torch.Size], ...],
    selector: _Optional[dict[str, _Any]],
) -> tuple[str, ...]:
    ordered_names = tuple(name for name, _ in named_shape_items)
    if selector is None:
        # Net-agnostic default: target every parameter. The model wrapper (Task 5) is
        # responsible for passing a selector to keep tensors (e.g. dilated convs) shared.
        return ordered_names
    selector_config = _normalize_selector_config(selector)
    known_names = set(ordered_names)
    include_names = selector_config["include_names"]
    exclude_names = selector_config["exclude_names"]
    unknown_names = sorted((set(include_names) | set(exclude_names)) - known_names)
    if unknown_names:
        raise ValueError(
            f"Hypernetwork selector references unknown parameter names: {unknown_names}"
        )

    include_suffixes = selector_config["include_suffixes"]
    exclude_suffixes = selector_config["exclude_suffixes"]

    def _matches_suffixes(name: str, suffixes: tuple[str, ...]) -> bool:
        return any(name.endswith(str(suffix)) for suffix in suffixes)

    use_includes = len(include_names) > 0 or len(include_suffixes) > 0
    resolved = []
    for name in ordered_names:
        selected = (
            name in include_names or _matches_suffixes(name, include_suffixes)
            if use_includes
            else True
        )
        if not selected:
            continue
        if name in exclude_names or _matches_suffixes(name, exclude_suffixes):
            continue
        resolved.append(name)
    if len(resolved) == 0:
        raise ValueError("Hypernetwork must target at least one parameter tensor")
    return tuple(resolved)


def _normalize_selector_config(
    selector: dict[str, _Any],
) -> dict[str, tuple[str, ...]]:
    selector_config = dict(selector)
    unexpected = sorted(set(selector_config) - _ALLOWED_SELECTOR_KEYS)
    if unexpected:
        raise ValueError(
            f"Hypernetwork selector config has unexpected keys: {unexpected}"
        )
    return {
        "include_names": _coerce_string_sequence(
            selector_config.get("include_names"),
            name="Hypernetwork selector include_names",
        ),
        "exclude_names": _coerce_string_sequence(
            selector_config.get("exclude_names"),
            name="Hypernetwork selector exclude_names",
        ),
        "include_suffixes": _coerce_string_sequence(
            selector_config.get("include_suffixes"),
            name="Hypernetwork selector include_suffixes",
        ),
        "exclude_suffixes": _coerce_string_sequence(
            selector_config.get("exclude_suffixes"),
            name="Hypernetwork selector exclude_suffixes",
        ),
    }


def _serialize_selector_config(
    selector_config: dict[str, tuple[str, ...]],
) -> dict[str, list[str]]:
    return {
        key: list(values) for key, values in selector_config.items() if len(values) > 0
    }


def _build_target_layout(
    *,
    name: str,
    shape: _torch.Size,
    mode: str,
    rank: _Optional[int],
) -> _TargetLayout:
    if mode == _FULL:
        return _TargetLayout(
            name=name,
            shape=shape,
            output_width=shape.numel(),
            mode=_FULL,
        )

    if rank is None:
        raise ValueError("Low-rank hypernetwork mode requires a positive integer rank")
    rank = _validate_positive_int(rank, name="rank")

    out_features = int(shape[0]) if len(shape) > 0 else 0
    rest_features = int(_math.prod(shape[1:])) if len(shape) > 1 else 1
    if out_features <= 1 or rest_features <= 1:
        return _TargetLayout(
            name=name,
            shape=shape,
            output_width=shape.numel(),
            mode=_FULL,
        )

    capped_rank = min(rank, out_features, rest_features)
    return _TargetLayout(
        name=name,
        shape=shape,
        output_width=capped_rank * (out_features + rest_features),
        mode=_LOW_RANK,
        rank=capped_rank,
        out_features=out_features,
        rest_features=rest_features,
    )


class Hypernetwork(_nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        named_shapes: _Mapping[str, _torch.Size],
        target_names: _Sequence[str],
        hidden_sizes: _Sequence[int] = _DEFAULT_HIDDEN_SIZES,
        activation: _Any = _DEFAULT_ACTIVATION,
        mode: str = _DEFAULT_MODE,
        rank: _Optional[int] = _DEFAULT_RANK,
        selector: _Optional[dict[str, _Any]] = None,
        low_rank_seed_std: float = _LOW_RANK_SEED_STD,
        seed: _Optional[int] = None,
    ):
        super().__init__()
        self._input_dim = _validate_positive_int(input_dim, name="input_dim")
        if mode not in _VALID_MODES:
            raise ValueError(
                f"Unsupported hypernetwork mode {mode!r}; expected one of {list(_VALID_MODES)!r}"
            )
        # Validate the activation eagerly so a bad name fails at construction even when the
        # trunk is empty (hidden_sizes == ()) and would otherwise never instantiate it.
        if not isinstance(activation, (str, dict)):
            raise ValueError(
                "activation must be a string name or an activation config dict; "
                f"got {type(activation).__name__}"
            )
        try:
            _get_activation(activation)
        except Exception as exc:
            raise ValueError(f"Invalid activation {activation!r}: {exc}") from exc

        named_shape_items = _validate_named_shapes(named_shapes)
        target_names_tuple = tuple(str(name) for name in target_names)
        if len(target_names_tuple) == 0:
            raise ValueError("target_names must contain at least one parameter tensor")
        if len(set(target_names_tuple)) != len(target_names_tuple):
            raise ValueError("target_names must be unique")

        ordered_names = tuple(name for name, _ in named_shape_items)
        self._ordered_names = ordered_names
        unknown_target_names = sorted(set(target_names_tuple) - set(ordered_names))
        if unknown_target_names:
            raise ValueError(
                f"target_names contain unknown parameter names: {unknown_target_names}"
            )
        ordered_named_shapes = dict(named_shape_items)
        ordered_targets = tuple(
            name for name in ordered_names if name in target_names_tuple
        )
        if ordered_targets != target_names_tuple:
            raise ValueError(
                "target_names must follow the same deterministic order as named_shapes"
            )

        hidden_sizes_tuple = _validate_hidden_sizes(hidden_sizes)
        selector_config = (
            None if selector is None else _normalize_selector_config(selector)
        )
        if selector is not None:
            selected_targets = _resolve_target_names(named_shape_items, selector)
            if selected_targets != target_names_tuple:
                raise ValueError(
                    "selector must resolve to the same deterministic target_names order"
                )
        target_layouts = tuple(
            _build_target_layout(
                name=name,
                shape=ordered_named_shapes[name],
                mode=mode,
                rank=rank,
            )
            for name in ordered_targets
        )

        trunk_layers: list[_nn.Module] = []
        in_features = self._input_dim
        for hidden_size in hidden_sizes_tuple:
            trunk_layers.append(_nn.Linear(in_features, hidden_size))
            trunk_layers.append(_get_activation(activation))
            in_features = hidden_size
        self._trunk = (
            _nn.Sequential(*trunk_layers) if len(trunk_layers) > 0 else _nn.Identity()
        )

        self._target_layouts = target_layouts
        self.target_names = tuple(layout.name for layout in target_layouts)
        self._mode = mode
        self._rank = rank
        self._hidden_sizes = hidden_sizes_tuple
        self._activation = activation
        self._selector_config = selector_config
        self._low_rank_seed_std = float(low_rank_seed_std)
        self._seed = seed
        final_output_dim = sum(layout.output_width for layout in target_layouts)
        self._final = _ZeroLinear(in_features, final_output_dim)
        # Zero-init the readout so every delta == 0 at construction (model == base net for
        # all params). Consequence: at the very first optimizer step the trunk receives no
        # gradient (dL/dh = Wᵀ·dL/dflat == 0 while W == 0); the readout itself DOES get a
        # gradient from the upstream weight loss, becomes nonzero, and the trunk trains from
        # then on. This one-step warmup is expected -- do not "fix" it by dropping zero-init.
        self.register_buffer("_low_rank_anchor", self._build_low_rank_anchor())

    @classmethod
    def from_config(
        cls,
        *,
        input_dim: int,
        named_shapes: _Mapping[str, _torch.Size],
        config: _Optional[dict[str, _Any]] = None,
    ) -> "Hypernetwork":
        config_dict = {} if config is None else dict(config)
        unexpected = sorted(set(config_dict) - _ALLOWED_CONFIG_KEYS)
        if unexpected:
            raise ValueError(f"Hypernetwork config has unexpected keys: {unexpected}")

        named_shape_items = _validate_named_shapes(named_shapes)
        selector = config_dict.get("selector")
        target_names = _resolve_target_names(named_shape_items, selector)
        return cls(
            input_dim=input_dim,
            named_shapes=dict(named_shape_items),
            target_names=target_names,
            hidden_sizes=config_dict.get("hidden_sizes", _DEFAULT_HIDDEN_SIZES),
            activation=config_dict.get("activation", _DEFAULT_ACTIVATION),
            mode=config_dict.get("mode", _DEFAULT_MODE),
            rank=config_dict.get("rank", _DEFAULT_RANK),
            selector=selector,
        )

    @property
    def config(self) -> dict[str, _Any]:
        # Enough to reconstruct the same targeting/architecture given input_dim + named_shapes.
        # The deterministic low-rank anchors use fixed defaults, so the init-only knobs remain
        # intentionally omitted here.
        config: dict[str, _Any] = {
            "hidden_sizes": list(self._hidden_sizes),
            # deepcopy so a caller mutating the returned dict can't alias internal state.
            "activation": _deepcopy(self._activation),
            "mode": self._mode,
        }
        # rank only matters in low_rank mode; emitting it for full mode would persist a
        # meaningless field and confuse readers.
        if self._mode == _LOW_RANK:
            config["rank"] = self._rank
        selector_config = self._selector_config
        if selector_config is None and self.target_names != self._ordered_names:
            selector_config = {
                "include_names": self.target_names,
                "exclude_names": (),
                "include_suffixes": (),
                "exclude_suffixes": (),
            }
        if selector_config is not None:
            config["selector"] = _serialize_selector_config(selector_config)
        return config

    def generate(self, p: _torch.Tensor) -> dict[str, _torch.Tensor]:
        reference = self._final.weight
        p = _torch.as_tensor(p, device=reference.device, dtype=reference.dtype)
        if p.ndim not in (1, 2):
            raise ValueError(
                f"Expected encoded params to have shape (E,) or (B, E); got {tuple(p.shape)}"
            )
        if p.shape[-1] != self._input_dim:
            raise ValueError(
                f"Expected encoded params trailing dimension {self._input_dim}; got {p.shape[-1]}"
            )

        flat = self._final(self._trunk(p))
        # final_output_dim > 0 always: __init__ requires >=1 target and every parameter
        # tensor has numel >= 1, so the split below covers exactly `flat`.
        chunks = _torch.split(
            flat,
            [layout.output_width for layout in self._target_layouts],
            dim=-1,
        )
        anchors = _torch.split(
            _cast(_torch.Tensor, self._low_rank_anchor),
            [layout.output_width for layout in self._target_layouts],
            dim=-1,
        )
        return {
            layout.name: layout.decode(chunk, anchor=anchor)
            for layout, chunk, anchor in zip(self._target_layouts, chunks, anchors)
        }

    def param_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def state_count(self) -> int:
        return sum(tensor.numel() for tensor in self._serialized_tensors())

    def export_state(self) -> _torch.Tensor:
        return _torch.cat(
            [tensor.detach().reshape(-1).cpu() for tensor in self._serialized_tensors()]
        )

    def import_state(self, weights: _torch.Tensor, i: int = 0) -> int:
        if weights.ndim != 1:
            raise ValueError(
                f"Hypernetwork state must be a flat 1-D tensor; got shape {tuple(weights.shape)}"
            )
        expected = self.state_count()
        remaining = len(weights) - i
        if remaining != expected:
            raise ValueError(
                f"Expected {expected} serialized hypernetwork values, but found {remaining}"
            )

        with _torch.no_grad():
            for tensor in self._serialized_tensors():
                n = tensor.numel()
                tensor.copy_(
                    weights[i : i + n]
                    .to(device=tensor.device, dtype=tensor.dtype)
                    .reshape(tensor.shape)
                )
                i += n
        return i

    def _serialized_tensors(self) -> tuple[_torch.Tensor, ...]:
        return tuple(parameter for _, parameter in self.named_parameters()) + (
            _cast(_torch.Tensor, self._low_rank_anchor),
        )

    def _build_low_rank_anchor(self) -> _torch.Tensor:
        anchor = self._final.bias.detach().clone()
        if self._low_rank_seed_std <= 0.0:
            return anchor

        # Local generator: the symmetry-breaking seed is deterministic and does not consume
        # the global torch RNG. The trunk's standard Linear init still draws from the global
        # RNG -- that part is unavoidable.
        generator = _torch.Generator(device=self._final.bias.device)
        generator.manual_seed(
            _DEFAULT_LOW_RANK_SEED if self._seed is None else self._seed
        )
        offset = 0
        with _torch.no_grad():
            for layout in self._target_layouts:
                next_offset = offset + layout.output_width
                if layout.is_low_rank:
                    if (
                        layout.rank is None
                        or layout.out_features is None
                        or layout.rest_features is None
                    ):
                        raise RuntimeError(
                            f"Low-rank target layout for {layout.name!r} is incomplete"
                        )
                    # Alternate anchored rank components between U and V. Each component has
                    # exactly one anchored side, so the seeded U@V product is still exactly 0
                    # while the first backward pass reaches both factor slices.
                    u_anchor = anchor[offset : offset + layout.u_width].reshape(
                        layout.out_features, layout.rank
                    )
                    v_anchor = anchor[offset + layout.u_width : next_offset].reshape(
                        layout.rank, layout.rest_features
                    )
                    for rank_index in range(layout.rank):
                        if rank_index % 2 == 0:
                            u_anchor[:, rank_index].normal_(
                                0.0, self._low_rank_seed_std, generator=generator
                            )
                        else:
                            v_anchor[rank_index].normal_(
                                0.0, self._low_rank_seed_std, generator=generator
                            )
                offset = next_offset
        return anchor
