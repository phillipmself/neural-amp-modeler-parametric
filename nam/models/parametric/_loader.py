"""
load_parametric_nam: entry point for loading .nam files that may be ParametricWaveNet.

Dispatch logic:
- "ParametricWaveNet" → reconstruct inner WaveNet from .nam export format, build
                        ParametricWaveNet, then load the full weight blob.
- "WaveNet"           → delegate to nam.models._from_nam.init_from_nam (no changes there)
- anything else       → raise ValueError naming the unknown architecture

AD-2 mandates a distinct-name loader that never silently loads "WaveNet" as parametric.
This module is the preferred loader path; _from_nam.py is NOT edited.

Why _init_wavenet is used directly: ParametricWaveNet._export_config() emits the inner
WaveNet in .nam export format (using "layers" key and {"type": "Tanh"} activation dicts).
The WaveNet init path expects construction format ("layers_configs", activation strings).
_init_wavenet already handles this normalization so we reuse it rather than duplicating
the conversion logic. This mirrors how init_from_nam handles plain WaveNet files.
"""

from copy import deepcopy as _deepcopy
from typing import Any

import torch as _torch

from nam.models._from_nam import _init_wavenet as _init_wavenet
from nam.models._from_nam import init_from_nam as _init_from_nam

from ._model import ParametricWaveNet
from ._model import _DEFAULT_ADAPTER_ACTIVATION
from ._model import _DEFAULT_ADAPTER_BETA_SCALE
from ._model import _DEFAULT_ADAPTER_GAMMA_SCALE
from ._model import _DEFAULT_ADAPTER_HIDDEN_DIM
from ._spec import ParamSpec as _ParamSpec


def load_parametric_nam(nam_dict: dict) -> Any:
    """
    Load a model from a deserialized .nam dict.

    Parameters
    ----------
    nam_dict:
        The dict produced by deserializing a .nam JSON file, e.g. via
        ``json.load(fp)`` or ``Exportable._get_export_dict()``.

        Expected top-level keys: ``"architecture"``, ``"config"``, ``"weights"``.

    Returns
    -------
    A model instance.  ``ParametricWaveNet`` for parametric files; whatever
    ``init_from_nam`` returns (a ``WaveNet``-based model) for plain WaveNet files.

    Raises
    ------
    ValueError
        If ``architecture`` is neither ``"ParametricWaveNet"`` nor ``"WaveNet"``.
        The error message names the offending architecture string so callers can
        diagnose unknown files without inspecting the source.
    """
    architecture: str = nam_dict["architecture"]

    if architecture == "ParametricWaveNet":
        config = _deepcopy(nam_dict["config"])
        sample_rate = config.pop("sample_rate", None)
        raw_specs = config.pop("params")
        adapter_hidden_dim = int(
            config.pop("adapter_hidden_dim", _DEFAULT_ADAPTER_HIDDEN_DIM)
        )
        adapter_activation = _deepcopy(
            config.pop("adapter_activation", _DEFAULT_ADAPTER_ACTIVATION)
        )
        adapter_gamma_scale = float(
            config.pop("adapter_gamma_scale", _DEFAULT_ADAPTER_GAMMA_SCALE)
        )
        adapter_beta_scale = float(
            config.pop("adapter_beta_scale", _DEFAULT_ADAPTER_BETA_SCALE)
        )
        adapter_first_n_layers = config.pop("adapter_first_n_layers", None)
        adapter_last_n_layers = config.pop("adapter_last_n_layers", None)
        adapter_layer_groups = config.pop("adapter_layer_groups", None)
        param_specs = [_ParamSpec.from_dict(d) for d in raw_specs]

        # The remaining config is the inner WaveNet in .nam export format.
        # _init_wavenet normalizes "layers" keys and activation dict format
        # to the construction format WaveNet.parse_config expects.
        inner_net = _init_wavenet(
            config=config,
            sample_rate=sample_rate,
        )

        model = ParametricWaveNet(
            net=inner_net._net,  # unwrap the WaveNet wrapper to get the inner _WaveNet
            param_specs=param_specs,
            sample_rate=sample_rate,
            adapter_hidden_dim=adapter_hidden_dim,
            adapter_activation=adapter_activation,
            adapter_gamma_scale=adapter_gamma_scale,
            adapter_beta_scale=adapter_beta_scale,
            adapter_first_n_layers=adapter_first_n_layers,
            adapter_last_n_layers=adapter_last_n_layers,
            adapter_layer_groups=adapter_layer_groups,
        )

        # Weights are stored as a plain Python list in the .nam JSON; convert once
        # to a float32 tensor before dispatching through import_weights.
        weights = _torch.tensor(nam_dict["weights"], dtype=_torch.float32)
        model.import_weights(weights)
        return model

    if architecture == "WaveNet":
        # "WaveNet" files are handled by the existing loader unchanged.
        # Delegating rather than returning a ParametricWaveNet is required by AD-2:
        # a "WaveNet" file must never be silently loaded as parametric.
        return _init_from_nam(nam_dict)

    raise ValueError(
        f"Unsupported architecture: {architecture!r}. "
        "load_parametric_nam() handles 'ParametricWaveNet' (parametric) and "
        "'WaveNet' (delegated to init_from_nam). "
        "For other architectures, use nam.models._from_nam.init_from_nam() directly."
    )
