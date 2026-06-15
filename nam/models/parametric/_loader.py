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
        param_names = config.pop("param_names")
        param_dim = config.pop("param_dim")
        nominal_params = config.pop("nominal_params")

        # The remaining config is the inner WaveNet in .nam export format.
        # _init_wavenet normalizes "layers" keys and activation dict format
        # to the construction format WaveNet.parse_config expects.
        # We pass a synthetic nam dict so _init_wavenet can extract the top-level
        # keys it needs ("config" with "layers", "head", "head_scale").
        inner_net = _init_wavenet(
            config=config,
            sample_rate=sample_rate,
        )

        model = ParametricWaveNet(
            net=inner_net._net,  # unwrap the WaveNet wrapper to get the inner _WaveNet
            param_names=param_names,
            param_dim=param_dim,
            nominal_params=nominal_params,
            sample_rate=sample_rate,
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
