"""
PA3, PA4, PA4b, PA6 — ParametricWaveNet export config tests.

These tests verify that:
- PA3: exported .nam has architecture == "ParametricWaveNet"
- PA4: exported config dict contains param_names, param_dim, and nominal_params
- PA4b: missing or mismatched nominal_params raises a clear error at construction
- PA6: _at_nominal_settings injects nominal_params so export completes without
       a missing-params-arg error
"""

import json as _json
from pathlib import Path as _Path
from typing import cast

import pytest
import torch

from nam.models.parametric import ParametricWaveNet
from nam.models.parametric._model import _ChannelAdapter

# ---------------------------------------------------------------------------
# Tiny single-channel config with nominal_params (C1.2 requirement)
# ---------------------------------------------------------------------------

_SINGLE_C_CONFIG = {
    "layers_configs": [
        {
            "input_size": 1,
            "condition_size": 1,
            "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
            "channels": 4,
            "kernel_size": 2,
            "dilations": [1, 2],
            "activation": "Tanh",
        }
    ],
    "head_scale": 1.0,
    "param_names": ["gain"],
    "param_dim": 1,
    "nominal_params": [0.5],
    "sample_rate": 44100.0,
}

_MULTI_C_CONFIG = {
    "layers_configs": [
        {
            "input_size": 1,
            "condition_size": 1,
            "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
            "channels": 8,
            "kernel_size": 2,
            "dilations": [1, 2],
            "activation": "Tanh",
        },
        {
            "input_size": 8,
            "condition_size": 1,
            "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
            "channels": 4,
            "kernel_size": 2,
            "dilations": [1, 2],
            "activation": "Tanh",
        },
    ],
    "head_scale": 0.02,
    "param_names": ["gain", "treble"],
    "param_dim": 2,
    "nominal_params": [0.5, 0.3],
    "sample_rate": 44100.0,
}


def _build(config):
    return ParametricWaveNet.init_from_config(config)


# ---------------------------------------------------------------------------
# PA3 — exported .nam architecture is "ParametricWaveNet"
# ---------------------------------------------------------------------------


def test_pa3_export_architecture_string(tmp_path):
    """PA3: .nam file has architecture == 'ParametricWaveNet'."""
    model = _build(_SINGLE_C_CONFIG)
    model.export(tmp_path, basename="model")
    with open(tmp_path / "model.nam") as fp:
        nam = _json.load(fp)
    assert nam["architecture"] == "ParametricWaveNet", (
        f"Expected 'ParametricWaveNet', got '{nam['architecture']}'"
    )


# ---------------------------------------------------------------------------
# PA4 — exported config contains param_names, param_dim, nominal_params
# ---------------------------------------------------------------------------


def test_pa4_export_config_contains_required_keys(tmp_path):
    """PA4: exported config dict has param_names, param_dim, and nominal_params."""
    model = _build(_SINGLE_C_CONFIG)
    model.export(tmp_path, basename="model")
    with open(tmp_path / "model.nam") as fp:
        nam = _json.load(fp)
    cfg = nam["config"]
    assert "param_names" in cfg, "param_names missing from exported config"
    assert "param_dim" in cfg, "param_dim missing from exported config"
    assert "nominal_params" in cfg, "nominal_params missing from exported config"
    # Check values are correct
    assert cfg["param_names"] == ["gain"]
    assert cfg["param_dim"] == 1
    assert cfg["nominal_params"] == [0.5]


def test_pa4_nominal_params_is_json_serializable(tmp_path):
    """PA4: nominal_params in exported config is a plain list of Python floats."""
    model = _build(_MULTI_C_CONFIG)
    model.export(tmp_path, basename="model")
    with open(tmp_path / "model.nam") as fp:
        nam = _json.load(fp)
    np_field = nam["config"]["nominal_params"]
    assert isinstance(np_field, list), "nominal_params must be a JSON list"
    for v in np_field:
        assert isinstance(v, float), f"Each nominal_params element must be float, got {type(v)}"
    # Use approx because float32 round-trip (tensor.tolist()) may differ from
    # the original Python float at float64 precision (e.g. 0.3 → 0.30000001...).
    assert np_field == pytest.approx([0.5, 0.3], rel=1e-5)


# ---------------------------------------------------------------------------
# PA4b — missing or mismatched nominal_params raises a clear error
# ---------------------------------------------------------------------------


def test_pa4b_missing_nominal_params_raises():
    """PA4b: constructing ParametricWaveNet without nominal_params raises a ValueError
    that mentions 'nominal_params' in the message."""
    config_no_nominal = {k: v for k, v in _SINGLE_C_CONFIG.items()
                         if k != "nominal_params"}
    with pytest.raises(ValueError, match="nominal_params"):
        _build(config_no_nominal)


def test_pa4b_nominal_params_length_mismatch_raises():
    """PA4b: nominal_params length != param_dim raises a clear ValueError."""
    bad_config = dict(_SINGLE_C_CONFIG)
    # param_dim=1 but nominal_params has 3 values
    bad_config = {**_SINGLE_C_CONFIG, "nominal_params": [0.1, 0.2, 0.3]}
    with pytest.raises(ValueError, match="nominal_params"):
        _build(bad_config)


# ---------------------------------------------------------------------------
# PA6 — _at_nominal_settings uses configured nominal_params (no missing-arg error)
# ---------------------------------------------------------------------------


def test_pa6_at_nominal_settings_does_not_raise():
    """PA6: _at_nominal_settings(x) injects self._nominal_params so it runs without
    a missing-params-arg error. Result is a sane tensor with correct shape."""
    model = _build(_SINGLE_C_CONFIG)
    model.eval()
    rf = model.receptive_field
    # _at_nominal_settings is called with 1-D x (from _metadata_loudness_x)
    # but our forward handles scalar inputs too; test with 2-D (1, L) for clarity.
    x = torch.randn(1, rf + 64)
    with torch.no_grad():
        y = model._at_nominal_settings(x)
    assert isinstance(y, torch.Tensor), "_at_nominal_settings must return a tensor"
    assert not torch.isnan(y).any(), "Output contains NaN"
    assert not torch.isinf(y).any(), "Output contains Inf"


def test_pa6_at_nominal_settings_uses_configured_nominal(tmp_path):
    """PA6: export() calls _at_nominal_settings internally (via _metadata_loudness).
    Verify the entire export completes without error when nominal_params is set."""
    model = _build(_SINGLE_C_CONFIG)
    # export() calls _get_non_user_metadata() → _metadata_loudness() →
    # _at_nominal_settings(), so a successful export proves the override works.
    model.export(tmp_path, basename="model")
    # Sanity: exported file exists and is valid JSON
    nam_file = tmp_path / "model.nam"
    assert nam_file.exists()
    with open(nam_file) as fp:
        nam = _json.load(fp)
    assert nam["architecture"] == "ParametricWaveNet"


def test_pa6_nominal_params_differ_from_zero():
    """PA6: verify _at_nominal_settings uses the actual nominal_params, not zeros.

    Build a model that has been "trained" (non-zero adapter weights) so that
    zero params and nominal params give different outputs. Without this test, a
    naive implementation that always passes zeros would still pass PA6 if
    nominal_params=[0.0, ...].
    """
    # Use a nonzero nominal setting
    config = {**_SINGLE_C_CONFIG, "nominal_params": [1.0]}
    model = _build(config)
    model.eval()

    # Manually set non-zero adapter weights so that the adapter is no longer
    # a no-op for non-zero p. Any non-trivial weight suffices.
    with torch.no_grad():
        for sub in model._adapter._adapters.values():
            sub_typed = cast(_ChannelAdapter, sub)
            torch.nn.init.constant_(sub_typed.gamma_map.weight, 0.01)

    rf = model.receptive_field
    x = torch.randn(1, rf + 32)

    with torch.no_grad():
        y_nominal = model._at_nominal_settings(x)
        # Compare against explicit p=zeros call
        y_zeros = model(x, torch.zeros(1), pad_start=False)

    # With non-zero adapter weights and nominal p=1.0 vs zero p, outputs must differ
    assert not torch.equal(y_nominal, y_zeros), (
        "_at_nominal_settings must use nominal_params, not zeros"
    )
