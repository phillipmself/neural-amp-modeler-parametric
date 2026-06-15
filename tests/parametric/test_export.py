"""
PA3, PA4, PA4b, PA6, PA5, PA5b, EC10, EC12 — ParametricWaveNet export/load tests.

These tests verify that:
- PA3: exported .nam has architecture == "ParametricWaveNet"
- PA4: exported config dict contains param_names, param_dim, and nominal_params
- PA4b: missing or mismatched nominal_params raises a clear error at construction
- PA6: _at_nominal_settings injects nominal_params so export completes without
       a missing-params-arg error
- PA5: load_parametric_nam() reconstructs a ParametricWaveNet from an exported dict
- PA5b: load_parametric_nam() delegates "WaveNet" files to init_from_nam (no exception)
- EC10: forward pass before and after export/reload round-trip gives identical output
- EC12: load_parametric_nam() given an unknown architecture raises a clear ValueError
"""

import json as _json
from pathlib import Path as _Path
from typing import cast

import pytest
import torch

from nam.models.parametric import ParametricWaveNet, load_parametric_nam
from nam.models.parametric._model import _ChannelAdapter
from nam.models.wavenet import WaveNet as _WaveNet

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


# ---------------------------------------------------------------------------
# Tiny plain-WaveNet config for PA5b (needs to export as "WaveNet" architecture)
# ---------------------------------------------------------------------------

_TINY_WAVENET_CONFIG = {
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
}


# ---------------------------------------------------------------------------
# PA5 — load_parametric_nam() round-trip reconstructs ParametricWaveNet
# ---------------------------------------------------------------------------


def test_pa5_load_parametric_nam_roundtrip(tmp_path):
    """PA5: export a ParametricWaveNet to a dict, reload via load_parametric_nam,
    assert the result is a ParametricWaveNet with the original config."""
    model = _build(_SINGLE_C_CONFIG)
    model.export(tmp_path, basename="model")
    with open(tmp_path / "model.nam") as fp:
        nam = _json.load(fp)

    loaded = load_parametric_nam(nam)

    assert isinstance(loaded, ParametricWaveNet), (
        f"Expected ParametricWaveNet, got {type(loaded).__name__}"
    )
    assert loaded._param_names == model._param_names
    assert loaded._param_dim == model._param_dim
    torch.testing.assert_close(loaded._nominal_params, model._nominal_params)


# ---------------------------------------------------------------------------
# PA5b — load_parametric_nam() delegates "WaveNet" to init_from_nam
# ---------------------------------------------------------------------------


def test_pa5b_load_parametric_nam_delegates_wavenet(tmp_path):
    """PA5b: load_parametric_nam() on a "WaveNet" .nam file returns a non-parametric
    model (whatever init_from_nam returns) without raising an exception."""
    wavenet = _WaveNet.init_from_config(_TINY_WAVENET_CONFIG)
    wavenet.export(tmp_path, basename="plain")
    with open(tmp_path / "plain.nam") as fp:
        nam = _json.load(fp)

    assert nam["architecture"] == "WaveNet", (
        f"Expected 'WaveNet', got '{nam['architecture']}'"
    )

    # Must not raise; must return something that is NOT a ParametricWaveNet
    loaded = load_parametric_nam(nam)
    assert not isinstance(loaded, ParametricWaveNet), (
        "load_parametric_nam must not return a ParametricWaveNet for a 'WaveNet' file"
    )


# ---------------------------------------------------------------------------
# EC10 — forward pass identical before and after export/reload
# ---------------------------------------------------------------------------


def test_ec10_forward_identical_after_roundtrip(tmp_path):
    """EC10: export a ParametricWaveNet, reload it, run forward with the same inputs —
    outputs must be bit-for-bit identical (float32 round-trip via JSON list)."""
    model = _build(_SINGLE_C_CONFIG)
    model.eval()

    rf = model.receptive_field
    x = torch.randn(1, rf + 64)
    p = torch.tensor([0.7])

    with torch.no_grad():
        y_before = model(x, p, pad_start=False)

    model.export(tmp_path, basename="model")
    with open(tmp_path / "model.nam") as fp:
        nam = _json.load(fp)

    loaded = load_parametric_nam(nam)
    loaded.eval()

    with torch.no_grad():
        y_after = loaded(x, p, pad_start=False)

    torch.testing.assert_close(
        y_after,
        y_before,
        msg="Forward pass changed after export/reload round-trip",
    )


# ---------------------------------------------------------------------------
# X1 — _export_input_output snapshot is computed at nominal_params, not zeros
# ---------------------------------------------------------------------------


def test_x1_export_snapshot_uses_nominal_params():
    """X1: _export_input_output() snapshot must match the nominal-params forward pass.

    Uses non-zero adapter weights so that zero params and nominal params give
    observably different outputs. Would fail on the old zeros implementation.
    """
    import math
    import numpy as np

    # Low sample rate keeps the 3-second sweep small (300 samples total).
    config = {**_SINGLE_C_CONFIG, "nominal_params": [1.0], "sample_rate": 100.0}
    model = _build(config)
    model.eval()

    # Give the adapter non-trivial weights so nominal p != zero p has a real effect.
    with torch.no_grad():
        for sub in model._adapter._adapters.values():
            cast(_ChannelAdapter, sub)
            torch.nn.init.constant_(sub.gamma_map.weight, 0.1)  # type: ignore[attr-defined]

    x_np, y_snap_np = model._export_input_output()

    # Reconstruct the same input signal the method uses internally.
    rate = model.sample_rate
    assert rate is not None
    n = int(rate)
    x = torch.cat(
        [
            torch.zeros(n),
            0.5 * torch.sin(
                2.0 * math.pi * 220.0 * torch.linspace(0.0, 1.0, n + 1)[:-1]
            ),
            torch.zeros(n),
        ]
    )

    p_nominal = model._nominal_params
    p_zeros = torch.zeros(model._param_dim)

    with torch.no_grad():
        y_nominal = model(x, p_nominal, pad_start=True).numpy()
        y_zeros = model(x, p_zeros, pad_start=True).numpy()

    np.testing.assert_allclose(
        y_snap_np,
        y_nominal,
        rtol=1e-5,
        err_msg="_export_input_output snapshot must match nominal_params run",
    )
    assert not np.allclose(y_snap_np, y_zeros), (
        "_export_input_output snapshot matched zero-params run; "
        "nominal_params may not be applied"
    )


# ---------------------------------------------------------------------------
# EC12 — unknown architecture raises a clear ValueError
# ---------------------------------------------------------------------------


def test_ec12_unknown_architecture_raises():
    """EC12: load_parametric_nam() given an unknown architecture raises ValueError
    with the unknown architecture name in the error message."""
    dummy_nam = {
        "architecture": "UnknownArch",
        "config": {},
        "weights": [],
    }
    with pytest.raises(ValueError, match="UnknownArch"):
        load_parametric_nam(dummy_nam)
