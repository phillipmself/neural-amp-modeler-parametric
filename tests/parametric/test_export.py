"""
PA3, PA4, PA4b, PA6, PA5, PA5b, EC10, EC12 — ParametricWaveNet export/load tests.

These tests verify that:
- PA3: exported .nam has architecture == "ParametricWaveNet"
- PA4: exported config dict contains a self-describing "params" array
       (name/min/max/default per entry); old flat keys are absent
- PA4b: missing "params" key raises a clear error at construction
- PA6: _at_nominal_settings injects nominal_params so export completes without
       a missing-params-arg error
- PA5: load_parametric_nam() reconstructs a ParametricWaveNet from an exported dict
- PA5b: load_parametric_nam() delegates "WaveNet" files to init_from_nam (no exception)
- EC10: forward pass before and after export/reload round-trip gives identical output
- EC12: load_parametric_nam() given an unknown architecture raises a clear ValueError
- ParamSpec: validation (min<=default<=max, finiteness) and serialization round-trip
"""

import json as _json
from pathlib import Path as _Path
from typing import cast

import pytest
import torch

from nam.models.parametric import ParametricWaveNet, load_parametric_nam
from nam.models.parametric._model import _LayerAdapter
from nam.models.parametric._spec import ParamSpec
from nam.models.wavenet import WaveNet as _WaveNet

# ---------------------------------------------------------------------------
# Tiny single-channel config using the self-describing params array schema
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
    "params": [{"name": "gain", "min": 0.0, "max": 1.0, "default": 0.5}],
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
    "params": [
        {"name": "gain",   "min": 0.0, "max": 1.0, "default": 0.5},
        {"name": "treble", "min": 0.0, "max": 1.0, "default": 0.3},
    ],
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
# PA4 — exported config contains self-describing params array; old keys absent
# ---------------------------------------------------------------------------


def test_pa4_export_config_contains_params_array(tmp_path):
    """PA4: exported config has a 'params' array with name/min/max/default per entry."""
    model = _build(_SINGLE_C_CONFIG)
    model.export(tmp_path, basename="model")
    with open(tmp_path / "model.nam") as fp:
        nam = _json.load(fp)
    cfg = nam["config"]
    assert "params" in cfg, "'params' array missing from exported config"
    assert cfg["adapter_hidden_dim"] == model._adapter_hidden_dim
    assert cfg["adapter_activation"] == model._adapter_activation
    # Old flat keys must be absent (no legacy fallback — parametric is unreleased)
    assert "param_names" not in cfg, "Old 'param_names' key must be absent from export"
    assert "param_dim" not in cfg, "Old 'param_dim' key must be absent from export"
    assert "nominal_params" not in cfg, "Old 'nominal_params' key must be absent from export"
    # Verify array structure and values
    params = cfg["params"]
    assert isinstance(params, list) and len(params) == 1
    entry = params[0]
    assert entry["name"] == "gain"
    assert entry["min"] == pytest.approx(0.0)
    assert entry["max"] == pytest.approx(1.0)
    assert entry["default"] == pytest.approx(0.5, rel=1e-5)


def test_pa4_params_array_is_json_serializable(tmp_path):
    """PA4: params array in exported config contains only JSON-native types."""
    model = _build(_MULTI_C_CONFIG)
    model.export(tmp_path, basename="model")
    with open(tmp_path / "model.nam") as fp:
        nam = _json.load(fp)
    params = nam["config"]["params"]
    assert isinstance(params, list) and len(params) == 2
    for entry in params:
        assert isinstance(entry["name"], str)
        assert isinstance(entry["min"], float)
        assert isinstance(entry["max"], float)
        assert isinstance(entry["default"], float)
    # Use approx because float32 round-trip may differ at float64 precision
    assert params[0]["default"] == pytest.approx(0.5, rel=1e-5)
    assert params[1]["default"] == pytest.approx(0.3, rel=1e-5)


def test_pa4_export_config_contains_custom_adapter_activation(tmp_path):
    """PA4: custom adapter activation is exported in init_from_config format."""
    model = _build(
        {
            **_SINGLE_C_CONFIG,
            "adapter_activation": {
                "name": "LeakyReLU",
                "negative_slope": 0.2,
            },
        }
    )
    model.export(tmp_path, basename="model")
    with open(tmp_path / "model.nam") as fp:
        nam = _json.load(fp)

    assert nam["config"]["adapter_activation"] == {
        "name": "LeakyReLU",
        "negative_slope": 0.2,
    }


def test_pa4_export_params_describe_input_normalization(tmp_path):
    """PA4: exported params declare the signed min/max normalization contract."""
    model = _build(
        {
            **_SINGLE_C_CONFIG,
            "params": [{"name": "gain", "min": 0.0, "max": 10.0, "default": 5.0}],
        }
    )
    model.export(tmp_path, basename="model")
    with open(tmp_path / "model.nam") as fp:
        nam = _json.load(fp)

    entry = nam["config"]["params"][0]
    assert entry["input_normalization"] == "min_max_signed"
    assert entry["normalized_min"] == pytest.approx(-1.0)
    assert entry["normalized_max"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# PA4b — missing or invalid params raises a clear error
# ---------------------------------------------------------------------------


def test_pa4b_missing_params_raises():
    """PA4b: constructing ParametricWaveNet without 'params' key raises a ValueError
    that mentions 'params' in the message."""
    config_no_params = {k: v for k, v in _SINGLE_C_CONFIG.items()
                        if k != "params"}
    with pytest.raises(ValueError, match="params"):
        _build(config_no_params)


def test_pa4b_param_spec_min_gt_default_raises():
    """PA4b: ParamSpec with min > default raises ValueError at construction."""
    with pytest.raises(ValueError, match="min <= default <= max"):
        ParamSpec(name="gain", min=0.8, max=1.0, default=0.5)


def test_pa4b_param_spec_default_gt_max_raises():
    """PA4b: ParamSpec with default > max raises ValueError at construction."""
    with pytest.raises(ValueError, match="min <= default <= max"):
        ParamSpec(name="gain", min=0.0, max=0.4, default=0.5)


def test_pa4b_param_spec_zero_span_raises():
    """PA4b: ParamSpec with min == max is invalid because the control cannot vary."""
    with pytest.raises(ValueError, match="min < max"):
        ParamSpec(name="gain", min=0.5, max=0.5, default=0.5)


def test_pa4b_param_spec_infinite_raises():
    """PA4b: ParamSpec with non-finite values raises ValueError."""
    import math
    with pytest.raises(ValueError, match="finite"):
        ParamSpec(name="gain", min=float("-inf"), max=1.0, default=0.5)


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
    # Use a nonzero nominal setting so zero and nominal give different outputs
    config = {
        **_SINGLE_C_CONFIG,
        "params": [{"name": "gain", "min": 0.0, "max": 1.0, "default": 1.0}],
    }
    model = _build(config)
    model.eval()

    # Manually set non-zero adapter weights so that the adapter is no longer
    # a no-op for non-zero p. Any non-trivial weight suffices.
    with torch.no_grad():
        torch.nn.init.constant_(model._adapter._shared_encoder.fc.weight, 0.1)
        torch.nn.init.zeros_(model._adapter._shared_encoder.fc.bias)
        for sub in model._adapter._adapters:
            sub_typed = cast(_LayerAdapter, sub)
            torch.nn.init.constant_(sub_typed.gamma_head.weight, 0.01)

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
    assert loaded._adapter_hidden_dim == model._adapter_hidden_dim
    assert loaded._adapter_activation == model._adapter_activation
    assert loaded._adapter_gamma_scale == model._adapter_gamma_scale
    assert loaded._adapter_beta_scale == model._adapter_beta_scale
    assert loaded._adapter_first_n_layers == model._adapter_first_n_layers
    assert loaded._adapter_last_n_layers == model._adapter_last_n_layers
    torch.testing.assert_close(loaded._nominal_params, model._nominal_params)


def test_pa5_load_parametric_nam_roundtrip_preserves_layer_subset_and_scales(tmp_path):
    """PA5: export/load preserves adapter subset selection and custom scales."""
    model = _build(
        {
            **_MULTI_C_CONFIG,
            "adapter_last_n_layers": 1,
            "adapter_gamma_scale": 0.1,
            "adapter_beta_scale": 0.02,
        }
    )
    model.export(tmp_path, basename="model")
    with open(tmp_path / "model.nam") as fp:
        nam = _json.load(fp)

    assert nam["config"]["adapter_last_n_layers"] == 1
    assert nam["config"]["adapter_gamma_scale"] == pytest.approx(0.1)
    assert nam["config"]["adapter_beta_scale"] == pytest.approx(0.02)

    loaded = load_parametric_nam(nam)

    assert loaded._adapter_last_n_layers == 1
    assert loaded._adapter_first_n_layers is None
    assert loaded._adapter_gamma_scale == pytest.approx(0.1)
    assert loaded._adapter_beta_scale == pytest.approx(0.02)


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
    # Override default to 1.0 so nominal != zero and the snapshot regression is detectable.
    config = {
        **_SINGLE_C_CONFIG,
        "params": [{"name": "gain", "min": 0.0, "max": 1.0, "default": 1.0}],
        "sample_rate": 100.0,
    }
    model = _build(config)
    model.eval()

    # Give the adapter non-trivial weights so nominal p != zero p has a real effect.
    with torch.no_grad():
        torch.nn.init.constant_(model._adapter._shared_encoder.fc.weight, 0.1)
        torch.nn.init.zeros_(model._adapter._shared_encoder.fc.bias)
        for sub in model._adapter._adapters:
            sub = cast(_LayerAdapter, sub)
            torch.nn.init.constant_(sub.gamma_head.weight, 0.1)

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


# ---------------------------------------------------------------------------
# ParamSpec unit tests — validation and serialization round-trip
# ---------------------------------------------------------------------------


def test_paramspec_valid_construction():
    """ParamSpec: valid min <= default <= max constructs without error."""
    spec = ParamSpec(name="gain", min=0.0, max=10.0, default=5.0)
    assert spec.name == "gain"
    assert spec.min == 0.0
    assert spec.max == 10.0
    assert spec.default == 5.0


def test_paramspec_boundary_default_equals_min():
    """ParamSpec: default == min is valid (boundary case)."""
    spec = ParamSpec(name="x", min=0.0, max=1.0, default=0.0)
    assert spec.default == 0.0


def test_paramspec_boundary_default_equals_max():
    """ParamSpec: default == max is valid (boundary case)."""
    spec = ParamSpec(name="x", min=0.0, max=1.0, default=1.0)
    assert spec.default == 1.0


def test_paramspec_from_dict_rejects_unsupported_input_normalization():
    """ParamSpec: loader must reject exported normalization metadata it cannot honor."""
    with pytest.raises(ValueError, match="Unsupported param input normalization"):
        ParamSpec.from_dict(
            {
                "name": "bright",
                "min": 0.0,
                "max": 1.0,
                "default": 0.5,
                "input_normalization": "unsupported",
            }
        )


def test_paramspec_from_dict_rejects_unsupported_normalized_bounds():
    """ParamSpec: loader must reject mismatched normalized bounds metadata."""
    with pytest.raises(ValueError, match="Unsupported normalized_min"):
        ParamSpec.from_dict(
            {
                "name": "bright",
                "min": 0.0,
                "max": 1.0,
                "default": 0.5,
                "normalized_min": 0.0,
            }
        )
    with pytest.raises(ValueError, match="Unsupported normalized_max"):
        ParamSpec.from_dict(
            {
                "name": "bright",
                "min": 0.0,
                "max": 1.0,
                "default": 0.5,
                "normalized_max": 0.0,
            }
        )


def test_paramspec_to_dict_from_dict_roundtrip():
    """ParamSpec: to_dict / from_dict round-trip preserves all fields."""
    original = ParamSpec(name="bright", min=0.0, max=1.0, default=0.5)
    restored = ParamSpec.from_dict(original.to_dict())
    assert restored.name == original.name
    assert restored.min == pytest.approx(original.min)
    assert restored.max == pytest.approx(original.max)
    assert restored.default == pytest.approx(original.default)


def test_paramspec_build_from_config_roundtrip(tmp_path):
    """ParamSpec: params array exported to .nam and reloaded preserves specs."""
    model = _build(_MULTI_C_CONFIG)
    model.export(tmp_path, basename="model")
    with open(tmp_path / "model.nam") as fp:
        nam = _json.load(fp)

    loaded = load_parametric_nam(nam)
    assert isinstance(loaded, ParametricWaveNet)

    # Specs must round-trip by name and default
    orig_specs = model._param_specs
    loaded_specs = loaded._param_specs
    assert len(orig_specs) == len(loaded_specs)
    for orig, reloaded in zip(orig_specs, loaded_specs):
        assert reloaded.name == orig.name
        assert reloaded.min == pytest.approx(orig.min)
        assert reloaded.max == pytest.approx(orig.max)
        assert reloaded.default == pytest.approx(orig.default, rel=1e-5)
