"""
PA1, PA1c, PA1d, PA2 — ParametricWaveNet model tests.

PA1c is the MERGE GATE: zero-init adapter must give bit-exact parity with an
ordinary WaveNet whose weights were imported into the inner net.
"""

from typing import cast

import pytest
import torch

from nam.models.parametric import ParametricWaveNet
from nam.models.parametric._model import _ChannelAdapter
from nam.models.wavenet._wavenet import WaveNet as _InnerWaveNet

# ---------------------------------------------------------------------------
# Configs — use the new self-describing params array schema
# ---------------------------------------------------------------------------

# Single-channel-size config (channels=4, Tanh activation).
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
}

# Multi-channel-size config: two layer arrays with DIFFERENT channel counts (8 and 4).
# The second layer array's input_size must equal the first's channels (8) because the
# first layer array outputs 8-channel tensors.  condition_size=1 throughout (mono audio).
# This exercises the heterogeneous _ChannelAdapter dispatch path (PA1c gate).
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
            "input_size": 8,  # receives output of first layer array (channels=8)
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
}

# Helpers for tests that need param_dim as an integer
_SINGLE_C_PARAM_DIM = 1
_MULTI_C_PARAM_DIM = 2


def _build_parametric(config):
    return ParametricWaveNet.init_from_config(config)


# ---------------------------------------------------------------------------
# PA1 — forward produces correct audio output shape
# ---------------------------------------------------------------------------


def test_pa1_forward_shape_single_c():
    """PA1: ParametricWaveNet.forward(x, params=(B,P)) returns correct audio shape."""
    model = _build_parametric(_SINGLE_C_CONFIG)
    model.eval()
    B, P = 3, 1
    seq_len = model.receptive_field + 64
    x = torch.randn(B, seq_len)
    params = torch.randn(B, P)
    with torch.no_grad():
        y = model(x, params, pad_start=False)
    assert isinstance(y, torch.Tensor)
    assert y.shape[0] == B
    # Output length = seq_len - receptive_field + 1
    assert y.shape[1] == seq_len - model.receptive_field + 1


def test_pa1_forward_shape_multi_c():
    """PA1: ParametricWaveNet.forward also works for multi-layer-array config."""
    model = _build_parametric(_MULTI_C_CONFIG)
    model.eval()
    B, P = 2, 2
    seq_len = model.receptive_field + 32
    x = torch.randn(B, seq_len)
    params = torch.randn(B, P)
    with torch.no_grad():
        y = model(x, params, pad_start=False)
    assert y.shape[0] == B
    assert y.shape[1] == seq_len - model.receptive_field + 1


# ---------------------------------------------------------------------------
# PA1c — MERGE GATE: zero-adapter parity with ordinary WaveNet
# ---------------------------------------------------------------------------


def test_pa1c_zero_adapter_parity_with_inner_wavenet():
    """
    PA1c (MERGE GATE): import ordinary inner WaveNet weights into ParametricWaveNet,
    leave adapter at zero-init, and assert outputs are IDENTICAL at float32 precision
    across multiple inputs AND multiple parameter vectors (including 1-D constant-param).

    Exercises the heterogeneous-channel path (multi-C config with 8 and 4 channels).
    """
    config = _MULTI_C_CONFIG

    # Build ordinary inner WaveNet with fixed random weights (strip parametric keys)
    inner_cfg = {k: v for k, v in config.items() if k != "params"}
    ordinary = _InnerWaveNet.init_from_config(inner_cfg)
    ordinary.eval()

    # Build ParametricWaveNet and copy inner weights from ordinary
    parametric = _build_parametric(config)
    parametric.eval()

    # Import the exact weights of `ordinary` into parametric's inner net.
    # export_weights() returns a 1-D numpy array in import order.
    weights = ordinary.export_weights()
    parametric._net.import_weights(torch.tensor(weights))

    P = parametric._param_dim

    # Adapter must still be zero-init (we haven't trained or set any adapter weights)
    for key, sub_adapter in parametric._adapter._adapters.items():
        sa = cast(_ChannelAdapter, sub_adapter)
        gamma_out = sa.gamma_map(torch.ones(1, P))
        beta_out = sa.beta_map(torch.ones(1, P))
        assert torch.all(gamma_out == 0), f"gamma_map not zero at C={key}"
        assert torch.all(beta_out == 0), f"beta_map not zero at C={key}"

    rf = parametric.receptive_field
    torch.manual_seed(42)

    # Multiple inputs, multiple param vectors (including 1-D constant-param case)
    test_cases = [
        # (x shape, params)
        (torch.randn(1, rf + 128), torch.zeros(1, P)),
        (torch.randn(2, rf + 64), torch.randn(2, P)),
        (torch.randn(3, rf + 48), torch.ones(3, P)),
        # 1-D constant-param case: same p broadcast to all batch items
        (torch.randn(2, rf + 32), torch.zeros(P)),
    ]

    for x, p in test_cases:
        with torch.no_grad():
            # Ordinary WaveNet: needs (B, 1, L) input internally
            x3 = x[None] if x.ndim == 1 else x
            x3d = x3[:, None, :]  # (B, 1, L)
            y_ordinary = ordinary(x3d)  # (B, 1, L')
            assert y_ordinary.shape[1] == 1
            y_ordinary = y_ordinary[:, 0, :]  # (B, L')

            y_parametric = parametric(x, p, pad_start=False)  # (B, L')

        assert y_ordinary.shape == y_parametric.shape, (
            f"Shape mismatch: ordinary={y_ordinary.shape}, parametric={y_parametric.shape}"
        )
        assert torch.equal(y_ordinary, y_parametric), (
            f"PA1c FAILED: outputs differ at x.shape={tuple(x.shape)}, p.shape={tuple(p.shape)}.\n"
            f"Max abs diff: {(y_ordinary - y_parametric).abs().max().item()}"
        )


# ---------------------------------------------------------------------------
# PA1d — gamma and beta maps output zero for arbitrary p at construction
# ---------------------------------------------------------------------------


def test_pa1d_zero_init_all_channel_sizes():
    """
    PA1d: at construction (before any training), gamma_map(p)==0 and beta_map(p)==0
    for arbitrary p, for EVERY channel-size sub-adapter in the ModuleDict.
    """
    model = _build_parametric(_MULTI_C_CONFIG)
    model.eval()
    P = model._param_dim

    # Try several arbitrary p vectors
    test_ps = [
        torch.zeros(1, P),
        torch.ones(1, P),
        torch.randn(1, P),
        torch.full((1, P), -3.7),
    ]

    for key, sub_adapter in model._adapter._adapters.items():
        sa = cast(_ChannelAdapter, sub_adapter)
        for p in test_ps:
            with torch.no_grad():
                gamma_out = sa.gamma_map(p)
                beta_out = sa.beta_map(p)
            assert torch.all(gamma_out == 0), (
                f"PA1d FAILED: gamma_map not zero for C={key}, p={p}"
            )
            assert torch.all(beta_out == 0), (
                f"PA1d FAILED: beta_map not zero for C={key}, p={p}"
            )


# ---------------------------------------------------------------------------
# PA2 — mismatched param_dim raises a clear error
# ---------------------------------------------------------------------------


def test_pa2_params_dim_mismatch_raises():
    """PA2: passing params with wrong dim raises ValueError with a clear message."""
    model = _build_parametric(_SINGLE_C_CONFIG)
    model.eval()
    P_configured = _SINGLE_C_PARAM_DIM  # 1
    B = 2
    seq_len = model.receptive_field + 32

    x = torch.randn(B, seq_len)
    # Wrong dim: configured P=1, passing P=3
    bad_params = torch.randn(B, P_configured + 2)

    with pytest.raises(ValueError, match="param_dim"):
        model(x, bad_params, pad_start=False)


def test_pa2_params_wrong_shape_raises():
    """PA2: 3-D params tensor raises ValueError."""
    model = _build_parametric(_SINGLE_C_CONFIG)
    x = torch.randn(2, model.receptive_field + 16)
    bad_params = torch.randn(2, 3, 1)  # 3-D, not supported
    with pytest.raises(ValueError):
        model(x, bad_params, pad_start=False)
