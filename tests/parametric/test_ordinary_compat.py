"""
Regression guard for the parametric adapter seam (RG1-RG6).

The seam adds optional adapter/p args to WaveNet, LayerArray, and _Layer forward
methods. Because all current callers omit these args, the seam must be invisible to
existing training, export, and load code paths. These tests enforce that contract so
future parametric work can't silently break ordinary A2 behavior.
"""

import json as _json
from pathlib import Path as _Path
from tempfile import TemporaryDirectory as _TemporaryDirectory

import pytest as _pytest
import torch as _torch

from nam import data as _data
from nam.models import _from_nam
from nam.models.wavenet import WaveNet as _WaveNet
from nam.train import lightning_module as _lightning_module

# ---------------------------------------------------------------------------
# Tiny WaveNet config reused across several tests (channels=2, 2 dilations).
# Matches the pattern in tests/test_nam/test_models/test_wavenet.py.
# ---------------------------------------------------------------------------

_TINY_CONFIG = {
    "layers_configs": [
        {
            "input_size": 1,
            "condition_size": 1,
            "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
            "channels": 2,
            "kernel_size": 2,
            "dilations": [1, 2],
            "activation": "Tanh",
        }
    ],
    "head_scale": 1.0,
}


def _build_wavenet() -> _WaveNet:
    """Return a tiny ordinary WaveNet (public wrapper)."""
    return _WaveNet.init_from_config(_TINY_CONFIG)


# ---------------------------------------------------------------------------
# RG1 — ordinary forward (no adapter/p) produces correct output shape.
# ---------------------------------------------------------------------------


def test_rg1_ordinary_forward_shape():
    """RG1: ordinary WaveNet forward with no adapter/p runs and has correct shape."""
    model = _build_wavenet()
    model.eval()
    batch_size, seq_len = 2, 128
    x = _torch.randn(batch_size, seq_len)
    with _torch.no_grad():
        y = model(x, pad_start=False)
    # Output should be (B, L-R+1); just assert it ran and is a tensor.
    assert isinstance(y, _torch.Tensor)
    assert y.shape[0] == batch_size


# ---------------------------------------------------------------------------
# RG1b — adapter=None, p=None gives bit-identical result to no-adapter call.
# ---------------------------------------------------------------------------


def test_rg1b_seam_is_inert():
    """RG1b: calling internal _net.forward with adapter=None, p=None is bit-identical
    to calling it without those kwargs at all."""
    model = _build_wavenet()
    model.eval()
    # Access the internal _WaveNet (not the public wrapper)
    internal = model._net
    x = _torch.randn(1, 1, 64)  # (B, C, L) for the internal forward
    with _torch.no_grad():
        out_no_kwargs = internal(x)
        out_with_none = internal(x, adapter=None, p=None)
    assert _torch.equal(out_no_kwargs, out_with_none), (
        "Seam is NOT inert: outputs differ when adapter=None, p=None vs no kwargs"
    )


# ---------------------------------------------------------------------------
# RG2 — public WaveNet wrapper still rejects unexpected kwargs.
# ---------------------------------------------------------------------------


def test_rg2_wrapper_rejects_kwargs():
    """RG2: the public WaveNet wrapper guard at _wavenet_wrapper.py:56-58 still
    raises ValueError on unexpected kwargs."""
    model = _build_wavenet()
    model.eval()
    x = _torch.randn(128)
    with _pytest.raises(ValueError, match="WaveNet does not support kwargs"):
        model(x, bogus_kwarg=True)


# ---------------------------------------------------------------------------
# RG3 — ordinary export yields architecture == "WaveNet" and schema unchanged.
# ---------------------------------------------------------------------------


def test_rg3_export_architecture():
    """RG3: exporting an ordinary WaveNet still produces architecture='WaveNet'
    with the expected top-level config keys."""
    model = _build_wavenet()
    with _TemporaryDirectory() as tmpdir:
        model.export(_Path(tmpdir), basename="model")
        with open(_Path(tmpdir, "model.nam"), "r") as fp:
            nam = _json.load(fp)

    assert nam["architecture"] == "WaveNet"
    # Config must contain the standard WaveNet keys (exported as "layers", not
    # "layers_configs" — the export schema serializes to the shorter form).
    config_keys = set(nam["config"].keys())
    for required_key in ("layers", "head_scale"):
        assert required_key in config_keys, (
            f"Expected config key '{required_key}' missing from export"
        )
    # Top-level NAM schema keys.
    for top_key in ("version", "architecture", "config", "weights"):
        assert top_key in nam, f"Expected top-level key '{top_key}' missing"


# ---------------------------------------------------------------------------
# RG4 — ordinary Dataset.__getitem__ returns a 2-tuple (x, y).
# ---------------------------------------------------------------------------


def test_rg4_dataset_getitem_2tuple():
    """RG4: nam.data.Dataset.__getitem__ returns a 2-tuple (x, y)."""
    n = 512
    x = _torch.randn(n)
    # Dataset validates abs(y).max() < 1.0, so keep y within (-1, 1).
    y = _torch.rand(n) * 0.9 - 0.45
    nx = 4
    ny = 16
    dataset = _data.Dataset(x, y, nx=nx, ny=ny, sample_rate=48_000.0)
    item = dataset[0]
    assert isinstance(item, tuple), "Dataset.__getitem__ must return a tuple"
    assert len(item) == 2, "Dataset.__getitem__ must return exactly 2 elements"
    xi, yi = item
    assert isinstance(xi, _torch.Tensor)
    assert isinstance(yi, _torch.Tensor)


# ---------------------------------------------------------------------------
# RG5 — ordinary LightningModule._shared_step on a 2-tuple batch yields scalar loss.
# ---------------------------------------------------------------------------


def test_rg5_shared_step_scalar_loss():
    """RG5: LightningModule._shared_step on a (x, y) batch yields a scalar loss.

    _shared_step calls model(x, pad_start=False), so the output is shorter than x
    by (receptive_field - 1) samples.  Match target length accordingly.
    """
    model = _build_wavenet()
    module = _lightning_module.LightningModule(
        model,
        loss_config=_lightning_module.LossConfig(mse_weight=1.0, mrstft_weight=None),
    )
    module.eval()
    batch_size = 2
    rf = model.receptive_field
    extra = 64
    seq_len = rf + extra  # x length
    target_len = seq_len - rf + 1  # output length when pad_start=False
    x = _torch.randn(batch_size, seq_len)
    y = _torch.randn(batch_size, target_len)
    batch = (x, y)
    with _torch.no_grad():
        preds, targets, loss_dict = module._shared_step(batch)
    assert isinstance(preds, _torch.Tensor)
    assert isinstance(targets, _torch.Tensor)
    # Compute the scalar loss the same way training_step does.
    # Use a loop with explicit None guards so pyright can narrow both weight and value.
    loss: _torch.Tensor = _torch.zeros(())
    for v in loss_dict.values():
        if v.weight is not None and v.weight > 0.0 and v.value is not None:
            loss = loss + v.weight * v.value
    assert isinstance(loss, _torch.Tensor)
    assert loss.ndim == 0, "Loss must be a scalar (0-dim tensor)"


# ---------------------------------------------------------------------------
# RG6 — ordinary .nam round-trip via init_from_nam still works.
# ---------------------------------------------------------------------------


def test_rg6_init_from_nam_roundtrip():
    """RG6: export → load via init_from_nam → re-export gives same architecture
    and config structure."""
    model = _build_wavenet()
    with _TemporaryDirectory() as tmpdir:
        model.export(_Path(tmpdir), basename="m1")
        with open(_Path(tmpdir, "m1.nam"), "r") as fp:
            nam1 = _json.load(fp)

        model2 = _from_nam.init_from_nam(nam1)

        model2.export(_Path(tmpdir), basename="m2")
        with open(_Path(tmpdir, "m2.nam"), "r") as fp:
            nam2 = _json.load(fp)

    assert nam2["architecture"] == "WaveNet"
    assert nam1["config"] == nam2["config"], (
        "Config changed across init_from_nam round-trip"
    )
    # Weights should round-trip exactly (float32 list).
    assert nam1["weights"] == nam2["weights"], (
        "Weights changed across init_from_nam round-trip"
    )
