"""
C2.1 dataset tests: PA7, PA11, PA12.

PA7  — ParametricDataset.__getitem__ returns 3-tuple (params, x, y) with correct shapes
        and DataLoader collation produces (B,P), (B,NX+NY-1), (B,NY).
PA11 — malformed param config (wrong number of params) raises ValueError with clear message.
PA12 — missing WAV path raises a clear error.
"""

from pathlib import Path as _Path

import numpy as _np
import pytest as _pytest
import torch as _torch
from torch.utils.data import DataLoader as _DataLoader

from nam.data import np_to_wav as _np_to_wav
from nam.models.parametric import ParametricConcatDataset, ParametricDataset  # import triggers registrations

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RATE = 48_000
_NX = 8
_NY = 16

# Require enough samples that Dataset validates without issues:
# need at least NX+NY-1 samples after pre-silence region, and the default
# pre-silence check needs _DEFAULT_REQUIRE_INPUT_PRE_SILENCE=0.4 s of silence
# at the start.  Use zero input for the pre-silence window, then random audio.
_PRE_SILENCE_SAMPLES = int(0.5 * _RATE)  # 0.5 s of zeros to satisfy 0.4 s check
_AUDIO_SAMPLES = _NX + _NY + 64         # additional audio after the silence


def _write_wav_pair(tmp_path: _Path, seed: int = 42):
    """Write a matched mono WAV pair (x, y) to tmp_path; return (x_path, y_path)."""
    rng = _np.random.default_rng(seed)
    n = _PRE_SILENCE_SAMPLES + _AUDIO_SAMPLES

    # Pre-silence so Dataset's require_input_pre_silence check passes.
    x_data = _np.concatenate([
        _np.zeros(_PRE_SILENCE_SAMPLES),
        rng.uniform(-0.4, 0.4, _AUDIO_SAMPLES),
    ])
    # y must have abs(max) < 1.0; keep it in (-0.5, 0.5).
    y_data = rng.uniform(-0.4, 0.4, n)

    x_path = tmp_path / "x.wav"
    y_path = tmp_path / "y.wav"
    _np_to_wav(x_data, x_path, rate=_RATE)
    _np_to_wav(y_data, y_path, rate=_RATE)
    return str(x_path), str(y_path)


def _base_config(x_path: str, y_path: str) -> dict:
    """Minimal Dataset-compatible config (no parametric keys yet)."""
    return {
        "x_path": x_path,
        "y_path": y_path,
        "nx": _NX,
        "ny": _NY,
        "sample_rate": _RATE,
    }


def _parametric_config(x_path: str, y_path: str, param_names, params) -> dict:
    cfg = _base_config(x_path, y_path)
    cfg["param_names"] = param_names
    cfg["params"] = params
    return cfg


# ---------------------------------------------------------------------------
# PA7 — __getitem__ 3-tuple and DataLoader collation
# ---------------------------------------------------------------------------


def test_pa7_single_param_getitem_shape(tmp_path):
    """PA7 (P=1): getitem returns (params, x, y) with shapes (1,), (NX+NY-1,), (NY,)."""
    x_path, y_path = _write_wav_pair(tmp_path)
    cfg = _parametric_config(x_path, y_path, ["gain"], [0.7])
    ds = ParametricDataset.init_from_config(cfg)

    assert len(ds) > 0, "Dataset must have at least one item"
    item = ds[0]
    assert isinstance(item, tuple), "__getitem__ must return a tuple"
    assert len(item) == 3, "__getitem__ must return exactly 3 elements"
    params, x, y = item

    assert isinstance(params, _torch.Tensor), "params must be a Tensor"
    assert params.dtype == _torch.float32, "params must be float32"
    assert params.shape == (1,), f"Expected params.shape=(1,), got {params.shape}"

    assert isinstance(x, _torch.Tensor), "x must be a Tensor"
    assert x.shape == (_NX + _NY - 1,), f"Expected x.shape=({_NX + _NY - 1},), got {x.shape}"

    assert isinstance(y, _torch.Tensor), "y must be a Tensor"
    assert y.shape == (_NY,), f"Expected y.shape=({_NY},), got {y.shape}"


def test_pa7_multi_param_getitem_shape(tmp_path):
    """PA7 (P=3): getitem returns (params, x, y) with shapes (3,), (NX+NY-1,), (NY,)."""
    x_path, y_path = _write_wav_pair(tmp_path, seed=1)
    P = 3
    cfg = _parametric_config(
        x_path, y_path, ["gain", "treble", "bass"], [0.3, 0.6, 0.9]
    )
    ds = ParametricDataset.init_from_config(cfg)

    params, x, y = ds[0]

    assert params.shape == (P,), f"Expected params.shape=({P},), got {params.shape}"
    assert params.dtype == _torch.float32
    assert x.shape == (_NX + _NY - 1,)
    assert y.shape == (_NY,)


def test_pa7_params_values_preserved(tmp_path):
    """PA7: param values are stored exactly and returned identically on every item."""
    x_path, y_path = _write_wav_pair(tmp_path, seed=2)
    param_values = [0.25, 0.75]
    cfg = _parametric_config(x_path, y_path, ["a", "b"], param_values)
    ds = ParametricDataset.init_from_config(cfg)

    for i in range(min(len(ds), 3)):
        params, _, _ = ds[i]
        expected = _torch.tensor(param_values, dtype=_torch.float32)
        assert _torch.allclose(params, expected), (
            f"params at idx={i} differ from config values"
        )


def test_pa7_dataloader_collation(tmp_path):
    """PA7: DataLoader collates a ParametricDataset into (B,P), (B,NX+NY-1), (B,NY)."""
    x_path, y_path = _write_wav_pair(tmp_path, seed=3)
    P = 2
    cfg = _parametric_config(x_path, y_path, ["a", "b"], [0.1, 0.9])
    ds = ParametricDataset.init_from_config(cfg)

    B = min(4, len(ds))
    loader = _DataLoader(ds, batch_size=B, shuffle=False)
    batch = next(iter(loader))

    assert isinstance(batch, (list, tuple)), "Batch must be a list or tuple"
    assert len(batch) == 3, f"Expected 3-element batch, got {len(batch)}"
    b_params, b_x, b_y = batch

    assert b_params.shape == (B, P), f"Expected ({B}, {P}), got {b_params.shape}"
    assert b_x.shape == (B, _NX + _NY - 1), (
        f"Expected ({B}, {_NX + _NY - 1}), got {b_x.shape}"
    )
    assert b_y.shape == (B, _NY), f"Expected ({B}, {_NY}), got {b_y.shape}"
    assert b_params.dtype == _torch.float32


# ---------------------------------------------------------------------------
# PA11 — wrong number of params raises ValueError with a clear message
# ---------------------------------------------------------------------------


def test_pa11_wrong_params_count_vs_param_names(tmp_path):
    """PA11: len(params) != len(param_names) raises ValueError naming the field."""
    x_path, y_path = _write_wav_pair(tmp_path, seed=4)
    cfg = _parametric_config(x_path, y_path, ["gain", "treble"], [0.5])  # 2 names, 1 value
    with _pytest.raises(ValueError, match="params"):
        ParametricDataset.init_from_config(cfg)


def test_pa11_param_dim_mismatch(tmp_path):
    """PA11: param_dim that doesn't match len(param_names) raises ValueError."""
    x_path, y_path = _write_wav_pair(tmp_path, seed=5)
    cfg = _parametric_config(x_path, y_path, ["gain"], [0.5])
    cfg["param_dim"] = 3  # claimed 3 but only 1 name/value
    with _pytest.raises(ValueError, match="param_dim"):
        ParametricDataset.init_from_config(cfg)


def test_pa11_direct_constructor_mismatch():
    """PA11: direct constructor also validates len(params)==len(param_names)."""
    # Use a tiny in-memory Dataset to avoid WAV I/O
    n = 128
    x = _torch.zeros(n)
    y = _torch.rand(n) * 0.4
    inner = __import__("nam.data", fromlist=["Dataset"]).Dataset(
        x, y, nx=4, ny=8, sample_rate=48_000.0, require_input_pre_silence=None
    )
    with _pytest.raises(ValueError, match="params"):
        ParametricDataset(inner, param_names=["a", "b"], params=[0.5])  # 2 names, 1 value


# ---------------------------------------------------------------------------
# PA12 — missing WAV path surfaces a clear error
# ---------------------------------------------------------------------------


def test_pa12_missing_x_wav(tmp_path):
    """PA12: missing x_path raises a clear error (FileNotFoundError or DataError)."""
    _, y_path = _write_wav_pair(tmp_path, seed=6)
    cfg = _parametric_config(
        str(tmp_path / "does_not_exist_x.wav"),
        y_path,
        ["gain"],
        [0.5],
    )
    # Dataset.parse_config → wav_to_tensor → wavio.read raises FileNotFoundError
    with _pytest.raises(Exception) as exc_info:
        ParametricDataset.init_from_config(cfg)
    # The error must be informative — either a FileNotFoundError or contain the path
    err_text = str(exc_info.value)
    assert "does_not_exist" in err_text or exc_info.type.__name__ in (
        "FileNotFoundError",
        "DataError",
        "OSError",
        "Error",
    ), f"Expected a clear file-not-found error, got {exc_info.type}: {err_text}"


def test_pa12_missing_y_wav(tmp_path):
    """PA12: missing y_path raises a clear error."""
    x_path, _ = _write_wav_pair(tmp_path, seed=7)
    cfg = _parametric_config(
        x_path,
        str(tmp_path / "does_not_exist_y.wav"),
        ["gain"],
        [0.5],
    )
    with _pytest.raises(Exception) as exc_info:
        ParametricDataset.init_from_config(cfg)
    err_text = str(exc_info.value)
    assert "does_not_exist" in err_text or exc_info.type.__name__ in (
        "FileNotFoundError",
        "DataError",
        "OSError",
        "Error",
    ), f"Expected a clear file-not-found error, got {exc_info.type}: {err_text}"


# ---------------------------------------------------------------------------
# EC8 — ParametricConcatDataset with matching param_dim succeeds
# ---------------------------------------------------------------------------


def _make_parametric_dataset(tmp_path: _Path, seed: int, params, param_names=None):
    """Helper: build one ParametricDataset from a WAV pair written to a seed subdir."""
    sub = tmp_path / str(seed)
    sub.mkdir(exist_ok=True)
    x_path, y_path = _write_wav_pair(sub, seed=seed)
    if param_names is None:
        param_names = [f"p{i}" for i in range(len(params))]
    cfg = _parametric_config(x_path, y_path, param_names, params)
    return ParametricDataset.init_from_config(cfg)


def test_ec8_concat_two_datasets_same_param_dim(tmp_path):
    """EC8: concatenating two datasets with the same P produces correct length and shapes."""
    P = 2
    ds1 = _make_parametric_dataset(tmp_path, seed=10, params=[0.1, 0.2])
    ds2 = _make_parametric_dataset(tmp_path, seed=11, params=[0.5, 0.6])

    concat = ParametricConcatDataset([ds1, ds2])

    assert len(concat) == len(ds1) + len(ds2), (
        f"Expected {len(ds1) + len(ds2)}, got {len(concat)}"
    )
    assert concat.param_dim == P

    params, x, y = concat[0]
    assert isinstance(params, _torch.Tensor)
    assert params.shape == (P,)
    assert x.shape == (_NX + _NY - 1,)
    assert y.shape == (_NY,)


def test_ec8_concat_item_routing_across_boundary(tmp_path):
    """EC8 (boundary): item at ds1 boundary comes from ds1; next item comes from ds2."""
    ds1 = _make_parametric_dataset(tmp_path, seed=12, params=[0.1, 0.2], param_names=["a", "b"])
    ds2 = _make_parametric_dataset(tmp_path, seed=13, params=[0.9, 0.8], param_names=["a", "b"])

    concat = ParametricConcatDataset([ds1, ds2])

    # Last item of ds1
    params_last_ds1, _, _ = concat[len(ds1) - 1]
    assert _torch.allclose(params_last_ds1, _torch.tensor([0.1, 0.2], dtype=_torch.float32)), (
        "Last item before boundary should have ds1 params"
    )

    # First item of ds2
    params_first_ds2, _, _ = concat[len(ds1)]
    assert _torch.allclose(params_first_ds2, _torch.tensor([0.9, 0.8], dtype=_torch.float32)), (
        "First item after boundary should have ds2 params"
    )


# ---------------------------------------------------------------------------
# EC9 — ParametricConcatDataset with mismatched param_dim raises ValueError
# ---------------------------------------------------------------------------


def test_ec9_concat_mismatched_param_dim_raises(tmp_path):
    """EC9: mixing datasets with different param_dim must raise ValueError mentioning param_dim."""
    ds1 = _make_parametric_dataset(tmp_path, seed=14, params=[0.5])         # P=1
    ds2 = _make_parametric_dataset(tmp_path, seed=15, params=[0.3, 0.7])    # P=2

    with _pytest.raises(ValueError, match="param_dim"):
        ParametricConcatDataset([ds1, ds2])


def test_ec9_concat_mismatched_param_names_order_raises(tmp_path):
    """EC9: same P but different param_names order must raise ValueError mentioning param_names."""
    ds1 = _make_parametric_dataset(
        tmp_path,
        seed=18,
        params=[0.1, 0.2],
        param_names=["gain", "treble"],
    )
    ds2 = _make_parametric_dataset(
        tmp_path,
        seed=19,
        params=[0.3, 0.4],
        param_names=["treble", "gain"],
    )

    with _pytest.raises(ValueError, match="param_names"):
        ParametricConcatDataset([ds1, ds2])


# ---------------------------------------------------------------------------
# IT2 — DataLoader collation of ParametricConcatDataset
# ---------------------------------------------------------------------------


def test_it2_dataloader_collation_from_concat(tmp_path):
    """IT2: DataLoader over ParametricConcatDataset produces (B,P), (B,NX+NY-1), (B,NY)."""
    P = 3
    ds1 = _make_parametric_dataset(tmp_path, seed=16, params=[0.1, 0.2, 0.3])
    ds2 = _make_parametric_dataset(tmp_path, seed=17, params=[0.4, 0.5, 0.6])
    concat = ParametricConcatDataset([ds1, ds2])

    B = min(4, len(concat))
    loader = _DataLoader(concat, batch_size=B, shuffle=False)
    batch = next(iter(loader))

    assert isinstance(batch, (list, tuple))
    assert len(batch) == 3
    b_params, b_x, b_y = batch

    assert b_params.shape == (B, P), f"Expected ({B},{P}), got {b_params.shape}"
    assert b_x.shape == (B, _NX + _NY - 1), f"Expected ({B},{_NX + _NY - 1}), got {b_x.shape}"
    assert b_y.shape == (B, _NY), f"Expected ({B},{_NY}), got {b_y.shape}"
    assert b_params.dtype == _torch.float32


# ---------------------------------------------------------------------------
# PA11 extension — empty dataset list raises ValueError
# ---------------------------------------------------------------------------


def test_pa11_concat_empty_list_raises():
    """PA11 (concat): constructing ParametricConcatDataset with empty list raises ValueError."""
    with _pytest.raises(ValueError):
        ParametricConcatDataset([])
