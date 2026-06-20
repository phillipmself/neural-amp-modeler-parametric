"""
C4.2 integration tests: IT1, EC1, EC2.

IT1 — multi-capture end-to-end (P=2): synthetic WAV pairs at 3 parameter settings,
      ParametricConcatDataset, one training_step, export, reload, forward.
EC1 — edge case P=1: single drive parameter, full round-trip.
EC2 — edge case P=8: 8 parameters, full round-trip.

These tests call training_step() directly rather than spinning up a Trainer, keeping
total runtime well under 30 seconds while still covering the full data→train→export→load
pipeline.
"""

import json as _json
from pathlib import Path as _Path
from typing import List as _List

import numpy as _np
import pytest
import torch

from nam.data import np_to_wav as _np_to_wav
from nam.models.parametric import (
    ParametricConcatDataset,
    ParametricDataset,
    ParametricWaveNet,
    load_parametric_nam,
)
from nam.train.lightning_module import LossConfig
from nam.train.parametric import ParametricLightningModule

# ---------------------------------------------------------------------------
# Shared constants — match test_dataset.py sizes to reuse its WAV-writing logic
# ---------------------------------------------------------------------------

_RATE = 48_000
_NX = 8
_NY = 16
_PRE_SILENCE_SAMPLES = int(0.5 * _RATE)  # 0.5 s zeros satisfies 0.4 s require_pre_silence
_AUDIO_SAMPLES = _NX + _NY + 64

_LOSS_CONFIG = LossConfig(mse_weight=1.0, mrstft_weight=None)


# ---------------------------------------------------------------------------
# Helpers — reuse the same WAV-pair writing pattern as test_dataset.py
# ---------------------------------------------------------------------------


def _write_wav_pair(directory: _Path, name: str, seed: int = 42):
    """Write a matched mono WAV pair to `directory/<name>_{x,y}.wav`; return (x_path, y_path)."""
    rng = _np.random.default_rng(seed)
    n = _PRE_SILENCE_SAMPLES + _AUDIO_SAMPLES

    x_data = _np.concatenate(
        [_np.zeros(_PRE_SILENCE_SAMPLES), rng.uniform(-0.4, 0.4, _AUDIO_SAMPLES)]
    )
    y_data = rng.uniform(-0.4, 0.4, n)

    x_path = directory / f"{name}_x.wav"
    y_path = directory / f"{name}_y.wav"
    _np_to_wav(x_data, x_path, rate=_RATE)
    _np_to_wav(y_data, y_path, rate=_RATE)
    return str(x_path), str(y_path)


def _make_parametric_dataset(
    tmp_path: _Path,
    name: str,
    param_names: _List[str],
    params: _List[float],
    seed: int = 0,
) -> ParametricDataset:
    """Create one ParametricDataset from a freshly written WAV pair."""
    x_path, y_path = _write_wav_pair(tmp_path, name, seed=seed)
    config = {
        "x_path": x_path,
        "y_path": y_path,
        "nx": _NX,
        "ny": _NY,
        "sample_rate": _RATE,
        "param_names": param_names,
        "params": params,
    }
    return ParametricDataset.init_from_config(config)


def _make_model(param_names: _List[str], nominal_params: _List[float]) -> ParametricWaveNet:
    """Build the smallest valid ParametricWaveNet for the given param schema.

    Uses symmetric [0, 1] min/max ranges and the supplied nominal as default.
    The model normalizes raw parameter values from these bounds into [-1, 1].
    """
    config = {
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
        "params": [
            {"name": n, "min": 0.0, "max": 1.0, "default": float(v)}
            for n, v in zip(param_names, nominal_params)
        ],
        "sample_rate": float(_RATE),
    }
    return ParametricWaveNet.init_from_config(config)


def _make_batch_from_concat(
    concat_ds: ParametricConcatDataset, model: ParametricWaveNet, batch_size: int = 2
):
    """
    Build a synthetic (params, x, y) batch whose x/y lengths are consistent with the
    model's receptive field, so MSE(preds, y) is shape-compatible.

    Dataset items have x of length NX+NY-1 which may not align with model.receptive_field.
    Rather than requiring that alignment, we draw params from the dataset (so real capture
    param values are used) and generate x/y tensors sized for the model.

    x: (B, RF + extra) — extra samples give output of length `extra + 1`
    y: (B, extra + 1)  — matches what model.forward returns
    """
    rf = model.receptive_field
    extra = 32  # small but nonzero output length

    P_list = []
    for i in range(batch_size):
        p, _x, _y = concat_ds[i % len(concat_ds)]
        P_list.append(p)

    params = torch.stack(P_list)
    x = torch.randn(batch_size, rf + extra)
    y = torch.randn(batch_size, extra + 1)
    return params, x, y


def _export_reload(model: ParametricWaveNet, tmp_path: _Path, basename: str) -> ParametricWaveNet:
    """Export model and reload it via load_parametric_nam; return the reloaded instance."""
    model.export(tmp_path, basename=basename)
    with open(tmp_path / f"{basename}.nam") as fp:
        nam_dict = _json.load(fp)
    return load_parametric_nam(nam_dict)


# ---------------------------------------------------------------------------
# IT1 — multi-capture end-to-end (P=2, 3 captures)
# ---------------------------------------------------------------------------


def test_IT1_multi_capture_end_to_end(tmp_path):
    """IT1: parametric multi-capture end-to-end — data → train step → export → reload → forward.

    Three captures at different (gain, treble) settings are concatenated into a
    ParametricConcatDataset. One training_step is called directly (no Trainer).
    The model is exported and reloaded; the reloaded model produces the correct output shape.
    """
    param_names = ["gain", "treble"]
    captures = [
        ([0.2, 0.1], 10),
        ([0.5, 0.5], 20),
        ([0.8, 0.9], 30),
    ]

    datasets = [
        _make_parametric_dataset(
            tmp_path, f"cap{i}", param_names, params, seed=seed
        )
        for i, (params, seed) in enumerate(captures)
    ]

    concat_ds = ParametricConcatDataset(datasets)
    assert concat_ds.param_dim == 2
    assert len(concat_ds) == sum(len(d) for d in datasets)

    # Verify a single item is a well-formed 3-tuple
    p0, x0, y0 = concat_ds[0]
    assert p0.shape == (2,), f"params shape wrong: {p0.shape}"
    assert x0.shape == (_NX + _NY - 1,)
    assert y0.shape == (_NY,)

    # Build model and training module
    model = _make_model(param_names, nominal_params=[0.5, 0.5])
    module = ParametricLightningModule(model, loss_config=_LOSS_CONFIG)

    # Directly call training_step (avoids Trainer overhead while exercising the loss path)
    batch = _make_batch_from_concat(concat_ds, model, batch_size=2)
    loss = module.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor), "training_step must return a Tensor loss"
    loss_t: torch.Tensor = loss
    assert loss_t.ndim == 0, "training_step loss must be scalar"
    assert torch.isfinite(loss_t), "loss must be finite"

    # Export and reload round-trip
    reloaded = _export_reload(model, tmp_path, basename="it1")
    assert isinstance(reloaded, ParametricWaveNet)

    # Forward pass on reloaded model
    reloaded.eval()
    rf = reloaded.receptive_field
    x_in = torch.randn(1, rf + 64)
    params_in = torch.tensor([[0.5, 0.3]])
    with torch.no_grad():
        y_out = reloaded(x_in, params=params_in, pad_start=False)
    assert y_out.shape[0] == 1, "batch dim must be 1"
    assert y_out.shape[1] > 0, "output length must be positive"


# ---------------------------------------------------------------------------
# EC1 — edge case: P=1 (single scalar parameter)
# ---------------------------------------------------------------------------


def test_EC1_p1_single_param(tmp_path):
    """EC1: P=1 edge case — drive-only model.

    Verifies that a 1-D params tensor flows correctly through dataset, training step,
    and export/reload round-trip. Also exercises the constant-param broadcast shape
    (B, 1) for a batch forward pass.
    """
    param_names = ["drive"]
    nominal_params = [0.5]

    ds = _make_parametric_dataset(tmp_path, "ec1", param_names, [0.5], seed=7)

    # Item shape
    p, x, y = ds[0]
    assert p.shape == (1,), f"P=1 params shape wrong: {p.shape}"

    concat_ds = ParametricConcatDataset([ds])
    assert concat_ds.param_dim == 1

    model = _make_model(param_names, nominal_params)
    module = ParametricLightningModule(model, loss_config=_LOSS_CONFIG)

    batch = _make_batch_from_concat(concat_ds, model, batch_size=2)
    params_b, x_b, y_b = batch
    assert params_b.shape == (2, 1), f"collated params shape wrong: {params_b.shape}"

    loss = module.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    loss_t: torch.Tensor = loss
    assert loss_t.ndim == 0 and torch.isfinite(loss_t)

    # Export + reload
    reloaded = _export_reload(model, tmp_path, basename="ec1")
    assert isinstance(reloaded, ParametricWaveNet)
    reloaded.eval()

    # Constant-param broadcast: expand scalar param across a batch
    B = 3
    rf = reloaded.receptive_field
    x_in = torch.randn(B, rf + 32)
    # Expand a single scalar param value across the batch (P=1 constant-param path)
    params_in = torch.tensor(0.5).expand(B, 1)
    with torch.no_grad():
        y_out = reloaded(x_in, params=params_in, pad_start=False)
    assert y_out.shape[0] == B
    assert y_out.shape[1] > 0


# ---------------------------------------------------------------------------
# EC2 — edge case: P=8 (many parameters)
# ---------------------------------------------------------------------------


def test_EC2_p8_multi_param(tmp_path):
    """EC2: P=8 edge case — 8 simultaneous parameters.

    Verifies that the full stack (dataset, concat, training step, export, reload, forward)
    handles a wide parameter vector without shape errors.
    """
    param_names = [f"p{i}" for i in range(8)]
    nominal_params = [0.5] * 8

    # Two captures at different 8-param settings
    captures = [
        [0.1 * (i + 1) for i in range(8)],
        [0.9 - 0.1 * i for i in range(8)],
    ]

    datasets = [
        _make_parametric_dataset(tmp_path, f"ec2_cap{i}", param_names, params, seed=100 + i)
        for i, params in enumerate(captures)
    ]

    concat_ds = ParametricConcatDataset(datasets)
    assert concat_ds.param_dim == 8

    # Item shape
    p, x, y = concat_ds[0]
    assert p.shape == (8,), f"P=8 params shape wrong: {p.shape}"

    model = _make_model(param_names, nominal_params)
    module = ParametricLightningModule(model, loss_config=_LOSS_CONFIG)

    batch = _make_batch_from_concat(concat_ds, model, batch_size=2)
    params_b, x_b, y_b = batch
    assert params_b.shape == (2, 8), f"collated params shape wrong: {params_b.shape}"

    loss = module.training_step(batch, 0)
    assert isinstance(loss, torch.Tensor)
    loss_t: torch.Tensor = loss
    assert loss_t.ndim == 0 and torch.isfinite(loss_t)

    # Export + reload
    reloaded = _export_reload(model, tmp_path, basename="ec2")
    assert isinstance(reloaded, ParametricWaveNet)
    reloaded.eval()

    rf = reloaded.receptive_field
    x_in = torch.randn(1, rf + 32)
    params_in = torch.tensor([nominal_params])
    with torch.no_grad():
        y_out = reloaded(x_in, params=params_in, pad_start=False)
    assert y_out.shape[0] == 1
    assert y_out.shape[1] > 0
