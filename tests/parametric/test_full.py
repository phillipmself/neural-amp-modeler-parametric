"""
N1 training entrypoint tests.

Proves that full.main() selects ParametricLightningModule when the model config is
"ParametricWaveNet", and completes at least one real training step on a (params, x, y)
batch without the 3-tuple mis-route caught by PA8b.

PA8  (in test_train.py) — ParametricLightningModule._shared_step routes correctly.
PA8b (in test_train.py) — base LightningModule mis-routes (confirmed, load-bearing).
TF1  — full.main() with "ParametricWaveNet" config selects ParametricLightningModule
        and completes a training run; model.nam is exported.
"""

import numpy as np
import pytest

from nam.data import np_to_wav as _np_to_wav
from nam.train import core as _core
from nam.models.parametric import ParametricConcatDataset  # triggers registrations
from nam.train import full as _full


_NUM_SAMPLES = 256
_NUM_VALIDATION_SAMPLES = 64
_NY = 8
_RATE = 48_000


def _write_wav_pair(tmp_path):
    t = np.arange(_NUM_SAMPLES, dtype=np.float64) / _RATE
    x = 0.10 * np.sin(2.0 * np.pi * 220.0 * t)
    y = 0.50 * x + 0.02 * np.sin(2.0 * np.pi * 440.0 * t)
    x_path = tmp_path / "input.wav"
    y_path = tmp_path / "output.wav"
    _np_to_wav(x, x_path, rate=_RATE)
    _np_to_wav(y, y_path, rate=_RATE)
    return x_path, y_path


def _data_config(x_path, y_path):
    return {
        "type": "parametric",
        "common": {
            "x_path": str(x_path),
            "y_path": str(y_path),
            "delay": 0,
            "require_input_pre_silence": None,
            "param_names": ["gain"],
            "params": [0.5],
        },
        "train": {
            "stop_samples": -_NUM_VALIDATION_SAMPLES,
            "ny": _NY,
        },
        "validation": {
            "start_samples": -_NUM_VALIDATION_SAMPLES,
            "ny": None,
        },
    }


def _net_config():
    return {
        "layers_configs": [
            {
                "input_size": 1,
                "condition_size": 1,
                "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
                "channels": 4,
                "kernel_size": 2,
                "dilations": [1],
                "activation": "Tanh",
            }
        ],
        "head_scale": 1.0,
        "params": [{"name": "gain", "min": 0.0, "max": 1.0, "default": 0.5}],
    }


def _model_config():
    return {
        "net": {
            "name": "ParametricWaveNet",
            "config": _net_config(),
        },
        "optimizer": {"lr": 0.001},
        "lr_scheduler": None,
        "loss": {"val_loss": "mse"},
    }


def _learning_config():
    return {
        "train_dataloader": {
            "batch_size": 2,
            "shuffle": False,
            "drop_last": False,
            "num_workers": 0,
        },
        "val_dataloader": {
            "batch_size": 1,
            "num_workers": 0,
        },
        "trainer": {
            "accelerator": "cpu",
            "devices": 1,
            "max_epochs": 1,
            "limit_train_batches": 1,
            "limit_val_batches": 1,
            "num_sanity_val_steps": 0,
            "enable_progress_bar": False,
            "enable_model_summary": False,
            "logger": False,
        },
        "trainer_fit_kwargs": {},
    }


def test_tf2_full_main_parametric_make_plots_true(tmp_path):
    """
    TF2: full.main() with make_plots=True completes without AttributeError;
    comparison.png is created.
    """
    x_path, y_path = _write_wav_pair(tmp_path)
    outdir = tmp_path / "out"
    outdir.mkdir()

    _full.main(
        _data_config(x_path, y_path),
        _model_config(),
        _learning_config(),
        outdir,
        no_show=True,
        make_plots=True,
    )

    assert (outdir / "model.nam").exists(), "model.nam was not exported"
    assert (outdir / "comparison.png").exists(), "comparison.png was not created"


def test_tf1_full_main_parametric_selects_parametric_lightning_module(tmp_path):
    """
    TF1: full.main() with ParametricWaveNet config completes training without the
    3-tuple mis-route and exports model.nam.

    The 3-tuple mis-route is proven fatal in PA8b (test_train.py): the base
    LightningModule._shared_step would pass params as x on a (params, x, y) batch.
    A successful run here confirms the dispatch seam selects ParametricLightningModule.
    """
    x_path, y_path = _write_wav_pair(tmp_path)
    outdir = tmp_path / "out"
    outdir.mkdir()

    _full.main(
        _data_config(x_path, y_path),
        _model_config(),
        _learning_config(),
        outdir,
        no_show=True,
        make_plots=False,
    )

    assert (outdir / "model.nam").exists(), "model.nam was not exported"


def _write_wav_pair_to(tmp_path, subname, seed_offset=0):
    """Write a WAV pair into a named subdirectory; return (x_path_str, y_path_str)."""
    sub = tmp_path / subname
    sub.mkdir()
    t = np.arange(_NUM_SAMPLES, dtype=np.float64) / _RATE
    x = 0.10 * np.sin(2.0 * np.pi * 220.0 * t + seed_offset)
    y = 0.50 * x + 0.02 * np.sin(2.0 * np.pi * 440.0 * t + seed_offset)
    x_path = sub / "input.wav"
    y_path = sub / "output.wav"
    _np_to_wav(x, x_path, rate=_RATE)
    _np_to_wav(y, y_path, rate=_RATE)
    return str(x_path), str(y_path)


def test_tf3_full_main_multi_capture_parametric(tmp_path):
    """
    TF3: full.main() with a list-based multi-capture parametric config builds
    ParametricConcatDataset (not ConcatDataset) and exports model.nam.

    Two WAV pairs with different param values are provided. The config uses
    type=="parametric" with list-valued train/validation splits, which must
    dispatch to ParametricConcatDataset via the concat registry (N3).
    """
    x1, y1 = _write_wav_pair_to(tmp_path, "cap0", seed_offset=0)
    x2, y2 = _write_wav_pair_to(tmp_path, "cap1", seed_offset=1)
    outdir = tmp_path / "out"
    outdir.mkdir()

    data_config = {
        "type": "parametric",
        "common": {
            "delay": 0,
            "require_input_pre_silence": None,
            "param_names": ["gain"],
        },
        "train": [
            {"x_path": x1, "y_path": y1, "params": [0.5], "stop_samples": -_NUM_VALIDATION_SAMPLES, "ny": _NY},
            {"x_path": x2, "y_path": y2, "params": [0.8], "stop_samples": -_NUM_VALIDATION_SAMPLES, "ny": _NY},
        ],
        "validation": [
            {"x_path": x1, "y_path": y1, "params": [0.5], "start_samples": -_NUM_VALIDATION_SAMPLES, "ny": None},
            {"x_path": x2, "y_path": y2, "params": [0.8], "start_samples": -_NUM_VALIDATION_SAMPLES, "ny": None},
        ],
    }

    _full.main(
        data_config,
        _model_config(),
        _learning_config(),
        outdir,
        no_show=True,
        make_plots=False,
    )

    assert (outdir / "model.nam").exists(), "model.nam was not exported"


def test_create_callbacks_includes_validation_stopping_when_threshold_esr_set():
    callbacks = _full._create_callbacks(
        _learning_config(),
        threshold_esr=0.01,
    )

    assert any(isinstance(cb, _core._ValidationStopping) for cb in callbacks)
