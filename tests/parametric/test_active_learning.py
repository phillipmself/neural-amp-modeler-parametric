from copy import deepcopy as _deepcopy

import numpy as _np
import pytest as _pytest
import torch as _torch

from nam.data import np_to_wav as _np_to_wav
from nam.train.active_learning import _clear_device_cache as _clear_device_cache
from nam.train.active_learning import train_ensemble as _train_ensemble
from nam.train.parametric import _ParametricLightningModule


def _concat_lstm_model_config() -> dict:
    return {
        "net": {
            "name": "ConcatLSTM",
            "config": {
                "hidden_size": 4,
                "num_layers": 1,
                "train_burn_in": 8,
                "train_truncate": 8,
                "params": [
                    {
                        "name": "gain",
                        "min": 0.0,
                        "max": 10.0,
                        "default": 5.0,
                        "type": "continuous",
                    },
                    {
                        "name": "mode",
                        "min": 0,
                        "max": 2,
                        "default": 1,
                        "type": "switch",
                        "enum_names": ["clean", "crunch", "lead"],
                    },
                ],
            },
        },
        "loss": {"val_loss": "esr"},
        "optimizer": {"lr": 0.01},
        "lr_scheduler": {"class": "ExponentialLR", "kwargs": {"gamma": 0.99}},
    }


def _learning_config() -> dict:
    return {
        "train_dataloader": {
            "batch_size": 2,
            "shuffle": True,
            "pin_memory": False,
            "drop_last": False,
            "num_workers": 0,
        },
        "val_dataloader": {"batch_size": 1, "pin_memory": False, "num_workers": 0},
        "trainer": {
            "accelerator": "cpu",
            "devices": 1,
            "max_epochs": 1,
            "logger": False,
            "enable_model_summary": False,
            "enable_progress_bar": False,
            "limit_train_batches": 2,
            "limit_val_batches": 1,
            "num_sanity_val_steps": 0,
        },
        "threshold_esr": None,
        "trainer_fit_kwargs": {},
    }


def _render_output(x: _np.ndarray, *, gain: float, mode_index: int) -> _np.ndarray:
    drive = 0.35 + 0.05 * gain
    voicing = 1.0 + 0.1 * mode_index
    return (voicing * _np.tanh(drive * x)).astype(_np.float32)


def _build_data_config(tmp_path, *, train_ny: int = 16) -> dict:
    sample_rate = 48_000
    x = (
        0.08 * _np.sin(2.0 * _np.pi * 220.0 * _np.arange(192) / sample_rate)
        + 0.03 * _np.cos(2.0 * _np.pi * 440.0 * _np.arange(192) / sample_rate)
    ).astype(_np.float32)

    x_path = tmp_path / "input.wav"
    _np_to_wav(x, x_path, rate=sample_rate)

    for basename, gain, mode_index in (
        ("gain2_clean.wav", 2.0, 0),
        ("gain8_crunch.wav", 8.0, 1),
        ("gain5_lead.wav", 5.0, 2),
    ):
        _np_to_wav(
            _render_output(x, gain=gain, mode_index=mode_index),
            tmp_path / basename,
            rate=sample_rate,
        )

    return {
        "type": "parametric",
        "common": {
            "x_path": str(x_path),
            "delay": 0,
            "require_input_pre_silence": None,
        },
        "train": [
            {
                "y_path": str(tmp_path / "gain2_clean.wav"),
                "params": {"gain": 2.0, "mode": "clean"},
                "start_samples": 0,
                "stop_samples": 128,
                "ny": train_ny,
            },
            {
                "y_path": str(tmp_path / "gain8_crunch.wav"),
                "params": {"gain": 8.0, "mode": "crunch"},
                "start_samples": 0,
                "stop_samples": 128,
                "ny": train_ny,
            },
        ],
        "validation": [
            {
                "y_path": str(tmp_path / "gain5_lead.wav"),
                "params": {"gain": 5.0, "mode": "lead"},
                "start_samples": 96,
                "stop_samples": None,
                "ny": None,
            }
        ],
    }


def test_train_ensemble_writes_stable_member_checkpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cpu"),
    )
    outdir = tmp_path / "run"

    checkpoint_paths = _train_ensemble(
        _build_data_config(tmp_path),
        _concat_lstm_model_config(),
        _learning_config(),
        outdir,
        ensemble_size=2,
    )

    assert len(checkpoint_paths) == 2
    assert [path.name for path in checkpoint_paths] == ["best.ckpt", "best.ckpt"]
    assert all(path.exists() for path in checkpoint_paths)


def test_train_ensemble_checkpoints_reload_via_task5_path(tmp_path, monkeypatch):
    # Task 5 (find_disagreement_settings) reloads each member with
    # load_from_checkpoint(path, **parse_config(model_config)) and runs net forward on
    # raw params. Prove a returned checkpoint honors exactly that contract.
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cpu"),
    )
    model_config = _concat_lstm_model_config()

    checkpoint_paths = _train_ensemble(
        _build_data_config(tmp_path),
        model_config,
        _learning_config(),
        tmp_path / "run",
        ensemble_size=1,
    )

    module = _ParametricLightningModule.load_from_checkpoint(
        str(checkpoint_paths[0]),
        **_ParametricLightningModule.parse_config(model_config),
    )
    module.eval()
    x = _torch.zeros(32)
    raw_params = _torch.tensor([5.0, 1.0])  # gain (continuous), mode index (switch)
    with _torch.no_grad():
        y = module(x, raw_params)
    assert y.shape[-1] == x.shape[-1]
    assert _torch.isfinite(y).all()


def test_train_ensemble_accepts_ny_in_common(tmp_path, monkeypatch):
    # init_dataset merges common into each entry, so ny declared once in common is valid
    # and must not be rejected as a missing per-entry ny.
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cpu"),
    )
    data_config = _build_data_config(tmp_path)
    data_config["common"]["ny"] = 16
    for entry in data_config["train"]:
        del entry["ny"]

    checkpoint_paths = _train_ensemble(
        data_config,
        _concat_lstm_model_config(),
        _learning_config(),
        tmp_path / "run",
        ensemble_size=1,
    )

    assert len(checkpoint_paths) == 1
    assert checkpoint_paths[0].exists()


def test_train_ensemble_allows_default_full_length_windows(tmp_path, monkeypatch):
    # ny=None is the stock "use the full clip length" sentinel: the normal long-window
    # LSTM case, which must not trip the burn-in guard.
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cpu"),
    )
    data_config = _build_data_config(tmp_path)
    for entry in data_config["train"]:
        entry["ny"] = None

    checkpoint_paths = _train_ensemble(
        data_config,
        _concat_lstm_model_config(),
        _learning_config(),
        tmp_path / "run",
        ensemble_size=1,
    )

    assert len(checkpoint_paths) == 1
    assert checkpoint_paths[0].exists()


def test_train_ensemble_rejects_train_windows_at_or_below_burn_in(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cpu"),
    )

    with _pytest.raises(ValueError, match=r"train\[0\]\.ny \(=8\) must be greater than train_burn_in=8"):
        _train_ensemble(
            _build_data_config(tmp_path, train_ny=8),
            _concat_lstm_model_config(),
            _learning_config(),
            tmp_path / "run",
            ensemble_size=1,
        )


def test_train_ensemble_allows_round_tripped_param_specs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cpu"),
    )
    model_config = _concat_lstm_model_config()
    data_config = _build_data_config(tmp_path)
    data_config["common"]["param_specs"] = _deepcopy(model_config["net"]["config"]["params"])

    checkpoint_paths = _train_ensemble(
        data_config,
        model_config,
        _learning_config(),
        tmp_path / "run",
        ensemble_size=1,
    )

    assert len(checkpoint_paths) == 1
    assert checkpoint_paths[0].exists()


def test_train_ensemble_rejects_mismatched_round_tripped_param_specs(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "nam.train.active_learning._resolve_device",
        lambda: _torch.device("cpu"),
    )
    model_config = _concat_lstm_model_config()
    data_config = _build_data_config(tmp_path)
    data_config["common"]["param_specs"] = _deepcopy(model_config["net"]["config"]["params"])
    data_config["common"]["param_specs"][0]["max"] = 11.0

    with _pytest.raises(ValueError, match="common.param_specs does not match"):
        _train_ensemble(
            data_config,
            model_config,
            _learning_config(),
            tmp_path / "run",
            ensemble_size=1,
        )


def test_clear_device_cache_calls_mps_empty_cache(monkeypatch):
    mps_module = getattr(_torch, "mps", None)
    if mps_module is None or not hasattr(mps_module, "empty_cache"):
        _pytest.skip("torch.mps.empty_cache is unavailable in this environment")

    calls = []
    monkeypatch.setattr(mps_module, "empty_cache", lambda: calls.append(True))

    _clear_device_cache(_torch.device("mps"))

    assert calls == [True]
