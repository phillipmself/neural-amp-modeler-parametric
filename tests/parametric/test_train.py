import json as _json
from pathlib import Path as _Path
from typing import Any as _Any
from typing import cast as _cast

import numpy as _np
import pytest as _pytest
import torch as _torch

from nam.data import Dataset as _Dataset
from nam.data import np_to_wav as _np_to_wav
from nam.models.parametric import ParametricDataset as _ParametricDataset
from nam.models._from_nam import init_from_nam as _init_from_nam
from nam.models.wavenet import WaveNet as _WaveNet
from nam.train.core import _ValidationStopping as _ValidationStopping
from nam.train.parametric import _CaptureBatchSampler as _CaptureBatchSampler
from nam.train.parametric import _ParametricLightningModule as _ParametricLightningModule
from nam.train.parametric import _TRAIN_ESR_BUCKET as _TRAIN_ESR_BUCKET
from nam.train.parametric import _VALIDATION_ESR_BUCKET as _VALIDATION_ESR_BUCKET
from nam.train.parametric import _create_parametric_callbacks as _create_parametric_callbacks
from nam.train.parametric import _make_parametric_dataloader as _make_parametric_dataloader
from nam.train.parametric import _parametric_plot_label as _parametric_plot_label
from nam.train.parametric import main as _main
from tests.test_nam.test_models.test_base import MockBaseNet as _MockBaseNet


def _write_json(path: _Path, payload: dict) -> None:
    with open(path, "w") as fp:
        _json.dump(payload, fp)


def _load_json(path: _Path) -> dict:
    with open(path) as fp:
        return _json.load(fp)


def _tiny_model_config() -> dict:
    return {
        "net": {
            "name": "HyperWaveNet",
            "config": {
                "layers": [
                    {
                        "input_size": 1,
                        "condition_size": 1,
                        "channels": 3,
                        "head": {
                            "out_channels": 2,
                            "kernel_size": 1,
                            "bias": False,
                        },
                        "kernel_size": 3,
                        "dilations": [1, 2],
                        "activation": "Tanh",
                    },
                    {
                        "condition_size": 1,
                        "input_size": 3,
                        "channels": 2,
                        "head": {
                            "out_channels": 1,
                            "kernel_size": 1,
                            "bias": True,
                        },
                        "kernel_sizes": [3, 3],
                        "dilations": [1, 2],
                        "activation": "Tanh",
                    },
                ],
                "head": None,
                "head_scale": 0.02,
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
                "hypernet": {},
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
        "val_dataloader": {"batch_size": 1},
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
    return voicing * _np.tanh(drive * x)


def _build_data_config(tmp_path, *, normalize: bool) -> dict:
    sample_rate = 48_000
    x = (
        0.08 * _np.sin(2.0 * _np.pi * 220.0 * _np.arange(192) / sample_rate)
        + 0.03 * _np.cos(2.0 * _np.pi * 440.0 * _np.arange(192) / sample_rate)
    ).astype(_np.float32)
    captures = {
        "gain2_clean": {"gain": 2.0, "mode": "clean", "mode_index": 0},
        "gain8_crunch": {"gain": 8.0, "mode": "crunch", "mode_index": 1},
        "gain5_lead": {"gain": 5.0, "mode": "lead", "mode_index": 2},
    }

    inputs_dir = tmp_path / "inputs"
    outputs_dir = tmp_path / "outputs"
    inputs_dir.mkdir()
    outputs_dir.mkdir()

    x_path = inputs_dir / "input.wav"
    _np_to_wav(x, x_path, rate=sample_rate)
    for basename, capture in captures.items():
        y = _render_output(
            x, gain=capture["gain"], mode_index=capture["mode_index"]
        ).astype(_np.float32)
        _np_to_wav(y, outputs_dir / f"{basename}.wav", rate=sample_rate)

    data_config = {
        "type": "parametric",
        "common": {
            "x_path": str(x_path),
            "delay": 0,
            "require_input_pre_silence": None,
        },
        "train": [
            {
                "y_path": str(outputs_dir / "gain2_clean.wav"),
                "params": {"gain": 2.0, "mode": "clean"},
                "start_samples": 0,
                "stop_samples": 128,
                "ny": 16,
            },
            {
                "y_path": str(outputs_dir / "gain8_crunch.wav"),
                "params": {"gain": 8.0, "mode": "crunch"},
                "start_samples": 0,
                "stop_samples": 128,
                "ny": 16,
            },
        ],
        "validation": [
            {
                "y_path": str(outputs_dir / "gain5_lead.wav"),
                "params": {"gain": 5.0, "mode": "lead"},
                "start_samples": 96,
                "stop_samples": None,
                "ny": None,
            }
        ],
    }
    if normalize:
        data_config["joint"] = [
            {
                "name": "nam.data.normalize_joint_dataset_output",
                "kwargs": {"level_rms_dbfs": -18.0},
            }
        ]
    return data_config


def test_parametric_training_main_exports_baked_and_parametric_models(tmp_path):
    model_config = _tiny_model_config()
    data_config = _build_data_config(tmp_path, normalize=True)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    _main(
        data_config=data_config,
        model_config=model_config,
        learning_config=_learning_config(),
        outdir=run_dir,
        no_show=True,
        make_plots=False,
    )

    baked = _load_json(run_dir / "model.nam")
    parametric = _load_json(run_dir / "model_parametric.nam")

    assert baked["architecture"] == "WaveNet"
    assert parametric["architecture"] == "HyperWaveNet"
    assert "params" in parametric["config"]
    assert (run_dir / "config_data.json").exists()
    assert (run_dir / "config_model.json").exists()
    assert (run_dir / "config_learning.json").exists()
    assert any(run_dir.rglob("*.ckpt"))

    # --- Output-scale compensation wiring (the fragile "approach B" path). ---
    # head_scale is a fixed (untrained) constant, so the exported value moving away from it
    # is direct evidence the joint -18 dBFS normalization was compensated for at export. A
    # silently-dropped scale (e.g. output_scale recovered as None) would leave it untouched.
    raw_head_scale = model_config["net"]["config"]["head_scale"]
    baked_head_scale = baked["config"]["head_scale"]
    assert baked_head_scale != _pytest.approx(raw_head_scale, rel=1e-3)
    # The stock WaveNet stores head_scale in its final weight slot; the scale hook moves both
    # by the same factor, so they must stay equal.
    assert baked["weights"][-1] == _pytest.approx(baked_head_scale)
    # Both export paths must apply the one shared compensation factor.
    assert parametric["config"]["head_scale"] == _pytest.approx(baked_head_scale)

    # The baked file is a plain stock WaveNet that reloads and runs.
    round_tripped = _init_from_nam(_load_json(run_dir / "model.nam"))
    assert isinstance(round_tripped, _WaveNet)
    y = round_tripped(_torch.randn(round_tripped.receptive_field + 64), pad_start=False)
    assert _torch.isfinite(y).all()


def test_parametric_training_without_normalization_skips_compensation(tmp_path):
    model_config = _tiny_model_config()
    data_config = _build_data_config(tmp_path, normalize=False)
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    _main(
        data_config=data_config,
        model_config=model_config,
        learning_config=_learning_config(),
        outdir=run_dir,
        no_show=True,
        make_plots=False,
    )

    baked = _load_json(run_dir / "model.nam")
    parametric = _load_json(run_dir / "model_parametric.nam")

    # No joint normalization => no output scaling => export applies no compensation, so the
    # constant head_scale survives untouched in both export paths.
    raw_head_scale = model_config["net"]["config"]["head_scale"]
    assert baked["config"]["head_scale"] == _pytest.approx(raw_head_scale)
    assert baked["weights"][-1] == _pytest.approx(raw_head_scale)
    assert parametric["config"]["head_scale"] == _pytest.approx(raw_head_scale)


def test_parametric_callbacks_include_validation_stopping():
    callbacks = _create_parametric_callbacks(
        {
            "trainer": {},
            "threshold_esr": 0.01,
        }
    )

    validation_stopping = [
        callback
        for callback in callbacks
        if isinstance(callback, _ValidationStopping)
    ]
    assert len(validation_stopping) == 1
    assert validation_stopping[0].monitor == "ESR"


def test_parametric_plot_label_uses_output_filename():
    dataset = _Dataset(
        x=_torch.zeros(3),
        y=_torch.zeros(3),
        nx=1,
        ny=None,
        y_path="/tmp/held_out_capture.wav",
    )
    ds = _ParametricDataset(dataset, _torch.tensor([1.0]))

    assert _parametric_plot_label(ds) == "held_out_capture.wav"


def test_parametric_plot_label_falls_back_to_params():
    dataset = _Dataset(x=_torch.zeros(3), y=_torch.zeros(3), nx=1, ny=None)
    ds = _ParametricDataset(dataset, _torch.tensor([3.0, -3.0]))

    assert _parametric_plot_label(ds) == "params=[3, -3]"


def test_parametric_lightning_training_logs_seen_audio_seen_params_esr(monkeypatch):
    module = _ParametricLightningModule(_MockBaseNet(1.0))
    captured: dict[str, _Any] = {}

    def capture(dictionary, **kwargs):
        captured.update(dictionary)
        captured["kwargs"] = kwargs

    monkeypatch.setattr(module, "log_dict", capture)
    x = _torch.randn(3, 9)
    targets = _torch.randn(3, 9)

    loss = module.training_step((x, targets), 0)
    logged_esr = _cast(_torch.Tensor, captured[_TRAIN_ESR_BUCKET])

    assert _TRAIN_ESR_BUCKET in captured
    assert _torch.allclose(logged_esr, module._esr_loss(x, targets))
    assert captured["kwargs"]["on_epoch"] is True
    assert captured["kwargs"]["on_step"] is False
    assert captured["kwargs"]["batch_size"] == targets.shape[0]
    assert _torch.allclose(
        loss,
        _cast(_torch.Tensor, module._get_loss_dict(x, targets)["MSE"].value),
    )


def test_parametric_lightning_validation_logs_unseen_audio_unseen_params_esr(monkeypatch):
    module = _ParametricLightningModule(_MockBaseNet(1.0))
    captured: dict[str, _Any] = {}

    def capture(dictionary, **kwargs):
        captured.update(dictionary)
        captured["kwargs"] = kwargs

    monkeypatch.setattr(module, "log_dict", capture)
    x = _torch.randn(2, 7)
    targets = _torch.randn(2, 7)

    val_loss = module.validation_step((x, targets), 0)
    logged_bucket_esr = _cast(_torch.Tensor, captured[_VALIDATION_ESR_BUCKET])
    logged_esr = _cast(_torch.Tensor, captured["ESR"])

    assert "val_loss" in captured
    assert "ESR" in captured
    assert _VALIDATION_ESR_BUCKET in captured
    assert _torch.allclose(logged_bucket_esr, logged_esr)
    assert _torch.allclose(_cast(_torch.Tensor, captured["val_loss"]), val_loss)
    assert captured["kwargs"]["on_epoch"] is True
    assert captured["kwargs"]["on_step"] is False
    assert captured["kwargs"]["batch_size"] == targets.shape[0]


def test_capture_batch_sampler_keeps_batches_within_one_capture():
    capture_lengths = [5, 3, 4]
    sampler = _CaptureBatchSampler(
        capture_lengths, batch_size=2, shuffle=False, drop_last=False
    )
    offsets = [0, 5, 8]
    ranges = [range(o, o + n) for o, n in zip(offsets, capture_lengths)]

    batches = list(sampler)
    # Every batch's indices must come from a single capture range.
    for batch in batches:
        owners = {
            next(i for i, r in enumerate(ranges) if idx in r) for idx in batch
        }
        assert len(owners) == 1, f"batch {batch} mixed captures {owners}"

    # ceil(5/2)+ceil(3/2)+ceil(4/2) = 3+2+2 = 7 batches, covering every index once.
    assert len(batches) == len(sampler) == 7
    assert sorted(idx for batch in batches for idx in batch) == list(range(12))


def test_capture_batch_sampler_drop_last_drops_partial_per_capture():
    sampler = _CaptureBatchSampler(
        [5, 3], batch_size=2, shuffle=False, drop_last=True
    )
    batches = list(sampler)
    # floor(5/2)+floor(3/2) = 2+1 = 3 full batches; the partial tail of each capture is gone.
    assert len(batches) == len(sampler) == 3
    assert all(len(batch) == 2 for batch in batches)


def test_capture_batch_sampler_shuffle_is_reproducible_and_varies_by_epoch():
    sampler = _CaptureBatchSampler(
        [4, 4], batch_size=2, shuffle=True, drop_last=False
    )
    _torch.manual_seed(0)
    epoch0 = list(sampler)
    epoch1 = list(sampler)
    _torch.manual_seed(0)
    rerun0 = list(sampler)
    assert epoch0 == rerun0  # same global seed -> same stream
    assert epoch0 != epoch1  # advancing the RNG reshuffles each epoch


def test_make_parametric_dataloader_can_opt_out_of_capture_grouping():
    dataset = list(range(6))
    grouped = _make_parametric_dataloader(
        dataset, {"batch_size": 2, "shuffle": False}
    )
    assert isinstance(grouped.batch_sampler, _CaptureBatchSampler)

    ungrouped = _make_parametric_dataloader(
        dataset, {"batch_size": 2, "shuffle": False, "capture_grouped_batches": False}
    )
    assert not isinstance(ungrouped.batch_sampler, _CaptureBatchSampler)
