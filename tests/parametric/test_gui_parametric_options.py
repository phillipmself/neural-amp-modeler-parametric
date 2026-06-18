import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from nam.data import np_to_wav as _np_to_wav
from nam.train import full as _full
from nam.train._version import Version as _Version
from nam.train.gui import _parametric as _helpers
from nam.train.gui import parametric as _gui


_ROOT = Path(__file__).resolve().parents[2]
_PACKED_MODEL_PATH = _ROOT / "nam" / "train" / "_resources" / "config_model_packed.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_build_parametric_model_config_matches_standard_a2_channels_8():
    packed = _load_json(_PACKED_MODEL_PATH)
    param_specs = _helpers.build_param_specs(
        [
            {"name": "gain", "min": 0.0, "max": 10.0, "default": 5.0},
            {"name": "bright", "min": 0.0, "max": 1.0, "default": 0.5},
        ]
    )

    built = _helpers.build_parametric_model_config(param_specs)
    channels_8 = next(
        submodel
        for submodel in packed["net"]["config"]["submodels"]
        if submodel["name"] == "channels_8"
    )

    assert built["net"]["name"] == "ParametricWaveNet"
    assert built["net"]["config"]["layers_configs"] == channels_8["config"]["layers_configs"]
    assert built["net"]["config"]["head_scale"] == channels_8["config"]["head_scale"]
    assert built["loss"] == packed["loss"]
    assert built["optimizer"] == packed["optimizer"]
    assert built["lr_scheduler"] == packed["lr_scheduler"]
    assert built["net"]["config"]["params"] == [spec.to_dict() for spec in param_specs]


def test_build_param_specs_requires_unique_nonempty_names():
    try:
        _helpers.build_param_specs(
            [
                {"name": "gain", "min": 0.0, "max": 1.0, "default": 0.5},
                {"name": "gain", "min": 0.0, "max": 1.0, "default": 0.5},
            ]
        )
    except ValueError as e:
        assert "Duplicate parameter name" in str(e)
    else:  # pragma: no cover
        raise AssertionError("Expected duplicate-name validation to fail")


def test_validate_capture_rows_rejects_out_of_range_values():
    param_specs = _helpers.build_param_specs(
        [{"name": "gain", "min": 0.0, "max": 1.0, "default": 0.5}]
    )
    try:
        _helpers.validate_capture_rows(
            [{"output_path": "capture.wav", "values": ["1.5"], "delay": 0}],
            param_specs,
        )
    except ValueError as e:
        assert "within [0.0, 1.0]" in str(e)
    else:  # pragma: no cover
        raise AssertionError("Expected out-of-range capture validation to fail")


def test_synchronize_capture_rows_adds_defaults_and_removes_extras():
    capture_rows = [
        {"output_path": "a.wav", "values": ["0.25"]},
        {"output_path": "b.wav", "values": ["0.75"]},
    ]

    expanded = _helpers.synchronize_capture_rows(capture_rows, ["0.5", "1.0"])
    assert expanded == [
        {"output_path": "a.wav", "values": ["0.25", "1.0"]},
        {"output_path": "b.wav", "values": ["0.75", "1.0"]},
    ]

    shrunk = _helpers.synchronize_capture_rows(expanded, ["0.5"])
    assert shrunk == [
        {"output_path": "a.wav", "values": ["0.25"]},
        {"output_path": "b.wav", "values": ["0.75"]},
    ]


def test_synchronize_capture_rows_removes_deleted_middle_param_column():
    capture_rows = [
        {"output_path": "a.wav", "values": ["0.25", "0.50", "0.75"]},
        {"output_path": "b.wav", "values": ["1.25", "1.50", "1.75"]},
    ]

    synchronized = _helpers.synchronize_capture_rows(
        capture_rows,
        ["0.10", "0.90"],
        removed_index=1,
    )

    assert synchronized == [
        {"output_path": "a.wav", "values": ["0.25", "0.75"]},
        {"output_path": "b.wav", "values": ["1.25", "1.75"]},
    ]


def test_add_unique_capture_rows_ignores_duplicates():
    capture_rows = [{"output_path": "a.wav", "values": ["0.5"]}]

    merged = _helpers.add_unique_capture_rows(
        capture_rows,
        ["a.wav", "b.wav", "b.wav"],
        ["0.25"],
    )

    assert merged == [
        {"output_path": "a.wav", "values": ["0.5"]},
        {"output_path": "b.wav", "values": ["0.25"]},
    ]


def test_find_missing_param_extrema_all_present():
    param_specs = _helpers.build_param_specs(
        [{"name": "gain", "min": 0.0, "max": 1.0, "default": 0.5}]
    )
    captures = _helpers.validate_capture_rows(
        [
            {"output_path": "a.wav", "values": ["0.0"], "delay": 0},
            {"output_path": "b.wav", "values": ["1.0"], "delay": 0},
        ],
        param_specs,
    )
    assert _helpers.find_missing_param_extrema(param_specs, captures) == []


def test_find_missing_param_extrema_missing_only_min():
    param_specs = _helpers.build_param_specs(
        [{"name": "gain", "min": 0.0, "max": 1.0, "default": 0.5}]
    )
    captures = _helpers.validate_capture_rows(
        [
            {"output_path": "a.wav", "values": ["0.5"], "delay": 0},
            {"output_path": "b.wav", "values": ["1.0"], "delay": 0},
        ],
        param_specs,
    )
    assert _helpers.find_missing_param_extrema(param_specs, captures) == [
        _helpers.CoverageGap(name="gain", missing_min=True, missing_max=False)
    ]


def test_find_missing_param_extrema_missing_only_max():
    param_specs = _helpers.build_param_specs(
        [{"name": "gain", "min": 0.0, "max": 1.0, "default": 0.5}]
    )
    captures = _helpers.validate_capture_rows(
        [
            {"output_path": "a.wav", "values": ["0.0"], "delay": 0},
            {"output_path": "b.wav", "values": ["0.5"], "delay": 0},
        ],
        param_specs,
    )
    assert _helpers.find_missing_param_extrema(param_specs, captures) == [
        _helpers.CoverageGap(name="gain", missing_min=False, missing_max=True)
    ]


def test_find_missing_param_extrema_multiple_params():
    param_specs = _helpers.build_param_specs(
        [
            {"name": "gain", "min": 0.0, "max": 1.0, "default": 0.5},
            {"name": "bright", "min": 0.0, "max": 10.0, "default": 5.0},
        ]
    )
    captures = _helpers.validate_capture_rows(
        [
            {"output_path": "a.wav", "values": ["0.0", "5.0"], "delay": 0},
            {"output_path": "b.wav", "values": ["0.5", "10.0"], "delay": 0},
        ],
        param_specs,
    )
    assert _helpers.find_missing_param_extrema(param_specs, captures) == [
        _helpers.CoverageGap(name="gain", missing_min=False, missing_max=True),
        _helpers.CoverageGap(name="bright", missing_min=True, missing_max=False),
    ]


def test_find_missing_param_extrema_single_capture_defaults_only():
    param_specs = _helpers.build_param_specs(
        [{"name": "gain", "min": 0.0, "max": 1.0, "default": 0.5}]
    )
    captures = _helpers.validate_capture_rows(
        [{"output_path": "a.wav", "values": ["0.5"], "delay": 0}],
        param_specs,
    )
    assert _helpers.find_missing_param_extrema(param_specs, captures) == [
        _helpers.CoverageGap(name="gain", missing_min=True, missing_max=True)
    ]


def _write_wav_pair_to(tmp_path: Path, subname: str, seed_offset: int = 0) -> tuple[str, str]:
    num_samples = 20_480
    rate = 48_000
    t = np.arange(num_samples, dtype=np.float64) / rate
    x = 0.10 * np.sin(2.0 * np.pi * 220.0 * t + seed_offset)
    y = 0.50 * x + 0.02 * np.sin(2.0 * np.pi * 440.0 * t + seed_offset)
    sub = tmp_path / subname
    sub.mkdir()
    x_path = sub / "input.wav"
    y_path = sub / "output.wav"
    _np_to_wav(x, x_path, rate=rate)
    _np_to_wav(y, y_path, rate=rate)
    return str(x_path), str(y_path)


def test_gui_helper_configs_train_through_full_main(tmp_path, monkeypatch):
    x1, y1 = _write_wav_pair_to(tmp_path, "cap0", seed_offset=0)
    x2, y2 = _write_wav_pair_to(tmp_path, "cap1", seed_offset=1)
    param_specs = _helpers.build_param_specs(
        [{"name": "gain", "min": 0.0, "max": 1.0, "default": 0.5}]
    )
    captures = _helpers.validate_capture_rows(
        [
            {"output_path": y1, "values": ["0.0"], "delay": 0},
            {"output_path": y2, "values": ["1.0"], "delay": 0},
        ],
        param_specs,
    )

    def fake_build_standardized_data_config(
        input_version, input_path, output_path, ny, latency
    ):
        return {
            "train": {"stop_samples": -8_192, "ny": 8},
            "validation": {
                "start_samples": -8_192,
                "ny": None,
                "require_input_pre_silence": False,
            },
            "common": {
                "x_path": str(input_path),
                "y_path": str(output_path),
                "delay": latency,
                "allow_unequal_lengths": True,
            },
            "joint": [
                {
                    "name": "nam.data.normalize_joint_dataset_output",
                    "kwargs": {"level_rms_dbfs": -18.0},
                }
            ],
        }

    monkeypatch.setattr(_helpers._core, "detect_input_version", lambda _path: _Version(3, 0, 0))
    monkeypatch.setattr(
        _helpers._core,
        "build_standardized_data_config",
        fake_build_standardized_data_config,
    )

    data_config = _helpers.build_parametric_data_config(x1, param_specs, captures, ny=8)
    model_config = _helpers.build_parametric_model_config(param_specs)
    model_config["loss"] = {"val_loss": "mse"}
    learning_config = {
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
    }
    outdir = tmp_path / "out"
    outdir.mkdir()

    _full.main(
        deepcopy(data_config),
        model_config,
        learning_config,
        outdir,
        no_show=True,
        make_plots=False,
    )

    assert (outdir / "model.nam").exists()


def test_add_output_files_cancel_is_noop(monkeypatch):
    gui = _gui.GUI.__new__(_gui.GUI)
    gui._raw_capture_rows = lambda: []
    gui._default_capture_values = lambda: []
    replaced_rows = []
    gui._replace_capture_rows = lambda rows: replaced_rows.append(rows)

    set_last_path_calls = []
    monkeypatch.setattr(_gui._settings, "get_last_path", lambda _key: None)
    monkeypatch.setattr(
        _gui._filedialog,
        "askopenfilenames",
        lambda initialdir=None: (),
    )
    monkeypatch.setattr(
        _gui._settings,
        "set_last_path",
        lambda *args, **kwargs: set_last_path_calls.append((args, kwargs)),
    )

    gui._add_output_files()

    assert replaced_rows == []
    assert set_last_path_calls == []


def test_validate_for_training_surfaces_unusable_latency(monkeypatch):
    gui = _gui.GUI.__new__(_gui.GUI)
    gui._input_path = "input.wav"
    gui._raw_param_rows = lambda: [
        {"name": "gain", "min": "0.0", "max": "1.0", "default": "0.5"}
    ]
    gui._raw_capture_rows = lambda: [{"output_path": "capture.wav", "values": ["0.5"]}]

    validation = SimpleNamespace(
        passed=False,
        passed_critical=True,
        sample_rate=SimpleNamespace(passed=True, input=48_000, output=48_000),
        length=SimpleNamespace(passed=True, delta_seconds=0.0),
        latency=SimpleNamespace(
            manual=None,
            calibration=SimpleNamespace(
                warnings=SimpleNamespace(
                    matches_lookahead=False,
                    disagreement_too_high=False,
                    not_detected=True,
                )
            ),
        ),
        checks=SimpleNamespace(passed=True),
        pytorch=SimpleNamespace(
            passed=True,
            train=SimpleNamespace(passed=True, msg=None),
            validation=SimpleNamespace(passed=True, msg=None),
        ),
    )

    showerror_calls = []
    askyesno_calls = []
    monkeypatch.setattr(
        _gui._core,
        "validate_input",
        lambda _path: SimpleNamespace(passed=True),
    )
    monkeypatch.setattr(_gui._core, "validate_data", lambda *args, **kwargs: validation)
    monkeypatch.setattr(
        _gui._core,
        "get_final_latency",
        lambda _latency: (_ for _ in ()).throw(RuntimeError("No usable latency")),
    )
    monkeypatch.setattr(
        _gui._messagebox,
        "askyesno",
        lambda title, message: askyesno_calls.append((title, message)) or True,
    )
    monkeypatch.setattr(
        _gui._messagebox,
        "showerror",
        lambda title, message: showerror_calls.append((title, message)),
    )

    validated = gui._validate_for_training()

    assert validated is None
    assert len(askyesno_calls) == 1
    assert showerror_calls == [
        (
            "Latency Validation Failed",
            "Could not determine a usable latency for:\n\ncapture.wav: No usable latency",
        )
    ]
