"""
Pure helpers for the parametric trainer GUI.
"""

from __future__ import annotations

import math as _math
from copy import deepcopy as _deepcopy
from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path
from typing import Any as _Any
from typing import Dict as _Dict
from typing import Iterable as _Iterable
from typing import List as _List
from typing import Sequence as _Sequence

import torch as _torch

from nam.models.parametric import ParamSpec as _ParamSpec
from nam.train import core as _core

_NY_DEFAULT = 8192


@_dataclass(frozen=True)
class CaptureValidation(object):
    output_path: str
    params: _List[float]
    delay: int


@_dataclass(frozen=True)
class CoverageGap(object):
    name: str
    missing_min: bool
    missing_max: bool


def _require_finite_float(value: _Any, *, field_name: str, param_name: str = "") -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as e:
        prefix = f"Parameter '{param_name}': " if param_name else ""
        raise ValueError(f"{prefix}{field_name} must be a number.") from e
    if not _math.isfinite(parsed):
        prefix = f"Parameter '{param_name}': " if param_name else ""
        raise ValueError(f"{prefix}{field_name} must be finite.")
    return parsed


def build_param_specs(param_rows: _Sequence[_Dict[str, _Any]]) -> _List[_ParamSpec]:
    if len(param_rows) == 0:
        raise ValueError("At least one parameter is required.")

    param_specs: _List[_ParamSpec] = []
    names = set()
    for i, row in enumerate(param_rows, start=1):
        raw_name = row["name"]
        name = str(raw_name).strip()
        if name == "":
            raise ValueError(f"Parameter row {i} is missing a name.")
        if name in names:
            raise ValueError(f"Duplicate parameter name '{name}'.")
        names.add(name)
        min_value = _require_finite_float(row["min"], field_name="min", param_name=name)
        max_value = _require_finite_float(row["max"], field_name="max", param_name=name)
        default_value = _require_finite_float(
            row["default"], field_name="default", param_name=name
        )
        param_specs.append(
            _ParamSpec(name=name, min=min_value, max=max_value, default=default_value)
        )
    return param_specs


def synchronize_capture_rows(
    capture_rows: _Sequence[_Dict[str, _Any]],
    default_values: _Sequence[_Any],
    removed_index: int | None = None,
) -> _List[_Dict[str, _Any]]:
    target_len = len(default_values)
    synchronized: _List[_Dict[str, _Any]] = []
    for row in capture_rows:
        old_values = list(row.get("values", []))
        if removed_index is not None and removed_index < len(old_values):
            old_values = old_values[:removed_index] + old_values[removed_index + 1 :]
        new_values = old_values[:target_len]
        if len(new_values) < target_len:
            new_values.extend(default_values[len(new_values) : target_len])
        synchronized.append({"output_path": row["output_path"], "values": new_values})
    return synchronized


def add_unique_capture_rows(
    capture_rows: _Sequence[_Dict[str, _Any]],
    output_paths: _Iterable[str],
    default_values: _Sequence[_Any],
) -> _List[_Dict[str, _Any]]:
    existing_paths = {row["output_path"] for row in capture_rows}
    merged = [dict(output_path=row["output_path"], values=list(row.get("values", []))) for row in capture_rows]
    for output_path in output_paths:
        if output_path in existing_paths:
            continue
        merged.append({"output_path": output_path, "values": list(default_values)})
        existing_paths.add(output_path)
    return merged


def validate_capture_rows(
    capture_rows: _Sequence[_Dict[str, _Any]], param_specs: _Sequence[_ParamSpec]
) -> _List[CaptureValidation]:
    if len(capture_rows) == 0:
        raise ValueError("At least one output capture is required.")

    validated: _List[CaptureValidation] = []
    for row in capture_rows:
        output_path = str(row["output_path"])
        values = list(row.get("values", []))
        if len(values) != len(param_specs):
            raise ValueError(
                f"Capture '{output_path}' has {len(values)} parameter values but "
                f"{len(param_specs)} parameters are defined."
            )
        parsed_values: _List[float] = []
        for value, spec in zip(values, param_specs):
            parsed_value = _require_finite_float(
                value, field_name="value", param_name=spec.name
            )
            if not (spec.min <= parsed_value <= spec.max):
                raise ValueError(
                    f"Capture '{output_path}' parameter '{spec.name}' must be within "
                    f"[{spec.min}, {spec.max}], got {parsed_value}."
                )
            parsed_values.append(parsed_value)
        delay = int(row.get("delay", 0))
        validated.append(
            CaptureValidation(
                output_path=output_path,
                params=parsed_values,
                delay=delay,
            )
        )
    return validated


def find_missing_param_extrema(
    param_specs: _Sequence[_ParamSpec],
    captures: _Sequence[CaptureValidation],
) -> _List[CoverageGap]:
    gaps: _List[CoverageGap] = []
    for i, spec in enumerate(param_specs):
        values = [capture.params[i] for capture in captures]
        missing_min = all(value != spec.min for value in values)
        missing_max = all(value != spec.max for value in values)
        if missing_min or missing_max:
            gaps.append(
                CoverageGap(
                    name=spec.name,
                    missing_min=missing_min,
                    missing_max=missing_max,
                )
            )
    return gaps


def format_coverage_message(gaps: _Sequence[CoverageGap]) -> str:
    lines = [
        "Your captures do not span the full declared parameter range.",
        "Training can continue, but you are missing these extrema:",
        "",
    ]
    for gap in gaps:
        missing = []
        if gap.missing_min:
            missing.append("min")
        if gap.missing_max:
            missing.append("max")
        lines.append(f" - {gap.name}: missing {', '.join(missing)}")
    lines.extend(["", "Continue anyway?"])
    return "\n".join(lines)


_DEFAULT_ADAPTER_LR = 5.0e-4


def _normalize_optional_non_negative_int(
    value: _Any,
    *,
    field_name: str,
) -> int | None:
    if value is None or value == "":
        return None
    parsed = int(value)
    if parsed < 0:
        raise ValueError(f"{field_name} must be non-negative, got {parsed}.")
    return parsed


def build_parametric_model_config(
    param_specs: _Sequence[_ParamSpec],
    *,
    adapter_first_n_layers: int | None = None,
    adapter_last_n_layers: int | None = None,
) -> _Dict[str, _Any]:
    adapter_first_n_layers = _normalize_optional_non_negative_int(
        adapter_first_n_layers,
        field_name="adapter_first_n_layers",
    )
    adapter_last_n_layers = _normalize_optional_non_negative_int(
        adapter_last_n_layers,
        field_name="adapter_last_n_layers",
    )
    packed = _core.get_packed_model_config()
    channels_8 = next(
        submodel
        for submodel in packed["net"]["config"]["submodels"]
        if submodel["name"] == "channels_8"
    )
    optimizer = _deepcopy(packed["optimizer"])
    optimizer["adapter_lr"] = _DEFAULT_ADAPTER_LR
    optimizer["adapter_weight_decay"] = optimizer["weight_decay"]
    return {
        "net": {
            "name": "ParametricWaveNet",
            "config": {
                "layers_configs": _deepcopy(channels_8["config"]["layers_configs"]),
                "head_scale": channels_8["config"]["head_scale"],
                "params": [spec.to_dict() for spec in param_specs],
                **(
                    {"adapter_first_n_layers": adapter_first_n_layers}
                    if adapter_first_n_layers is not None
                    else {}
                ),
                **(
                    {"adapter_last_n_layers": adapter_last_n_layers}
                    if adapter_last_n_layers is not None
                    else {}
                ),
            },
        },
        "loss": _deepcopy(packed["loss"]),
        "optimizer": optimizer,
        "lr_scheduler": _deepcopy(packed["lr_scheduler"]),
    }


def build_learning_config(
    num_epochs: int,
    batch_size: int,
    threshold_esr: float | None = None,
) -> _Dict[str, _Any]:
    if _torch.cuda.is_available():
        device_config = {"accelerator": "gpu", "devices": 1}
    elif _torch.backends.mps.is_available():
        device_config = {"accelerator": "mps", "devices": 1}
    else:
        device_config = {}
    config = {
        "train_dataloader": {
            "batch_size": batch_size,
            "shuffle": True,
            "pin_memory": True,
            "drop_last": True,
            "num_workers": 0,
        },
        "val_dataloader": {},
        "trainer": {"max_epochs": num_epochs, **device_config},
    }
    if threshold_esr is not None:
        config["threshold_esr"] = threshold_esr
    return config


def default_num_epochs() -> int:
    return 100 if (_torch.cuda.is_available() or _torch.backends.mps.is_available()) else 20


def default_batch_size() -> int:
    return 16 if (_torch.cuda.is_available() or _torch.backends.mps.is_available()) else 1


def build_parametric_data_config(
    input_path: str,
    param_specs: _Sequence[_ParamSpec],
    captures: _Sequence[CaptureValidation],
    ny: int = _NY_DEFAULT,
) -> _Dict[str, _Any]:
    input_version = _core.detect_input_version(_Path(input_path))
    train_entries = []
    validation_entries = []
    base_joint = None
    for capture in captures:
        capture_config = _core.build_standardized_data_config(
            input_version=input_version,
            input_path=_Path(input_path),
            output_path=_Path(capture.output_path),
            ny=ny,
            latency=capture.delay,
        )
        if base_joint is None:
            base_joint = _deepcopy(capture_config["joint"])
        train_entries.append(
            {
                "y_path": capture.output_path,
                "delay": capture.delay,
                "params": list(capture.params),
                **_deepcopy(capture_config["train"]),
            }
        )
        validation_entries.append(
            {
                "y_path": capture.output_path,
                "delay": capture.delay,
                "params": list(capture.params),
                **_deepcopy(capture_config["validation"]),
            }
        )
    return {
        "type": "parametric",
        "common": {
            "x_path": input_path,
            "allow_unequal_lengths": True,
            "param_names": [spec.name for spec in param_specs],
        },
        "train": train_entries,
        "validation": validation_entries,
        "joint": [] if base_joint is None else base_joint,
    }
