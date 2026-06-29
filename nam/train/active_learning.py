"""
Active-learning ensemble training adapted from PANAMA.

The serial, device-agnostic ensemble flow follows PANAMA (Parametric Active-learning
for Neural Amp Modeling Assistance), arXiv:2509.26564v1, adapted to this repo's
parametric NAM training stack and runtime-selected device handling.
"""

import gc as _gc
import shutil as _shutil
from copy import deepcopy as _deepcopy
from pathlib import Path as _Path
from typing import Any as _Any
from warnings import warn as _warn

import pytorch_lightning as _pl
import torch as _torch
from lightning_fabric.utilities.warnings import PossibleUserWarning as _PossibleUserWarning
from pytorch_lightning.callbacks import ModelCheckpoint as _ModelCheckpoint

from nam.data import ConcatDataset as _ConcatDataset
from nam.data import Split as _Split
from nam.data import apply_joint_dataset_hooks as _apply_joint_dataset_hooks
from nam.data import get_joint_dataset_hooks as _get_joint_dataset_hooks
from nam.data import init_dataset as _init_dataset
from nam.models.parametric import data_config_from_model as _data_config_from_model
from nam.models.parametric._dataset import _coerce_param_specs
from nam.train.full import _handshake_datasets
from nam.train.parametric import _ParametricLightningModule
from nam.train.parametric import _create_parametric_callbacks
from nam.train.parametric import _iter_inner_datasets
from nam.train.parametric import _make_parametric_dataloader
from nam.util import filter_warnings as _filter_warnings


def _resolve_device() -> _torch.device:
    if _torch.cuda.is_available():
        return _torch.device("cuda")
    mps_backend = getattr(_torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return _torch.device("mps")
    return _torch.device("cpu")


def _trainer_device_config(device: _torch.device) -> dict[str, str | int]:
    if device.type == "cuda":
        return {"accelerator": "gpu", "devices": 1}
    if device.type == "mps":
        return {"accelerator": "mps", "devices": 1}
    if device.type == "cpu":
        return {"accelerator": "cpu", "devices": 1}
    raise ValueError(f"Unsupported device type {device.type!r}")


def _prepare_learning_config(
    learning_config: dict,
    device: _torch.device,
) -> dict:
    learning_config = _deepcopy(learning_config)
    trainer_config = dict(learning_config["trainer"])
    trainer_config.update(_trainer_device_config(device))
    learning_config["trainer"] = trainer_config
    return learning_config


def _canonical_param_specs(raw_param_specs: _Any) -> list[dict[str, _Any]]:
    # Reuse the stock coercion/validation (unique names, ParamSpec-or-mapping) and
    # canonicalize to dicts so two spec lists can be compared by value.
    return [spec.to_dict() for spec in _coerce_param_specs(raw_param_specs)]


def _prepare_data_config(data_config: dict, model_config: dict) -> dict:
    data_config = _deepcopy(data_config)
    common = data_config.get("common", {})
    if not isinstance(common, dict):
        raise ValueError("Data config common section must be a mapping")

    # Aggregated configs re-fed across rounds (Task 7, round > 0) may already carry
    # param_specs. Tolerate that only when it matches the model's specs, then strip it
    # so data_config_from_model re-injects the canonical copy (it raises on duplicates).
    if "param_specs" in common:
        try:
            model_param_specs = _canonical_param_specs(
                model_config["net"]["config"]["params"]
            )
        except KeyError as exc:
            raise ValueError(
                "Model config must define net.config.params for parametric dataset loading"
            ) from exc
        if _canonical_param_specs(common["param_specs"]) != model_param_specs:
            raise ValueError(
                "Data config common.param_specs does not match "
                "model_config['net']['config']['params']"
            )
        common = dict(common)
        del common["param_specs"]
        data_config["common"] = common

    return _data_config_from_model(data_config, model_config)


def _prepare_member_data_config(
    data_config: dict,
    receptive_field: int,
) -> dict:
    data_config = _deepcopy(data_config)
    common = data_config.setdefault("common", {})
    existing_nx = common.get("nx")
    if existing_nx is not None and existing_nx != receptive_field:
        _warn(
            f"Overriding data nx={existing_nx} with model required {receptive_field}"
        )
    common["nx"] = receptive_field
    return data_config


def _iter_split_entries(split_config: _Any):
    if isinstance(split_config, dict):
        yield "train", split_config
        return
    if isinstance(split_config, list):
        for i, item in enumerate(split_config):
            if not isinstance(item, dict):
                raise ValueError(
                    f"Expected train[{i}] to be a mapping, got {type(item).__name__}"
                )
            yield f"train[{i}]", item
        return
    raise ValueError(
        "Expected data_config['train'] to be a mapping or list of mappings, got "
        f"{type(split_config).__name__}"
    )


def _validate_train_window_lengths(data_config: dict, model_config: dict) -> None:
    net_config = model_config["net"]["config"]
    train_burn_in = net_config.get("train_burn_in")
    train_truncate = net_config.get("train_truncate")
    # ConcatLSTM only consumes train_burn_in inside the truncated-BPTT path; with no
    # train_truncate the full sequence is processed in one (gradient-carrying) pass and
    # burn-in is irrelevant. Nothing to guard against in that case.
    if train_burn_in is None or train_truncate is None:
        return
    # init_dataset merges common into every entry, so ny may live in either place;
    # entry overrides common. ny=None falls back to the full clip length, which is the
    # normal long-window LSTM case and effectively always exceeds burn-in.
    common = data_config.get("common", {})
    common_ny = common.get("ny") if isinstance(common, dict) else None
    for label, train_entry in _iter_split_entries(data_config["train"]):
        ny = train_entry.get("ny", common_ny)
        if ny is None:
            continue
        if not isinstance(ny, int) or isinstance(ny, bool):
            raise ValueError(f"{label}.ny must be an integer, got {ny!r}")
        if ny <= train_burn_in:
            raise ValueError(
                f"{label}.ny (={ny}) must be greater than train_burn_in={train_burn_in}: "
                "otherwise the burn-in window consumes the whole sequence and the member "
                "trains on zero gradient"
            )


def _build_dataloaders(
    data_config: dict,
    learning_config: dict,
    model: _ParametricLightningModule,
):
    net = model.net
    # getattr (not net.receptive_field) keeps the type checker from widening to nn.Module.
    receptive_field = int(getattr(net, "receptive_field"))
    data_config = _prepare_member_data_config(data_config, receptive_field)
    dataset_train = _init_dataset(data_config, _Split.TRAIN)
    dataset_validation = _init_dataset(data_config, _Split.VALIDATION)

    inner_train = _ConcatDataset(_iter_inner_datasets(dataset_train))
    inner_validation = _ConcatDataset(_iter_inner_datasets(dataset_validation))
    _apply_joint_dataset_hooks(
        dataset_train=inner_train,
        dataset_validation=inner_validation,
        hooks=_get_joint_dataset_hooks(data_config.get("joint", [])),
    )

    setattr(net, "sample_rate", getattr(dataset_train, "sample_rate", None))
    _handshake_datasets(model, dataset_train, dataset_validation)

    train_loader_config = dict(learning_config["train_dataloader"])
    train_loader_config["capture_grouped_batches"] = False
    train_loader_config["shuffle"] = True
    val_loader_config = dict(learning_config["val_dataloader"])
    val_loader_config["capture_grouped_batches"] = False
    val_loader_config["shuffle"] = False

    return (
        dataset_train,
        dataset_validation,
        _make_parametric_dataloader(dataset_train, train_loader_config),
        _make_parametric_dataloader(dataset_validation, val_loader_config),
    )


def _stabilize_checkpoint_path(best_checkpoint: str, member_outdir: _Path) -> _Path:
    if best_checkpoint == "":
        raise RuntimeError(
            f"No best checkpoint was produced for ensemble member output dir {member_outdir}"
        )
    best_checkpoint_path = _Path(best_checkpoint)
    if not best_checkpoint_path.exists():
        raise RuntimeError(f"Best checkpoint does not exist: {best_checkpoint_path}")
    stable_path = member_outdir / "best.ckpt"
    if best_checkpoint_path.resolve() != stable_path.resolve():
        _shutil.copy2(best_checkpoint_path, stable_path)
    return stable_path


def _clear_device_cache(device: _torch.device) -> None:
    if device.type == "cuda":
        _torch.cuda.empty_cache()
        return
    if device.type == "mps":
        mps_module = getattr(_torch, "mps", None)
        if mps_module is not None and hasattr(mps_module, "empty_cache"):
            mps_module.empty_cache()


def train_ensemble(
    data_config: dict,
    model_config: dict,
    learning_config: dict,
    outdir: _Path,
    *,
    ensemble_size: int = 4,
    base_seed: int = 0,
) -> list[_Path]:
    if ensemble_size <= 0:
        raise ValueError(f"ensemble_size must be positive; got {ensemble_size}")
    if model_config["net"]["name"] != "ConcatLSTM":
        raise ValueError(
            "train_ensemble requires model_config['net']['name'] == 'ConcatLSTM'; "
            f"got {model_config['net']['name']!r}"
        )

    outdir = _Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device()
    learning_config = _prepare_learning_config(learning_config, device)
    data_config = _prepare_data_config(data_config, model_config)
    _validate_train_window_lengths(data_config, model_config)

    checkpoint_paths: list[_Path] = []
    for member_idx in range(ensemble_size):
        _torch.manual_seed(base_seed + member_idx)
        member_outdir = outdir / f"member_{member_idx:02d}"
        member_outdir.mkdir(parents=True, exist_ok=True)

        dataset_train = None
        dataset_validation = None
        train_dataloader = None
        val_dataloader = None
        trainer = None
        model = None
        try:
            model = _ParametricLightningModule.init_from_config(model_config)
            (
                dataset_train,
                dataset_validation,
                train_dataloader,
                val_dataloader,
            ) = _build_dataloaders(data_config, learning_config, model)

            trainer = _pl.Trainer(
                callbacks=_create_parametric_callbacks(learning_config),
                default_root_dir=member_outdir,
                **learning_config["trainer"],
            )
            with _filter_warnings("ignore", category=_PossibleUserWarning):
                trainer.fit(
                    model,
                    train_dataloader,
                    val_dataloader,
                    **learning_config.get("trainer_fit_kwargs", {}),
                )

            checkpoint_callback = trainer.checkpoint_callback
            best_checkpoint = (
                checkpoint_callback.best_model_path
                if isinstance(checkpoint_callback, _ModelCheckpoint)
                else ""
            )
            checkpoint_paths.append(
                _stabilize_checkpoint_path(best_checkpoint, member_outdir)
            )
        finally:
            if dataset_train is not None:
                dataset_train.teardown()
            if dataset_validation is not None:
                dataset_validation.teardown()
            del val_dataloader
            del train_dataloader
            del trainer
            del model
            _gc.collect()
            _clear_device_cache(device)

    return checkpoint_paths
