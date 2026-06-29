"""
Active-learning ensemble training adapted from PANAMA.

The serial, device-agnostic ensemble flow follows PANAMA (Parametric Active-learning
for Neural Amp Modeling Assistance), arXiv:2509.26564v1, adapted to this repo's
parametric NAM training stack and runtime-selected device handling.
"""

import contextlib as _contextlib
import gc as _gc
import importlib as _importlib
import math as _math
import shutil as _shutil
from copy import deepcopy as _deepcopy
from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path
from collections.abc import Sequence as _Sequence
from typing import Any as _Any
from typing import cast as _cast
from warnings import warn as _warn

import pytorch_lightning as _pl
import torch as _torch
from lightning_fabric.utilities.warnings import PossibleUserWarning as _PossibleUserWarning
from pytorch_lightning.callbacks import ModelCheckpoint as _ModelCheckpoint
from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data import TensorDataset as _TensorDataset

from nam.data import ConcatDataset as _ConcatDataset
from nam.data import Split as _Split
from nam.data import WavInfo as _WavInfo
from nam.data import apply_joint_dataset_hooks as _apply_joint_dataset_hooks
from nam.data import get_joint_dataset_hooks as _get_joint_dataset_hooks
from nam.data import init_dataset as _init_dataset
from nam.data import wav_to_tensor as _wav_to_tensor
from nam.models.parametric import assemble_raw_params as _assemble_raw_params
from nam.models.parametric import data_config_from_model as _data_config_from_model
from nam.models.parametric import split_param_indices as _split_param_indices
from nam.models.parametric import switch_combinations as _switch_combinations
from nam.models.parametric import ParamSpec as _ParamSpec
from nam.models.parametric._dataset import _coerce_param_specs
from nam.train.full import _handshake_datasets
from nam.train.parametric import _ParametricLightningModule
from nam.train.parametric import _create_parametric_callbacks
from nam.train.parametric import _iter_inner_datasets
from nam.train.parametric import _make_parametric_dataloader
from nam.util import filter_warnings as _filter_warnings

__all__ = [
    "DisagreementCandidate",
    "find_disagreement_settings",
    "train_ensemble",
]


@_dataclass(frozen=True)
class DisagreementCandidate:
    raw_params: _torch.Tensor
    switch_combo: tuple[int, ...]
    score: float


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


def _param_specs_from_model_config(model_config: dict) -> tuple[_ParamSpec, ...]:
    try:
        raw_specs = model_config["net"]["config"]["params"]
    except KeyError as exc:
        raise ValueError(
            "Model config must define net.config.params for disagreement search"
        ) from exc
    return tuple(_coerce_param_specs(raw_specs))


def _validate_g_opt_args(
    *,
    checkpoint_paths: _Sequence[_Path],
    num_restarts: int,
    num_steps: int,
    g_opt_ny: int,
    g_opt_batch_size: int,
    lr: float,
    z_init_scale: float,
) -> None:
    if len(checkpoint_paths) == 0:
        raise ValueError("checkpoint_paths must contain at least one checkpoint")
    if num_restarts <= 0:
        raise ValueError(f"num_restarts must be positive; got {num_restarts}")
    if num_steps < 0:
        raise ValueError(f"num_steps must be non-negative; got {num_steps}")
    if g_opt_ny <= 0:
        raise ValueError(f"g_opt_ny must be positive; got {g_opt_ny}")
    if g_opt_batch_size <= 0:
        raise ValueError(
            f"g_opt_batch_size must be positive; got {g_opt_batch_size}"
        )
    if not _math.isfinite(lr) or lr <= 0.0:
        raise ValueError(f"lr must be a positive finite number; got {lr}")
    if not _math.isfinite(z_init_scale) or z_init_scale <= 0.0:
        raise ValueError(
            f"z_init_scale must be a positive finite number; got {z_init_scale}"
        )


def _load_disagreement_members(
    checkpoint_paths: _Sequence[_Path],
    model_config: dict,
    device: _torch.device,
) -> list[_torch.nn.Module]:
    members: list[_torch.nn.Module] = []
    for checkpoint_path in checkpoint_paths:
        module = _ParametricLightningModule.load_from_checkpoint(
            str(checkpoint_path),
            map_location=device,
            **_ParametricLightningModule.parse_config(model_config),
        )
        member = module.net.to(device)
        member.requires_grad_(False)
        # Always eval(): find_disagreement_settings runs the g-opt loop under
        # cuDNN-disabled on cuda so the RNN backward works without train() mode. train()
        # would otherwise divert ConcatLSTM through its truncated-BPTT path (detached
        # burn-in + per-chunk-detached hidden state), silently changing the gradient to z
        # on cuda relative to cpu/mps.
        member.eval()
        members.append(member)
    return members


def _build_g_opt_batches(
    g_opt_input_wav: str | _Path,
    *,
    g_opt_ny: int,
    g_opt_batch_size: int,
    receptive_field: int,
    device: _torch.device,
) -> tuple[list[_torch.Tensor], int]:
    if g_opt_ny <= receptive_field:
        raise ValueError(
            f"g_opt_ny must be greater than receptive_field={receptive_field}; got {g_opt_ny}"
        )
    signal, wavinfo = _cast(
        tuple[_torch.Tensor, _WavInfo],
        _wav_to_tensor(g_opt_input_wav, info=True),
    )
    if signal.ndim != 1:
        raise ValueError(
            f"Expected mono g-opt input wav to load as shape (L,); got {tuple(signal.shape)}"
        )
    if signal.shape[0] < g_opt_ny:
        raise ValueError(
            f"g-opt input wav must contain at least g_opt_ny={g_opt_ny} samples; "
            f"got {signal.shape[0]}"
        )

    hop = g_opt_ny - receptive_field
    windows = signal.unfold(0, g_opt_ny, hop).contiguous()
    if windows.shape[0] == 0:
        raise RuntimeError("Failed to construct any g-opt windows from the input wav")
    # receptive_field is 1 for the LSTM, so windows are effectively non-overlapping and
    # each is processed from the member's learned initial hidden state (no burn-in warmup
    # at g-opt time, unlike training). The first samples of every window are therefore
    # transient; acceptable for a coarse disagreement proxy.

    loader = _DataLoader(
        _TensorDataset(windows),
        batch_size=g_opt_batch_size,
        shuffle=False,
    )
    batches = [batch[0].to(device) for batch in loader]
    if len(batches) == 0:
        raise RuntimeError("Failed to construct any g-opt batches from the input wav")
    return batches, wavinfo.rate


def _build_mel_transforms(
    *,
    use_mel: bool,
    sample_rate: int,
    device: _torch.device,
):
    if not use_mel:
        return ()
    try:
        _torchaudio = _importlib.import_module("torchaudio")
    except ImportError as exc:
        raise RuntimeError(
            "use_mel=True requires torchaudio to be installed"
        ) from exc

    transforms = []
    for n_fft, hop_length, n_mels in (
        (512, 128, 64),
        (1024, 256, 80),
        (2048, 512, 128),
    ):
        transforms.append(
            _torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=n_fft,
                hop_length=hop_length,
                n_mels=n_mels,
                power=2.0,
                center=True,
            ).to(device)
        )
    return tuple(transforms)


def _disagreement_score(
    outputs: _torch.Tensor,
    mel_transforms,
) -> _torch.Tensor:
    score = outputs.var(dim=0, unbiased=False).mean()
    if len(mel_transforms) == 0:
        return score

    num_members, batch_size, output_len = outputs.shape
    flattened = outputs.reshape(num_members * batch_size, output_len)
    mel_score = _torch.zeros((), device=outputs.device, dtype=outputs.dtype)
    for transform in mel_transforms:
        mel = transform(flattened).reshape(num_members, batch_size, -1)
        mel_score = mel_score + mel.var(dim=0, unbiased=False).mean()
    return score + mel_score / len(mel_transforms)


def _evaluate_disagreement(
    members: _Sequence[_torch.nn.Module],
    x_batch: _torch.Tensor,
    raw_params: _torch.Tensor,
    mel_transforms,
) -> _torch.Tensor:
    outputs = _torch.stack([member(x_batch, raw_params) for member in members], dim=0)
    return _disagreement_score(outputs, mel_transforms)


def _final_disagreement_score(
    members: _Sequence[_torch.nn.Module],
    g_opt_batches: _Sequence[_torch.Tensor],
    raw_params: _torch.Tensor,
    mel_transforms,
) -> float:
    # Rank a candidate on the mean disagreement over the *whole* g-opt signal (no_grad),
    # not the single last training chunk: Task 6's clustering + global top-N selection
    # rides on these scores, so a one-batch estimate would make the ranking noisy.
    with _torch.no_grad():
        total = 0.0
        for batch in g_opt_batches:
            total += float(
                _evaluate_disagreement(members, batch, raw_params, mel_transforms)
                .detach()
                .cpu()
            )
    score = total / len(g_opt_batches)
    if not _math.isfinite(score):
        raise RuntimeError(f"Non-finite disagreement score encountered: {score}")
    return score


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


def find_disagreement_settings(
    checkpoint_paths: _Sequence[_Path],
    model_config: dict,
    *,
    g_opt_input_wav: str | _Path,
    num_restarts: int = 8,
    num_steps: int = 200,
    g_opt_ny: int = 32768,
    g_opt_batch_size: int = 16,
    lr: float = 0.05,
    z_init_scale: float = 3.0,
    use_mel: bool = False,
    seed: int = 0,
) -> list[DisagreementCandidate]:
    """
    Find high-disagreement control settings with a PANAMA-style query-by-committee search.

    ``z_init_scale`` is the std of the Gaussian latent inits: each restart draws
    ``z ~ N(0, z_init_scale^2)`` and the continuous params are ``min + (max-min)*sigmoid(z)``
    (see ``assemble_raw_params``). A larger scale biases inits toward the saturated
    extremes of sigmoid (often the high-disagreement knob extremes) at the cost of more
    restarts landing in vanishing-gradient regions; the default echoes PANAMA's spread.
    """
    checkpoint_paths = tuple(_Path(path) for path in checkpoint_paths)
    _validate_g_opt_args(
        checkpoint_paths=checkpoint_paths,
        num_restarts=num_restarts,
        num_steps=num_steps,
        g_opt_ny=g_opt_ny,
        g_opt_batch_size=g_opt_batch_size,
        lr=lr,
        z_init_scale=z_init_scale,
    )
    specs = _param_specs_from_model_config(model_config)
    continuous_idx, _, _ = _split_param_indices(specs)

    device = _resolve_device()
    members = _load_disagreement_members(checkpoint_paths, model_config, device)
    if len(members) == 0:
        return []

    receptive_fields = {int(getattr(member, "receptive_field")) for member in members}
    if len(receptive_fields) != 1:
        raise RuntimeError(
            f"Expected all ensemble members to share one receptive field; got {receptive_fields}"
        )
    receptive_field = next(iter(receptive_fields))
    g_opt_batches, sample_rate = _build_g_opt_batches(
        g_opt_input_wav,
        g_opt_ny=g_opt_ny,
        g_opt_batch_size=g_opt_batch_size,
        receptive_field=receptive_field,
        device=device,
    )
    mel_transforms = _build_mel_transforms(
        use_mel=use_mel,
        sample_rate=sample_rate,
        device=device,
    )

    generator = _torch.Generator()
    generator.manual_seed(seed)
    n_cont = len(continuous_idx)
    # Disable cuDNN only on cuda so the RNN backward runs with the members in eval() mode
    # (see _load_disagreement_members); cpu/mps need nothing. This keeps the forward and
    # the gradient to z identical across devices.
    cudnn_ctx = (
        _torch.backends.cudnn.flags(enabled=False)
        if device.type == "cuda"
        else _contextlib.nullcontext()
    )
    candidates: list[DisagreementCandidate] = []
    try:
        with cudnn_ctx:
            for switch_combo in _switch_combinations(specs):
                if n_cont == 0:
                    # No continuous latents to ascend: the setting is fully determined by
                    # the switch combo, so every restart/step would reproduce the identical
                    # candidate. Evaluate once and move on.
                    final_raw_params = _assemble_raw_params(
                        _torch.zeros((0,), dtype=_torch.float32, device=device),
                        switch_combo,
                        specs,
                    )
                    candidates.append(
                        DisagreementCandidate(
                            raw_params=final_raw_params.detach().cpu(),
                            switch_combo=switch_combo,
                            score=_final_disagreement_score(
                                members, g_opt_batches, final_raw_params, mel_transforms
                            ),
                        )
                    )
                    continue
                for _ in range(num_restarts):
                    z = (
                        z_init_scale
                        * _torch.randn(
                            (n_cont,),
                            generator=generator,
                            dtype=_torch.float32,
                        )
                    ).to(device)
                    z.requires_grad_(True)
                    optimizer = _torch.optim.Adam([z], lr=lr)
                    for step in range(num_steps):
                        batch = g_opt_batches[step % len(g_opt_batches)]
                        raw_params = _assemble_raw_params(z, switch_combo, specs)
                        score = _evaluate_disagreement(
                            members, batch, raw_params, mel_transforms
                        )
                        optimizer.zero_grad(set_to_none=True)
                        (-score).backward()
                        optimizer.step()

                    final_raw_params = _assemble_raw_params(z, switch_combo, specs)
                    candidates.append(
                        DisagreementCandidate(
                            raw_params=final_raw_params.detach().cpu(),
                            switch_combo=switch_combo,
                            score=_final_disagreement_score(
                                members, g_opt_batches, final_raw_params, mel_transforms
                            ),
                        )
                    )
    finally:
        del mel_transforms
        del g_opt_batches
        del members
        _gc.collect()
        _clear_device_cache(device)

    return candidates
