# File: full.py
# Created Date: Tuesday March 26th 2024
# Author: Enrico Schifano (eraz1997@live.it)

import json as _json
from copy import deepcopy as _deepcopy
from pathlib import Path as _Path
from time import time as _time
from typing import List as _List
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from typing import Union as _Union
from warnings import warn as _warn

import matplotlib.pyplot as _plt
import numpy as _np
import pytorch_lightning as _pl
import torch as _torch
from pytorch_lightning.utilities.warnings import (
    PossibleUserWarning as _PossibleUserWarning,
)
from torch.utils.data import DataLoader as _DataLoader

from nam.data import AbstractDataset as _AbstractDataset
from nam.data import ConcatDataset as _ConcatDataset
from nam.data import Split as _Split
from nam.data import apply_joint_dataset_hooks as _apply_joint_dataset_hooks
from nam.data import get_joint_dataset_hooks as _get_joint_dataset_hooks
from nam.data import init_dataset as _init_dataset
from nam.train import core as _core
from nam.train import lightning_module as _lightning_module
from nam.util import filter_warnings as _filter_warnings

_torch.manual_seed(0)

_PRIMARY_VALIDATION_NAME = "unseen_audio_seen_param"
_DEPLOYMENT_VALIDATION_NAME = "unseen_audio_unseen_param"
_WINDOW_OVERRIDE_KEYS = frozenset(
    ("start_samples", "stop_samples", "start_seconds", "stop_seconds", "ny")
)


def _handshake_datasets(model, *datasets: _AbstractDataset) -> None:
    for dataset in datasets:
        dataset.handshake(model.net)
        model.net.handshake(dataset)


def _rms(x: _Union[_np.ndarray, _torch.Tensor]) -> float:
    if isinstance(x, _np.ndarray):
        return _np.sqrt(_np.mean(_np.square(x)))
    elif isinstance(x, _torch.Tensor):
        return _torch.sqrt(_torch.mean(_torch.square(x))).item()
    else:
        raise TypeError(type(x))


def _plot(
    model,
    ds,
    savefig=None,
    show=True,
    window_start: _Optional[int] = None,
    window_end: _Optional[int] = None,
):
    try:
        from nam.models.parametric._dataset import (
            ParametricDataset as _ParametricDataset,
            ParametricConcatDataset as _ParametricConcatDataset,
        )
    except ImportError:  # pragma: no cover
        _ParametricDataset = None  # type: ignore[assignment, misc]
        _ParametricConcatDataset = None  # type: ignore[assignment, misc]

    if isinstance(ds, _ConcatDataset) or (
        _ParametricConcatDataset is not None
        and isinstance(ds, _ParametricConcatDataset)
    ):

        def extend_savefig(i, savefig):
            if savefig is None:
                return None
            savefig = _Path(savefig)
            extension = savefig.name.split(".")[-1]
            stem = savefig.name[: -len(extension) - 1]
            return _Path(savefig.parent, f"{stem}_{i}.{extension}")

        for i, ds_i in enumerate(ds.datasets):
            _plot(
                model,
                ds_i,
                savefig=extend_savefig(i, savefig),
                show=show and i == len(ds.datasets) - 1,
                window_start=window_start,
                window_end=window_end,
            )
        return

    if _ParametricDataset is not None and isinstance(ds, _ParametricDataset):
        _net = model.net
        with _torch.no_grad():
            inner = ds._inner
            x = inner.x
            tx = len(x) / 48_000
            print(f"Run [nominal params] (t={tx:.2f})")
            t0 = _time()
            output = _net._at_nominal_settings(x).cpu().numpy()
            t1 = _time()
            try:
                rt = f"{tx / (t1 - t0):.2f}"
            except ZeroDivisionError:
                rt = "???"
            print(f"Took {t1 - t0:.2f} ({rt}x)")
        _plot_arrays(
            output.flatten(),
            inner.y,
            savefig=savefig,
            show=show,
            window_start=window_start,
            window_end=window_end,
        )
        return

    with _torch.no_grad():
        tx = len(ds.x) / 48_000
        print(f"Run (t={tx:.2f})")
        t0 = _time()
        output = model(ds.x).cpu().numpy()
        t1 = _time()
        try:
            rt = f"{tx / (t1 - t0):.2f}"
        except ZeroDivisionError as e:
            rt = "???"
        print(f"Took {t1 - t0:.2f} ({rt}x)")

    if output.ndim == 2:
        for i in range(output.shape[0]):
            packed_savefig = None
            if savefig is not None:
                savefig = _Path(savefig)
                packed_savefig = savefig.with_name(
                    f"{savefig.stem}_packed_{i}{savefig.suffix}"
                )
            _plot_arrays(
                output[i],
                ds.y,
                savefig=packed_savefig,
                show=show and i == output.shape[0] - 1,
                window_start=window_start,
                window_end=window_end,
                title_prefix=f"Packed {i}: ",
            )
        return

    output = output.flatten()
    _plot_arrays(
        output,
        ds.y,
        savefig=savefig,
        show=show,
        window_start=window_start,
        window_end=window_end,
    )


def _plot_arrays(
    output,
    target,
    savefig=None,
    show=True,
    window_start: _Optional[int] = None,
    window_end: _Optional[int] = None,
    title_prefix: str = "",
):

    _plt.figure(figsize=(16, 5))
    _plt.plot(output[window_start:window_end], label="Prediction")
    _plt.plot(target[window_start:window_end], linestyle="--", label="Target")
    nrmse = _rms(_torch.Tensor(output) - target) / _rms(target)
    esr = nrmse**2
    _plt.title(f"{title_prefix}ESR={esr:.3f}")
    _plt.legend()
    if savefig is not None:
        _plt.savefig(savefig)
    if show:
        _plt.show()
    _plt.close()


def _strip_split_window(entry: dict) -> dict:
    return {k: _deepcopy(v) for k, v in entry.items() if k not in _WINDOW_OVERRIDE_KEYS}


def _apply_eval_window(entries, window: dict):
    def apply(entry: dict) -> dict:
        # Reuse the same capture metadata/path/params, but swap in a different
        # time window so one physical file can serve multiple evaluation buckets.
        out = _strip_split_window(entry)
        out.update(_deepcopy(window))
        out.setdefault("ny", None)
        return out

    if isinstance(entries, list):
        return [apply(entry) for entry in entries]
    if isinstance(entries, dict):
        return apply(entries)
    raise TypeError(f"Unsupported parametric split type: {type(entries)}")


def _build_eval_config(data_config: dict, entries, window: dict) -> dict:
    return {
        "type": data_config.get("type", "dataset"),
        "common": _deepcopy(data_config.get("common", {})),
        _Split.VALIDATION.value: _apply_eval_window(entries, window),
    }


def _init_validation_datasets(
    data_config: dict,
) -> _List[_Tuple[str, _AbstractDataset]]:
    primary_validation = _init_dataset(data_config, _Split.VALIDATION)
    datasets = [("validation", primary_validation)]
    if data_config.get("type") != "parametric":
        return datasets

    has_verification = "verification" in data_config
    has_windows = "verification_windows" in data_config
    if has_verification != has_windows:
        raise ValueError(
            "Parametric verification requires both 'verification' and "
            "'verification_windows' in the data config."
        )
    if not has_verification:
        return datasets

    datasets[0] = (_PRIMARY_VALIDATION_NAME, primary_validation)
    windows = data_config["verification_windows"]
    missing_windows = [
        key for key in ("seen_audio", "unseen_audio") if key not in windows
    ]
    if missing_windows:
        raise ValueError(
            "Parametric verification_windows is missing required keys: "
            + ", ".join(missing_windows)
        )

    datasets.extend(
        [
            (
                "seen_audio_seen_param",
                _init_dataset(
                    # Build an eval-only view over the train captures so we can score
                    # "seen audio, seen param" separately from the held-out tail.
                    _build_eval_config(data_config, data_config[_Split.TRAIN.value], windows["seen_audio"]),
                    _Split.VALIDATION,
                ),
            ),
            (
                "seen_audio_unseen_param",
                _init_dataset(
                    # Verification captures are held-out parameter settings; pairing
                    # them with the seen-audio window isolates parametric generalization.
                    _build_eval_config(data_config, data_config["verification"], windows["seen_audio"]),
                    _Split.VALIDATION,
                ),
            ),
            (
                "unseen_audio_unseen_param",
                _init_dataset(
                    # Same held-out parameter settings, but now scored on the unseen
                    # audio window to approximate the full deployment case.
                    _build_eval_config(data_config, data_config["verification"], windows["unseen_audio"]),
                    _Split.VALIDATION,
                ),
            ),
        ]
    )
    return datasets


def _get_threshold_esr_monitor(
    learning_config: dict, validation_names: _List[str]
) -> str:
    if "threshold_esr_monitor" in learning_config:
        return learning_config["threshold_esr_monitor"]

    checkpoint_monitor = learning_config.get("checkpoint_monitor")
    if checkpoint_monitor == "val_loss":
        return "ESR"
    if isinstance(checkpoint_monitor, str) and checkpoint_monitor.startswith("val_loss_"):
        return f"ESR_{checkpoint_monitor.removeprefix('val_loss_')}"
    if _DEPLOYMENT_VALIDATION_NAME in validation_names:
        return f"ESR_{_DEPLOYMENT_VALIDATION_NAME}"
    return "ESR"


def _get_checkpoint_monitor(
    learning_config: dict, validation_names: _List[str]
) -> str:
    checkpoint_monitor = learning_config.get("checkpoint_monitor")
    if checkpoint_monitor is not None:
        return checkpoint_monitor
    if _DEPLOYMENT_VALIDATION_NAME in validation_names:
        return f"val_loss_{_DEPLOYMENT_VALIDATION_NAME}"
    return "val_loss"


def _get_checkpoint_metric_suffix(checkpoint_monitor: str) -> _Optional[str]:
    for prefix in ("val_loss_", "ESR_", "MSE_", "MRSTFT_"):
        if checkpoint_monitor.startswith(prefix):
            return checkpoint_monitor.removeprefix(prefix)
    return None


def _get_best_checkpoint_filename(checkpoint_monitor: str) -> str:
    suffix = _get_checkpoint_metric_suffix(checkpoint_monitor)
    if suffix is None:
        return "{epoch:04d}_{step}_{ESR:.3e}_{MSE:.3e}"
    return (
        f"{{epoch:04d}}_{{step}}_{{ESR_{suffix}:.3e}}_{{MSE_{suffix}:.3e}}"
    )


def _create_callbacks(
    learning_config,
    packed: bool = False,
    threshold_esr: _Optional[float] = None,
    validation_names: _Optional[_List[str]] = None,
):
    """
    Checkpointing, essentially
    """
    # Checkpoints should be run every time the validation check is run.
    # So base it off of learning_config["trainer"]["val_check_interval"] if it's there.
    validate_inside_epoch = "val_check_interval" in learning_config["trainer"]
    if validate_inside_epoch:
        kwargs = {
            "every_n_train_steps": learning_config["trainer"]["val_check_interval"]
        }
    else:
        kwargs = {
            "every_n_epochs": learning_config["trainer"].get(
                "check_val_every_n_epoch", 1
            )
        }

    checkpoint_monitor = _get_checkpoint_monitor(
        learning_config, validation_names or ["validation"]
    )

    checkpoint_best = _core._ModelCheckpoint(
        filename=_get_best_checkpoint_filename(checkpoint_monitor),
        save_top_k=3,
        monitor=checkpoint_monitor,
        **kwargs,
    )

    # return [checkpoint_best, checkpoint_last]
    # The last epoch that was finished.
    checkpoint_epoch = _core._ModelCheckpoint(
        filename="checkpoint_epoch_{epoch:04d}", every_n_epochs=1
    )
    callbacks = [checkpoint_best]
    if packed:
        callbacks.extend(
            [
                _lightning_module.PackedBestCheckpoint(),
                _lightning_module.PackedMaskCallback(),
            ]
        )
    if threshold_esr is not None:
        callbacks.append(
            _core._ValidationStopping(
                monitor=_get_threshold_esr_monitor(
                    learning_config, validation_names or ["validation"]
                ),
                stopping_threshold=threshold_esr,
            )
        )
    if not validate_inside_epoch:
        callbacks.append(checkpoint_epoch)
        return callbacks
    else:
        # The last validation pass, whether at the end of an epoch or not
        checkpoint_last = _core._ModelCheckpoint(
            filename="checkpoint_last_{epoch:04d}_{step}", **kwargs
        )
        callbacks.extend([checkpoint_last, checkpoint_epoch])
        return callbacks


def main(
    data_config,
    model_config,
    learning_config,
    outdir: _Path,
    no_show: bool = False,
    make_plots=True,
    save_plot: _Optional[bool] = None,
):
    if not outdir.exists():
        raise RuntimeError(f"No output location found at {outdir}")
    # Write
    for basename, config in (
        ("data", data_config),
        ("model", model_config),
        ("learning", learning_config),
    ):
        with open(_Path(outdir, f"config_{basename}.json"), "w") as fp:
            _json.dump(config, fp, indent=4)

    _net_name = model_config["net"]["name"]
    is_packed = _net_name == "PackedWaveNet"
    is_parametric = _net_name == "ParametricWaveNet"
    if is_packed:
        lightning_cls = _lightning_module.PackedLightningModule
    elif is_parametric:
        import nam.models.parametric  # noqa: F401 — registers model + dataset
        from nam.train.parametric import (
            ParametricLightningModule as _ParametricLightningModule,
        )
        lightning_cls = _ParametricLightningModule
    else:
        lightning_cls = _lightning_module.LightningModule
    model = lightning_cls.init_from_config(model_config)
    # Add receptive field to data config:
    data_config["common"] = data_config.get("common", {})
    if "nx" in data_config["common"]:
        _warn(
            f"Overriding data nx={data_config['common']['nx']} with model required {model.net.receptive_field}"
        )
    data_config["common"]["nx"] = model.net.receptive_field

    dataset_train = _init_dataset(data_config, _Split.TRAIN)
    validation_sets = _init_validation_datasets(data_config)
    validation_names = [name for name, _ in validation_sets]
    validation_datasets = [dataset for _, dataset in validation_sets]
    dataset_validation = validation_datasets[0]
    _apply_joint_dataset_hooks(
        dataset_train=dataset_train,
        dataset_validation=validation_datasets,
        hooks=_get_joint_dataset_hooks(data_config.get("joint", [])),
    )
    model.net.sample_rate = dataset_train.sample_rate

    # Perform handshakes:
    _handshake_datasets(model, dataset_train, *validation_datasets)
    if hasattr(model, "set_validation_loader_names"):
        model.set_validation_loader_names(validation_names)

    train_dataloader = _DataLoader(dataset_train, **learning_config["train_dataloader"])
    val_dataloader = [
        _DataLoader(dataset_validation_i, **learning_config["val_dataloader"])
        for dataset_validation_i in validation_datasets
    ]
    # Preserve Lightning's original single-loader shape for legacy configs so the
    # rest of the training path behaves exactly as before when verification is absent.
    if len(val_dataloader) == 1:
        val_dataloader = val_dataloader[0]

    callbacks = _create_callbacks(
        learning_config,
        packed=is_packed,
        threshold_esr=learning_config.get("threshold_esr"),
        validation_names=validation_names,
    )
    packed_best_callback = next(
        (c for c in callbacks if isinstance(c, _lightning_module.PackedBestCheckpoint)),
        None,
    )
    trainer = _pl.Trainer(
        callbacks=callbacks,
        default_root_dir=outdir,
        **learning_config["trainer"],
    )

    try:
        with _filter_warnings("ignore", category=_PossibleUserWarning):
            trainer.fit(
                model,
                train_dataloader,
                val_dataloader,
                **learning_config.get("trainer_fit_kwargs", {}),
            )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        # Always try to export a model, even if training was interrupted
        # Go to best checkpoint
        best_checkpoint = trainer.checkpoint_callback.best_model_path
        if best_checkpoint != "":
            model = lightning_cls.load_from_checkpoint(
                trainer.checkpoint_callback.best_model_path,
                **lightning_cls.parse_config(model_config),
            )
        model.cpu()
        model.eval()
        model.net.sample_rate = dataset_train.sample_rate
        if hasattr(model, "set_validation_loader_names"):
            model.set_validation_loader_names(validation_names)
        _handshake_datasets(model, dataset_train, *validation_datasets)
        should_save_plot = make_plots if save_plot is None else save_plot
        if make_plots or should_save_plot:
            _plot(
                model,
                dataset_validation,
                savefig=_Path(outdir, "comparison.png") if should_save_plot else None,
                window_start=100_000,
                window_end=110_000,
                show=make_plots and not no_show,
            )
        # Export!
        if is_packed:
            checkpoint_paths = (
                None
                if packed_best_callback is None
                or not packed_best_callback.checkpoint_paths
                else packed_best_callback.checkpoint_paths
            )
            model.net.export_container(
                outdir,
                checkpoint_paths_by_submodel=checkpoint_paths,
            )
        else:
            model.net.export(outdir)

        # Tear down the datasets
        train_dataloader.dataset.teardown()
        for dataset_validation_i in validation_datasets:
            dataset_validation_i.teardown()
