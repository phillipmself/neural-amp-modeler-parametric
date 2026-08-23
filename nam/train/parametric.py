import json as _json
from dataclasses import dataclass as _dataclass
from dataclasses import fields as _fields
from pathlib import Path as _Path
from time import time as _time
from collections.abc import Sequence as _Sequence
from typing import Optional as _Optional
from typing import Tuple as _Tuple
from warnings import warn as _warn

import librosa as _librosa
import matplotlib.pyplot as _plt
import pytorch_lightning as _pl
import torch as _torch
from lightning_fabric.utilities.warnings import (
    PossibleUserWarning as _PossibleUserWarning,
)
from pytorch_lightning.callbacks import ModelCheckpoint as _ModelCheckpoint
from torch.utils.data import DataLoader as _DataLoader
from torch.utils.data import Sampler as _Sampler

from nam.data import ConcatDataset as _ConcatDataset
from nam.data import Dataset as _Dataset
from nam.data import Split as _Split
from nam.data import apply_joint_dataset_hooks as _apply_joint_dataset_hooks
from nam.data import get_joint_dataset_hooks as _get_joint_dataset_hooks
from nam.data import init_dataset as _init_dataset
from nam.models.exportable import Exportable as _Exportable
from nam.models.parametric import HyperWaveNet as _HyperWaveNet
from nam.models.parametric import ParametricDataset as _ParametricDataset
from nam.models.parametric import ParametricNet as _ParametricNet
from nam.models.parametric._anchors import anchor_output as _anchor_output
from nam.models.parametric._anchors import AnchorWindow as _AnchorWindow
from nam.models.parametric._anchors import anchor_window as _anchor_window
from nam.models.parametric._anchors import (
    as_held_trajectory as _as_held_trajectory,
)
from nam.models.parametric._anchors import (
    sample_param_trajectories as _sample_param_trajectories,
)
from nam.models.parametric._anchors import sample_raw_params as _sample_raw_params
from nam.models.parametric._splice import SpliceSource as _SpliceSource
from nam.models.parametric._splice import SpliceWindow as _SpliceWindow
from nam.models.parametric._splice import splice_output as _splice_output
from nam.models.parametric._splice import splice_window as _splice_window
from nam.models.parametric import bake as _bake
from nam.models.parametric import data_config_from_model as _data_config_from_model
from nam.models.parametric import export_parametric as _export_parametric
from nam.models.parametric import (
    output_scale_from_datasets as _output_scale_from_datasets,
)
from nam.train.core import _ValidationStopping
from nam.train.full import _handshake_datasets
from nam.train.full import _rms as _rms
from nam.train.lightning_module import LightningModule as _LightningModule
from nam.train.lightning_module import LossConfig as _LossConfig
from nam.train.lightning_module import _LossItem
from nam.util import filter_warnings as _filter_warnings

# NB: the global RNG seed is set once by `nam.train.full` (imported above); no need to
# reseed here.


def _iter_inner_datasets(dataset) -> tuple[_Dataset, ...]:
    if isinstance(dataset, _Dataset):
        return (dataset,)
    if isinstance(dataset, _ParametricDataset):
        return _iter_inner_datasets(dataset.dataset)
    if isinstance(dataset, _ConcatDataset):
        inner = []
        for child in dataset.datasets:
            inner.extend(_iter_inner_datasets(child))
        return tuple(inner)
    raise TypeError(
        "Expected a Dataset, ParametricDataset, or ConcatDataset; "
        f"got {type(dataset).__name__}"
    )


# Default-on. Each capture pairs one control setting with many audio windows, and the train
# ConcatDataset lays the captures out in contiguous index ranges. HyperWaveNet._run_conditioned
# groups a batch by unique control setting and runs one functional_call per group, so a batch
# that mixes captures fans out into several tiny GPU passes. Keeping every batch inside one
# capture collapses that to a single full-batch functional_call per step.
_CAPTURE_GROUPED_KEY = "capture_grouped_batches"
# Metrics are tracked per data bucket: the training split is audio and control
# settings the model fit on; the validation split is held out on both axes.
_TRAIN_BUCKET = "seen_audio_seen_params"
_VALIDATION_BUCKET = "unseen_audio_unseen_params"
# Means reduced as sample-weighted averages across the epoch. ESR is handled
# separately because it is a ratio of energies, not a mean.
_BUCKET_MEAN_METRICS = (
    "MSE",
    "MRSTFT",
    "Silence Anchor",
    "Silence Anchor Ramp",
    "Param Step",
)


class _MelSpectrogram(_torch.nn.Module):
    """Pure-torch mel spectrogram equivalent to ``torchaudio.transforms.MelSpectrogram``
    with its default HTK, unnormalized filterbank and centered reflection-padded STFT.

    The PANAMA mel objectives previously depended on torchaudio, whose prebuilt wheel loads
    a compiled extension that must match the installed torch ABI exactly; a mismatched pair
    fails to ``dlopen`` at runtime. This reimplements the only piece those objectives use
    (an STFT magnitude raised to ``power`` and projected onto a mel filterbank) using
    ``torch.stft`` plus a librosa-built filterbank, so training needs no torchaudio.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        power: float = 1.0,
    ):
        super().__init__()
        self._n_fft = int(n_fft)
        self._win_length = int(win_length)
        self._hop_length = int(hop_length)
        self._power = float(power)
        self.register_buffer("_window", _torch.hann_window(self._win_length))
        # (n_mels, n_freqs); htk + norm=None matches torchaudio's melscale_fbanks defaults.
        filterbank = _librosa.filters.mel(
            sr=sample_rate, n_fft=self._n_fft, n_mels=n_mels, htk=True, norm=None
        )
        self.register_buffer(
            "_mel_fb", _torch.tensor(filterbank, dtype=_torch.float32)
        )

    def forward(self, waveform: _torch.Tensor) -> _torch.Tensor:
        original = waveform.shape
        spec = _torch.stft(
            waveform.reshape(-1, original[-1]),
            n_fft=self._n_fft,
            hop_length=self._hop_length,
            win_length=self._win_length,
            window=self._window,
            center=True,
            pad_mode="reflect",
            normalized=False,
            return_complex=True,
        )
        spectrogram = spec.abs().pow(self._power)  # (B, n_freqs, n_frames)
        mel = _torch.matmul(self._mel_fb, spectrogram)  # (B, n_mels, n_frames)
        return mel.reshape(*original[:-1], mel.shape[-2], mel.shape[-1])


class _CaptureBatchSampler(_Sampler):
    """
    Yield batches whose sample indices all fall inside one capture's contiguous range, so a
    HyperWaveNet step sees a single control setting and runs one functional_call over the
    whole batch instead of one per unique setting.

    Trade-off vs. global shuffling: rows are shuffled *within* a capture and the batch order
    is shuffled across captures, but a batch never mixes captures. Uses the global torch RNG
    (seeded once by ``nam.train.full``) so the order is reproducible yet varies per epoch.

    Since every batch is capture-pure, each ``__iter__`` call also records which capture
    every yielded batch belongs to (``last_batch_captures``), so a caller holding a reference
    to this sampler can recover a batch's capture from its ``batch_idx`` -- including across
    epochs, where ``shuffle=True`` reorders both the rows and the batch list on every call.
    This is how ``_EpochMetrics`` groups batches by capture without re-deriving the mapping
    itself (see ``_ParametricLightningModule._capture_index``).
    """

    def __init__(
        self,
        capture_lengths: _Sequence[int],
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive; got {batch_size}")
        self._capture_lengths = tuple(int(length) for length in capture_lengths)
        self._batch_size = int(batch_size)
        self._shuffle = bool(shuffle)
        self._drop_last = bool(drop_last)
        # Populated by __iter__; index i is the capture index of the i-th batch of the most
        # recently started iteration (epoch).
        self.last_batch_captures: tuple[int, ...] = ()

    def _capture_offsets(self) -> tuple[int, ...]:
        offsets = []
        offset = 0
        for length in self._capture_lengths:
            offsets.append(offset)
            offset += length
        return tuple(offsets)

    def __iter__(self):
        batches: list[list[int]] = []
        batch_captures: list[int] = []
        for capture_idx, (offset, length) in enumerate(
            zip(self._capture_offsets(), self._capture_lengths)
        ):
            if length == 0:
                continue
            if self._shuffle:
                order = [offset + i for i in _torch.randperm(length).tolist()]
            else:
                order = list(range(offset, offset + length))
            for start in range(0, length, self._batch_size):
                batch = order[start : start + self._batch_size]
                if self._drop_last and len(batch) < self._batch_size:
                    continue
                batches.append(batch)
                batch_captures.append(capture_idx)
        if self._shuffle:
            permutation = _torch.randperm(len(batches)).tolist()
            batches = [batches[i] for i in permutation]
            batch_captures = [batch_captures[i] for i in permutation]
        # Set before yielding the first batch: everything above runs eagerly on the first
        # `next()` call, since it all precedes the `yield from` below.
        self.last_batch_captures = tuple(batch_captures)
        yield from batches

    def __len__(self) -> int:
        total = 0
        for length in self._capture_lengths:
            if self._drop_last:
                total += length // self._batch_size
            else:
                total += (length + self._batch_size - 1) // self._batch_size
        return total

    def capture_for_batch(self, batch_idx: int) -> int:
        """Which capture the ``batch_idx``-th batch of the current iteration came from."""
        return self.last_batch_captures[batch_idx]


def _iter_parametric_datasets(dataset) -> tuple[_ParametricDataset, ...]:
    """The captures a split is made of, each still carrying its control setting.

    `_iter_inner_datasets` unwraps one layer further and drops the params, which the
    splice source needs.
    """
    if isinstance(dataset, _ParametricDataset):
        return (dataset,)
    if isinstance(dataset, _ConcatDataset):
        found: list[_ParametricDataset] = []
        for child in dataset.datasets:
            found.extend(_iter_parametric_datasets(child))
        return tuple(found)
    raise ValueError(
        "Expected a ParametricDataset or ConcatDataset of ParametricDataset; "
        f"got {type(dataset).__name__}"
    )


def _capture_lengths(dataset) -> tuple[int, ...]:
    if isinstance(dataset, _ConcatDataset):
        return tuple(len(child) for child in dataset.datasets)
    return (len(dataset),)


def _make_parametric_dataloader(dataset, loader_config: dict) -> _DataLoader:
    loader_config = dict(loader_config)
    capture_grouped = loader_config.pop(_CAPTURE_GROUPED_KEY, True)
    if not capture_grouped:
        return _DataLoader(dataset, **loader_config)

    # batch_sampler is mutually exclusive with these DataLoader args, so the sampler owns them.
    batch_size = loader_config.pop("batch_size", 1)
    shuffle = loader_config.pop("shuffle", False)
    drop_last = loader_config.pop("drop_last", False)
    loader_config.pop("sampler", None)
    batch_sampler = _CaptureBatchSampler(
        _capture_lengths(dataset),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return _DataLoader(dataset, batch_sampler=batch_sampler, **loader_config)


def _enable_training_fast_paths(
    net: _ParametricNet,
    dataloaders: _Sequence[_DataLoader],
    compile_config: dict,
) -> None:
    """
    Turn on the HyperWaveNet step optimizations that are only valid during `fit`.

    `torch_compile` applies to any net that supports a compiled step (HyperWaveNet,
    ConcatWaveNet); nets that do not just warn and stay eager.

    The uniform-params promise is HyperWaveNet-only -- it is the one net that has to *find*
    the batch's grouping, and capture-grouped batching makes every batch one control setting
    so it can skip that search and the host syncs it costs. The promise only holds for these
    dataloaders, so `_release_training_fast_paths` takes it back once fitting is over and
    later calls (bake, export checks) go through the general path.
    """
    if isinstance(net, _HyperWaveNet):
        net.set_uniform_batch_params(
            all(
                isinstance(dataloader.batch_sampler, _CaptureBatchSampler)
                for dataloader in dataloaders
            )
        )
    compile_config = dict(compile_config)
    net.set_compiled(bool(compile_config.pop("enabled", False)), **compile_config)
    net.set_validate_params(False)


def _release_training_fast_paths(net: _ParametricNet) -> None:
    if isinstance(net, _HyperWaveNet):
        net.set_uniform_batch_params(False)
    net.set_compiled(False)
    net.set_validate_params(True)


def _get_parametric_net(model: _LightningModule) -> _ParametricNet:
    net = model.net
    if not isinstance(net, _ParametricNet):
        raise TypeError(
            "Parametric training expects model_config['net']['name'] to initialize "
            f"a ParametricNet, got {type(net).__name__}"
        )
    return net


class _EnergyAccumulator:
    """Pool metrics over a set of batches by summing energies, exactly as
    ``_EpochMetrics`` used to do over an entire split.

    ESR is summed as energies -- the squared error and squared target are summed
    separately and divided only at the end -- so a near-silent batch cannot divide
    by ~zero and poison the pool the way a per-batch ESR average does. MSE and
    MRSTFT/Mel are reduced as sample-weighted means over the same batches. This is
    the correct reduction *within* one capture (every window shares one control
    setting, so there is no cross-capture level mismatch to worry about); see
    ``_EpochMetrics`` for why pooling energies *across* captures is a defect.
    """

    def __init__(self) -> None:
        self._err_sq: _Optional[_torch.Tensor] = None
        self._tgt_sq: _Optional[_torch.Tensor] = None
        self._weighted: dict[str, _torch.Tensor] = {}
        self.rows = 0

    def update(
        self,
        preds: _torch.Tensor,
        targets: _torch.Tensor,
        means: dict[str, _torch.Tensor],
    ) -> None:
        preds = preds.detach()
        targets = targets.detach()
        err_sq = _torch.sum(_torch.square(preds - targets))
        tgt_sq = _torch.sum(_torch.square(targets))
        self._err_sq = err_sq if self._err_sq is None else self._err_sq + err_sq
        self._tgt_sq = tgt_sq if self._tgt_sq is None else self._tgt_sq + tgt_sq
        rows = targets.shape[0]
        self.rows += rows
        for key, value in means.items():
            # Equal-length windows make a row-count weighting exact for per-element
            # means like MSE and consistent for MRSTFT/Mel.
            contrib = value.detach() * rows
            self._weighted[key] = (
                contrib if key not in self._weighted else self._weighted[key] + contrib
            )

    def compute(self) -> dict[str, float]:
        if self.rows == 0:
            return {}
        tgt_sq = float(self._tgt_sq)
        out: dict[str, float] = {
            "ESR": float(self._err_sq) / tgt_sq if tgt_sq > 0.0 else float("inf")
        }
        for key, total in self._weighted.items():
            out[key] = float(total) / self.rows
        return out


class _EpochMetrics:
    """Accumulate per-batch metrics across an epoch and reduce them once at the end,
    using a two-level reduction across captures.

    A parametric split lays out several captures (one control setting each) back to
    back, and their output levels can differ by many dB. Pooling energies (or taking a
    sample-weighted mean) directly across ALL windows of ALL captures -- what this class
    used to do -- lets the loud captures swamp the denominator, so a quiet capture's own
    ESR barely moves the reported number even though it can be terrible. The fix keeps
    the original pooling *within* a capture (see ``_EnergyAccumulator``) -- that part was
    never wrong, a near-silent capture still can't divide by ~zero -- and then takes an
    equal-weight mean of the per-capture results, so every capture counts the same
    regardless of how loud it is.

    This requires knowing which capture each batch belongs to. Callers that can supply a
    ``capture_index`` to ``update`` get the two-level treatment. Callers that cannot (no
    ``_CaptureBatchSampler`` behind the dataloader -- see
    ``_ParametricLightningModule._capture_index``) fall back to the single-level pooling
    over the whole epoch, matching the old behavior exactly rather than silently mixing
    the two reductions.
    """

    def __init__(self) -> None:
        self._captures: dict[int, _EnergyAccumulator] = {}
        # Fallback accumulator, used only when capture identity is unavailable.
        self._ungrouped = _EnergyAccumulator()
        self._capture_grouped: _Optional[bool] = None

    def update(
        self,
        preds: _torch.Tensor,
        targets: _torch.Tensor,
        means: dict[str, _torch.Tensor],
        capture_index: _Optional[int] = None,
    ) -> None:
        grouped = capture_index is not None
        if self._capture_grouped is None:
            self._capture_grouped = grouped
        elif self._capture_grouped != grouped:
            raise RuntimeError(
                "_EpochMetrics received a capture_index on some updates but not others "
                "within the same epoch; capture identity must be available for either "
                "every batch or none of them"
            )
        if grouped:
            accumulator = self._captures.setdefault(capture_index, _EnergyAccumulator())
        else:
            accumulator = self._ungrouped
        accumulator.update(preds, targets, means)

    def compute(self) -> dict[str, float]:
        if not self._capture_grouped:
            return self._ungrouped.compute()
        per_capture = [
            accumulator.compute()
            for accumulator in self._captures.values()
            if accumulator.rows > 0
        ]
        if not per_capture:
            return {}
        keys: dict[str, None] = {}
        for metrics in per_capture:
            keys.update(dict.fromkeys(metrics))
        out: dict[str, float] = {}
        for key in keys:
            values = [metrics[key] for metrics in per_capture if key in metrics]
            out[key] = sum(values) / len(values)
        return out


def _bucket_means(loss_dict: dict) -> dict[str, _torch.Tensor]:
    return {
        key: loss_dict[key].value
        for key in _BUCKET_MEAN_METRICS
        if key in loss_dict and loss_dict[key].value is not None
    }


def _loss_term_from_config(cls, config: _Optional[dict], label: str):
    """Build a loss-term dataclass from its JSON block, or ``None`` when it is absent.

    Both the accepted keys and their coercion come off the dataclass, so adding or
    dropping a field is a one-line edit with nothing else to keep in step. This holds
    only while every field is annotated with a plain callable type: an ``Optional[...]``
    annotation, or deferred annotations (``from __future__ import annotations``), leave
    ``field.type`` a non-callable and break every parse.
    """
    if config is None:
        return None
    if not isinstance(config, dict):
        raise ValueError(f"{label} config must be a mapping; got {type(config).__name__}")
    coerce = {field.name: field.type for field in _fields(cls)}
    unknown = sorted(set(config) - set(coerce))
    if unknown:
        raise ValueError(f"{label} config has unknown field(s): " + ", ".join(unknown))
    if "weight" not in config:
        raise ValueError(f"{label} config must define weight")
    return cls(**{name: coerce[name](value) for name, value in config.items()})


def _silence_anchor_norm(y: _torch.Tensor) -> _torch.Tensor:
    """Score an anchor's output against zero with mean absolute error.

    A squared term goes quieter the closer the model gets, which is the opposite of what
    an anchor needs: the offset being driven out is small in absolute terms (a trained
    SD1 capture sits around 2.6e-3), and there a squared term's gradient is orders of
    magnitude weaker than L1's constant pull. Raising the weight is no substitute, since
    it also scales the term early in training where the offset is still large and the
    anchor would then distort the fit it is supposed to leave alone.
    """
    return y.abs().mean()


@_dataclass
class _SilenceAnchorConfig:
    """
    Settings for the silence-in / silence-out anchor term.

    ``ny`` is the number of scored output samples per anchor row. The net runs on a
    longer window than that: ``receptive_field - 1`` samples of history, plus any
    ``training_warmup`` the net detaches before it starts scoring (``ConcatLSTM``'s
    burn-in, 8192 in the shipping config). Only the last ``ny`` samples are scored.

    For the shipping ConcatWaveNet the receptive field is 6347, so history dominates the
    cost at any sane ``ny`` -- widening the window is nearly free while widening the
    batch is not. That trade reverses on a net that requires uniform batch params
    (HyperWaveNet), where each row is a separate forward with its own generated weights,
    so ``batch_size`` multiplies the whole cost.

    ``silence_anchor_ramp`` covers everything this term does and more on a net that
    accepts a param trajectory, so this one is what remains for the nets that cannot take
    one (HyperWaveNet).
    """

    weight: float
    batch_size: int = 8
    ny: int = 512
    rail_probability: float = 0.25

    def __post_init__(self):
        # Validated here rather than in `from_config` so a config built in code gets the
        # same guarantees as one read from JSON.
        if self.weight < 0.0:
            raise ValueError(f"Silence anchor weight must be >= 0; got {self.weight}")
        if self.batch_size < 1:
            raise ValueError(
                f"Silence anchor batch_size must be >= 1; got {self.batch_size}"
            )
        if self.ny < 1:
            raise ValueError(f"Silence anchor ny must be >= 1; got {self.ny}")
        if not 0.0 <= self.rail_probability <= 1.0:
            raise ValueError(
                "Silence anchor rail_probability must be within [0, 1]; got "
                f"{self.rail_probability}"
            )

    def sample_params(self, net: _ParametricNet, window: _AnchorWindow) -> _torch.Tensor:
        """The control settings to run one anchor batch at, as ``(batch_size, P)``."""
        return _sample_raw_params(
            net.param_specs,
            self.batch_size,
            rail_probability=self.rail_probability,
            device=next(net.parameters()).device,
        )

    def as_trajectory(
        self, net: _ParametricNet, window: _AnchorWindow
    ) -> _torch.Tensor:
        """This anchor's batch as ``(batch_size, window length, P)``.

        The shape anchors have in common, which is what lets two of them share one
        forward. A held setting is a trajectory whose value never changes.
        """
        return _as_held_trajectory(self.sample_params(net, window), window.length)

    def check_net(self, net: _ParametricNet) -> None:
        """Raise if this anchor cannot be run against ``net``, called at construction."""

    @classmethod
    def from_config(cls, config: _Optional[dict]) -> "_Optional[_SilenceAnchorConfig]":
        return _loss_term_from_config(cls, config, "Silence anchor")


@_dataclass
class _SilenceAnchorRampConfig(_SilenceAnchorConfig):
    """
    Settings for the moving-control silence anchor.

    Gesture speed is drawn as window travel rather than in seconds; see
    :func:`sample_param_trajectories` for why the scored tail rather than the wall clock
    is the right yardstick. Travel above 1 is a gesture that resolves inside the tail, so
    its settling tail is scored.

    The default range brackets the runtime's own 200 ms smoothing ramp, which at the
    shipping ``ny`` of 512 is a travel of about 0.05 -- the range's log-midpoint. It
    reaches an order of magnitude either side of that: a slow two-second sweep, and a
    move faster than the smoothing can produce, which is what an automation jump or a
    preset recall looks like.

    ``hold_probability`` of the rows hold still, which is what the static anchor does. A
    net that accepts a trajectory therefore needs only this term; ``silence_anchor`` is
    for the nets that cannot take one.
    """

    min_window_travel: float = 0.005
    max_window_travel: float = 0.5
    hold_probability: float = 0.25

    def __post_init__(self):
        super().__post_init__()
        if not 0.0 < self.min_window_travel <= self.max_window_travel:
            raise ValueError(
                "Silence anchor window travel must satisfy "
                "0 < min_window_travel <= max_window_travel; got "
                f"{self.min_window_travel} and {self.max_window_travel}"
            )
        if not 0.0 <= self.hold_probability <= 1.0:
            raise ValueError(
                "Silence anchor hold_probability must be within [0, 1]; got "
                f"{self.hold_probability}"
            )

    def sample_params(self, net: _ParametricNet, window: _AnchorWindow) -> _torch.Tensor:
        """One control gesture per row, as ``(batch_size, window length, P)``."""
        return _sample_param_trajectories(
            net.param_specs,
            self.batch_size,
            window,
            min_window_travel=self.min_window_travel,
            max_window_travel=self.max_window_travel,
            rail_probability=self.rail_probability,
            hold_probability=self.hold_probability,
            device=next(net.parameters()).device,
        )

    def as_trajectory(
        self, net: _ParametricNet, window: _AnchorWindow
    ) -> _torch.Tensor:
        return self.sample_params(net, window)

    def check_net(self, net: _ParametricNet) -> None:
        if not (isinstance(net, _ParametricNet) and net.supports_param_trajectory):
            raise ValueError(
                "loss.silence_anchor_ramp requires a net that accepts a param "
                f"trajectory; {type(net).__name__} does not"
            )


def _splice_norm(
    preds: _torch.Tensor, targets: _torch.Tensor, energy_floor: float
) -> _torch.Tensor:
    """Score a splice batch as the mean of the rows' error-to-signal ratios.

    The artifact this term exists to remove is a click, and a click under a quiet passage
    is far more audible than the same absolute error under a loud chord. A plain squared
    error weights every row by its own loudness, so the loud windows own the gradient and
    the audible clicks are never punished; normalizing per row takes that back out.
    Rows are reduced as a mean of ratios rather than as pooled energies for the same
    reason -- pooling would reintroduce the loud-row dominance the ratio just removed.

    ``energy_floor`` is a per-sample squared amplitude, taken relative to the captures'
    own level so it survives output normalization. Without it a window landing in a
    near-silent stretch divides by nearly nothing and takes the step with it; below the
    floor the term degrades to an absolute measure, which is the right behaviour there
    because the silence anchors already own true silence.
    """
    error = (preds - targets).square().sum(dim=-1)
    energy = targets.square().sum(dim=-1)
    return (error / energy.clamp(min=energy_floor * targets.shape[-1])).mean()


@_dataclass
class _ParamStepSpliceConfig:
    """
    Settings for the param-step splice term.

    Captures made through one input file are the same device answering the same audio at
    different settings, so a target assembled from several of them at matching source
    times is supervised ground truth for a control that just changed -- the destination's
    steady state, arriving at once. That is the behaviour being asked for: the model owes
    the setting's response as if it had always been there, not the analog settling of the
    real part, which is deliberately not modeled. See :mod:`nam.models.parametric._splice`.

    The window is derived from the net rather than configured, so one block is correct for
    both a convolutional and a recurrent net. ``min_tail`` sets the scored tail for a net
    with no receptive field to bound it, and ``settle`` how much pre-step history the model
    gets; on the shipping ConcatWaveNet the receptive field already exceeds both, so
    neither costs it anything. ``batch_size`` 4 there and 8 on ConcatLSTM each put the
    extra forward at roughly a fifth of a training step -- which stacks with whatever the
    silence anchors are already spending.

    ``steps`` above 1 walks a chain of settings through the window. It is an experiment
    rather than the default: with a plateau shorter than the model's memory the
    per-segment target stops being reachable, and with two captures a chain is only a
    flick back and forth.

    ``weight`` 0 skips the training forward but still reports the validation metric, which
    is how an existing checkpoint's knob-move penalty gets measured before anything is
    trained on it.
    """

    weight: float
    batch_size: int = 4
    steps: int = 1
    spacing: int = 1024
    min_tail: int = 4096
    settle: int = 4096
    neighbors: int = 3
    neighbor_probability: float = 0.75
    esr_floor_fraction: float = 1e-3
    validation_windows: int = 64
    dc_report_db: float = -60.0
    lag_tolerance: int = 2

    def __post_init__(self):
        # Validated here rather than in `from_config` so a config built in code gets the
        # same guarantees as one read from JSON.
        if self.weight < 0.0:
            raise ValueError(f"Param step splice weight must be >= 0; got {self.weight}")
        if self.batch_size < 1:
            raise ValueError(
                f"Param step splice batch_size must be >= 1; got {self.batch_size}"
            )
        if self.validation_windows < 0:
            raise ValueError(
                "Param step splice validation_windows must be >= 0; got "
                f"{self.validation_windows}"
            )
        if self.neighbors < 1:
            raise ValueError(
                f"Param step splice neighbors must be >= 1; got {self.neighbors}"
            )
        if not 0.0 <= self.neighbor_probability <= 1.0:
            raise ValueError(
                "Param step splice neighbor_probability must be within [0, 1]; got "
                f"{self.neighbor_probability}"
            )
        if self.esr_floor_fraction <= 0.0:
            raise ValueError(
                "Param step splice esr_floor_fraction must be > 0; got "
                f"{self.esr_floor_fraction}"
            )
        if self.lag_tolerance < 1:
            raise ValueError(
                f"Param step splice lag_tolerance must be >= 1; got {self.lag_tolerance}"
            )
        # steps/spacing/min_tail/settle are window shapes; `splice_window` owns their
        # bounds, and `check_net` calls it at construction so a bad one fails here too.

    def window(self, net: _ParametricNet) -> _SpliceWindow:
        return _splice_window(
            net,
            steps=self.steps,
            spacing=self.spacing,
            min_tail=self.min_tail,
            settle=self.settle,
        )

    def source(
        self, datasets: _Sequence[_ParametricDataset], net: _ParametricNet
    ) -> _SpliceSource:
        return _SpliceSource.from_datasets(
            datasets,
            net,
            dc_report_db=self.dc_report_db,
            lag_tolerance=self.lag_tolerance,
        )

    def check_net(self, net: _ParametricNet) -> None:
        """Raise if this term cannot be run against ``net``, called at construction."""
        if not (isinstance(net, _ParametricNet) and net.supports_param_trajectory):
            raise ValueError(
                "loss.param_step_splice requires a net that accepts a param trajectory; "
                f"{type(net).__name__} does not"
            )
        # Surfaces a bad window here rather than on the first step.
        self.window(net)

    @classmethod
    def from_config(cls, config: _Optional[dict]) -> "_Optional[_ParamStepSpliceConfig]":
        return _loss_term_from_config(cls, config, "Param step splice")


@_dataclass
class _ParametricLossConfig(_LossConfig):
    mel_weight: _Optional[float] = None
    silence_anchor: _Optional[_SilenceAnchorConfig] = None
    silence_anchor_ramp: _Optional[_SilenceAnchorRampConfig] = None
    param_step_splice: _Optional[_ParamStepSpliceConfig] = None

    @classmethod
    def parse_config(cls, config):
        parsed = super().parse_config(config)
        parsed["mel_weight"] = config.get("mel_weight")
        parsed["silence_anchor"] = _SilenceAnchorConfig.from_config(
            config.get("silence_anchor")
        )
        parsed["silence_anchor_ramp"] = _SilenceAnchorRampConfig.from_config(
            config.get("silence_anchor_ramp")
        )
        parsed["param_step_splice"] = _ParamStepSpliceConfig.from_config(
            config.get("param_step_splice")
        )
        return parsed


class _ParametricLightningModule(_LightningModule):
    def __init__(
        self,
        net,
        optimizer_config=None,
        scheduler_config=None,
        loss_config=None,
    ):
        super().__init__(
            net,
            optimizer_config=optimizer_config,
            scheduler_config=scheduler_config,
            loss_config=(
                _ParametricLossConfig() if loss_config is None else loss_config
            ),
        )
        self._mel_mrstft = None
        # Fail here rather than on the first training step: a net that conditions on the
        # control vector as a whole has no meaning to give a control that moves within
        # one forward, so the setting cannot be honoured at all.
        for _, anchor in self._silence_anchors():
            anchor.check_net(net)
        if self._loss_config.param_step_splice is not None:
            self._loss_config.param_step_splice.check_net(net)
        # Set via `set_splice_sources` once the datasets exist (see `main`).
        self._train_splice_source: _Optional[_SpliceSource] = None
        self._val_splice_source: _Optional[_SpliceSource] = None
        # Set via `set_capture_batch_samplers` once the dataloaders exist (see `main`).
        # `None` means "capture identity unavailable" -- `_EpochMetrics` then falls back
        # to single-level pooling over the whole split instead of a wrong two-level split.
        self._train_capture_sampler: _Optional[_CaptureBatchSampler] = None
        self._val_capture_sampler: _Optional[_CaptureBatchSampler] = None

    @classmethod
    def parse_config(cls, config):
        parsed = super().parse_config(config)
        parsed["loss_config"] = _ParametricLossConfig.init_from_config(
            config.get("loss", {})
        )
        return parsed

    def set_capture_batch_samplers(
        self,
        train_sampler: _Optional[_Sampler] = None,
        val_sampler: _Optional[_Sampler] = None,
    ) -> None:
        """Give the module a reference to the (possibly capture-grouped) batch
        samplers backing its dataloaders, so `_EpochMetrics` can pool ESR/MSE/MRSTFT
        per capture rather than over the whole split. Only a `_CaptureBatchSampler`
        (`capture_grouped_batches` not disabled) actually enables the two-level
        reduction; anything else is treated the same as passing `None`.
        """
        self._train_capture_sampler = train_sampler
        self._val_capture_sampler = val_sampler

    def set_splice_sources(
        self,
        train: _Optional[_Sequence[_ParametricDataset]] = None,
        validation: _Optional[_Sequence[_ParametricDataset]] = None,
    ) -> None:
        """Give the module the captures the param-step splice term builds its windows
        from. A no-op when the term is not configured, so a run that does not ask for it
        pays none of the setup -- the shared-input, DC and alignment checks included.
        """
        config = self._loss_config.param_step_splice
        if config is None:
            return
        net = _get_parametric_net(self)
        self._train_splice_source = None if train is None else config.source(train, net)
        if validation is None:
            self._val_splice_source = None
        elif len(validation) < 2:
            # The metric is a reading, not the thing that was asked for, so a split with
            # nothing to step between costs the run its number rather than the run.
            _warn(
                "loss.param_step_splice reports no validation metric: the validation "
                f"split has {len(validation)} capture(s) and a step needs two. Hold out "
                "a second setting to measure what a control change costs."
            )
            self._val_splice_source = None
        else:
            self._val_splice_source = config.source(validation, net)

    @staticmethod
    def _capture_index(
        sampler: _Optional[_Sampler], batch_idx: int
    ) -> _Optional[int]:
        if isinstance(sampler, _CaptureBatchSampler):
            return sampler.capture_for_batch(batch_idx)
        return None

    def _get_loss_dict(self, preds, targets):
        loss_dict = super()._get_loss_dict(preds, targets)
        if self._loss_config.mel_weight is not None:
            loss_dict["Mel"] = _LossItem(
                self._loss_config.mel_weight,
                self._mel_mrstft_loss(preds, targets),
            )
        # Training-only: the anchors are regularizers on the model, not measurements of
        # held-out data, and computing them under validation would cost a forward per
        # batch for a number that says nothing about the split.
        if self.training:
            loss_dict.update(self._silence_anchor_items())
            loss_dict.update(self._param_step_splice_items())
        return loss_dict

    def _param_step_splice_items(self) -> dict[str, _LossItem]:
        config = self._loss_config.param_step_splice
        # A zero weight contributes nothing to the total, so its forward is skipped
        # rather than run and multiplied away. The validation metric still runs, which is
        # how a checkpoint's knob-move penalty is measured before it is trained on.
        if config is None or config.weight <= 0.0:
            return {}
        source = self._train_splice_source
        if source is None:
            # Silently training without a term that was asked for is worse than not
            # starting: the run looks normal and the model comes out unchanged.
            raise RuntimeError(
                "loss.param_step_splice is enabled but the training captures were never "
                "handed to the module; set_splice_sources must be called before fit"
            )
        net = _get_parametric_net(self)
        window = config.window(net)
        cases = source.sample_cases(
            config.batch_size,
            window,
            neighbors=config.neighbors,
            neighbor_probability=config.neighbor_probability,
        )
        preds, targets = self._splice_predictions(net, source, cases, window)
        return {
            "Param Step": _LossItem(
                config.weight,
                _splice_norm(
                    preds, targets, config.esr_floor_fraction * source.mean_square
                ),
            )
        }

    def _splice_predictions(
        self,
        net: _ParametricNet,
        source: _SpliceSource,
        cases,
        window: _SpliceWindow,
        static: bool = False,
    ) -> _Tuple[_torch.Tensor, _torch.Tensor]:
        """One splice batch, run and paired with its target.

        The captures stay on the host: a batch is a couple of hundred kilobytes against a
        forward of tens of thousands of samples, and keeping them there leaves tens of
        megabytes per capture off the accelerator.
        """
        x, params, targets = source.build(cases, window, static=static)
        device = next(net.parameters()).device
        x = x.to(device, non_blocking=True)
        params = params.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        return _splice_output(net, x, params, window), targets

    def _silence_anchor_items(self) -> dict[str, _LossItem]:
        """The enabled anchors, in as few forwards as the net allows.

        Anchors differ only in the control tensor they feed the same silent window, so
        when their windows agree they are one forward over a stacked batch rather than
        one each. Worth doing because these models are launch-bound rather than
        compute-bound: two forwards of B rows cost close to twice one forward of 2B
        however small the rows are, so halving the launches is worth more than the
        arithmetic the stack adds.
        """
        anchors = [
            (name, anchor)
            for name, anchor in self._silence_anchors()
            # A zero weight contributes nothing to the total, so its forward is skipped
            # rather than run and multiplied away.
            if anchor.weight > 0.0
        ]
        if not anchors:
            return {}
        net = _get_parametric_net(self)
        if (
            len(anchors) > 1
            # Equal ny keeps the shared window identical to what each anchor would have
            # run alone, so stacking cannot change either term's value.
            and len({anchor.ny for _, anchor in anchors}) == 1
            and net.supports_param_trajectory
            # This net evaluates a batch row by row, so a stacked batch would buy
            # nothing.
            and not net.requires_uniform_batch_params
        ):
            return self._merged_silence_anchor_items(net, anchors)
        return {
            name: _LossItem(anchor.weight, self._silence_anchor_loss(anchor))
            for name, anchor in anchors
        }

    def _merged_silence_anchor_items(
        self,
        net: _ParametricNet,
        anchors: _Sequence[_Tuple[str, _SilenceAnchorConfig]],
    ) -> dict[str, _LossItem]:
        """Every anchor in one forward, each still scored over its own rows."""
        window = _anchor_window(net, anchors[0][1].ny)
        y = _anchor_output(
            net,
            _torch.cat(
                [anchor.as_trajectory(net, window) for _, anchor in anchors], dim=0
            ),
            window.ny,
        )
        items = {}
        start = 0
        for name, anchor in anchors:
            rows = y[start : start + anchor.batch_size]
            items[name] = _LossItem(anchor.weight, _silence_anchor_norm(rows))
            start += anchor.batch_size
        return items

    def _silence_anchors(self) -> _Sequence[_Tuple[str, _SilenceAnchorConfig]]:
        """The configured anchors, each with the name it reports under."""
        return [
            (name, anchor)
            for name, anchor in (
                ("Silence Anchor", self._loss_config.silence_anchor),
                ("Silence Anchor Ramp", self._loss_config.silence_anchor_ramp),
            )
            if anchor is not None
        ]

    def _silence_anchor_loss(self, config: _SilenceAnchorConfig) -> _torch.Tensor:
        """Silence in, the controls wherever the anchor puts them, silence out."""
        net = _get_parametric_net(self)
        window = _anchor_window(net, config.ny)
        return _silence_anchor_norm(
            _anchor_output(net, config.sample_params(net, window), window.ny)
        )

    def _mel_mrstft_loss(
        self, preds: _torch.Tensor, targets: _torch.Tensor
    ) -> _torch.Tensor:
        """PANAMA's seven-resolution log-mel L1 loss."""
        if self._mel_mrstft is None:
            sample_rate = self.net.sample_rate
            if sample_rate is None:
                raise RuntimeError("mel_weight requires the model sample rate")
            self._mel_mrstft = [
                _MelSpectrogram(
                    sample_rate=sample_rate,
                    n_fft=window_length,
                    hop_length=window_length // 4,
                    win_length=window_length,
                    n_mels=n_mels,
                    power=1.0,
                ).to(preds.device)
                for window_length, n_mels in zip(
                    (32, 64, 128, 256, 512, 1024, 2048),
                    (5, 10, 20, 40, 80, 160, 320),
                )
            ]
        return sum(
            _torch.nn.functional.l1_loss(
                _torch.log10(transform(preds) + 1e-5),
                _torch.log10(transform(targets) + 1e-5),
            )
            for transform in self._mel_mrstft
        )

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
        # A non-finite gradient poisons Adam's moment estimates. Marking every gradient None
        # excludes the parameters from this step entirely: weights, weight decay, moments,
        # and per-parameter step counters all stay unchanged.
        non_finite = any(
            p.grad is not None and not _torch.isfinite(p.grad).all()
            for p in self.parameters()
        )
        if non_finite:
            for parameter in self.parameters():
                parameter.grad = None
        super().configure_gradient_clipping(
            optimizer, gradient_clip_val, gradient_clip_algorithm
        )

    def configure_optimizers(self):
        # Mirrors the base wiring but (a) uses AdamW for decoupled weight decay -- the
        # optimizer_config's "weight_decay" curbs the proxy's train/val overfitting, and
        # weight_decay 0.0 (the default) is identical to plain Adam -- and (b) adds optional
        # linear LR warmup via scheduler_config["warmup_epochs"], which keeps the earliest
        # updates small so full-BPTT LSTM training settles onto the descent before full-size
        # steps, avoiding the early gradient blow-ups that otherwise drive a run to NaN.
        optimizer_config = {"weight_decay": 0.0, **self._optimizer_config}
        optimizer = _torch.optim.AdamW(self.parameters(), **optimizer_config)
        if self._scheduler_config is None:
            return optimizer
        base_scheduler = getattr(
            _torch.optim.lr_scheduler, self._scheduler_config["class"]
        )(optimizer, **self._scheduler_config["kwargs"])
        warmup_epochs = self._scheduler_config.get("warmup_epochs") or 0
        if warmup_epochs > 0:
            warmup = _torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1.0 / warmup_epochs,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            scheduler = _torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, base_scheduler],
                milestones=[warmup_epochs],
            )
        else:
            scheduler = base_scheduler
        lr_scheduler_config = {"scheduler": scheduler}
        for key in ("interval", "frequency", "monitor"):
            if key in self._scheduler_config:
                lr_scheduler_config[key] = self._scheduler_config[key]
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def on_train_epoch_start(self):
        self._train_metrics = _EpochMetrics()

    def training_step(self, batch, batch_idx):
        preds, targets, loss_dict = self._shared_step(batch)
        loss = _torch.zeros((), device=preds.device)
        for v in loss_dict.values():
            if v.weight is not None and v.weight > 0.0:
                if v.value is None:
                    raise RuntimeError("Weighted training losses must define a tensor value")
                loss = loss + v.weight * v.value
        capture_index = self._capture_index(self._train_capture_sampler, batch_idx)
        self._train_metrics.update(preds, targets, _bucket_means(loss_dict), capture_index)
        return loss

    def on_train_epoch_end(self):
        self._log_bucket(self._train_metrics.compute(), _TRAIN_BUCKET)

    def on_validation_epoch_start(self):
        self._val_metrics = _EpochMetrics()

    def validation_step(self, batch, batch_idx):
        preds, targets, loss_dict = self._shared_step(batch)
        capture_index = self._capture_index(self._val_capture_sampler, batch_idx)
        self._val_metrics.update(preds, targets, _bucket_means(loss_dict), capture_index)
        # Reduction happens once in on_validation_epoch_end; nothing to return.

    def on_validation_epoch_end(self):
        metrics = self._val_metrics.compute()
        if not metrics:
            return
        self._log_bucket(metrics, _VALIDATION_BUCKET)
        val_loss_key = self._val_loss_key()
        if val_loss_key not in metrics:
            raise RuntimeError(
                f"Validation loss {val_loss_key!r} was not accumulated this epoch"
            )
        # Bare keys back the checkpoint monitor (val_loss), the checkpoint filename
        # (ESR, MSE), and any threshold-ESR early stopping.
        monitored = dict(metrics)
        monitored["val_loss"] = metrics[val_loss_key]
        self.log_dict(monitored)
        self._log_bucket(self._param_step_splice_metrics(), _VALIDATION_BUCKET)

    def _param_step_splice_metrics(self) -> dict[str, float]:
        """Knob-move error on held-out captures, against the same windows held still.

        The held-out captures share an input too, so a spliced target over them is a
        measurement rather than a regularizer -- the one number here that says what a
        control change costs. Raw spliced error would conflate that with how well the
        model fits those settings at all, so the same windows are rendered again with the
        control parked at the destination and the ratio is what gets read.

        Logged on its own rather than through the epoch metrics: it must not reach
        ``val_loss`` or the checkpoint monitor. The term has a floor no model can go
        below -- a recurrent net's state cannot hold the destination's history one sample
        after the step -- so selecting on it would trade away static fidelity.
        """
        config = self._loss_config.param_step_splice
        source = self._val_splice_source
        if config is None or source is None or config.validation_windows < 1:
            return {}
        net = _get_parametric_net(self)
        window = config.window(net)
        cases = source.enumerate_cases(config.validation_windows, window)
        floor = config.esr_floor_fraction * source.mean_square
        with _torch.no_grad():
            stepped = _splice_norm(
                *self._splice_predictions(net, source, cases, window), floor
            )
            held = _splice_norm(
                *self._splice_predictions(net, source, cases, window, static=True), floor
            )
        metrics = {
            "Param Step ESR": float(stepped),
            "Param Step Static ESR": float(held),
        }
        if metrics["Param Step Static ESR"] > 0.0:
            metrics["Param Step Penalty"] = (
                metrics["Param Step ESR"] / metrics["Param Step Static ESR"]
            )
        return metrics

    def _val_loss_key(self) -> str:
        val_loss_type = self._loss_config.val_loss
        if isinstance(val_loss_type, str):
            return val_loss_type.upper()
        return val_loss_type.value.upper()

    def _log_bucket(self, metrics: dict[str, float], bucket: str) -> None:
        if not metrics:
            return
        self.log_dict({f"{name}/{bucket}": value for name, value in metrics.items()})


class _ParametricModelCheckpoint(_ModelCheckpoint):
    """
    Extends ModelCheckpoint to export a `_parametric.nam` (and, for HyperWaveNet nets, a
    baked stock-WaveNet `.nam`) alongside each `.ckpt` it writes. Mirrors what
    ``nam.train.core._ModelCheckpoint`` does for the non-parametric trainer, so a hard
    crash mid-training still leaves usable `.nam` exports instead of only `.ckpt` files
    that require reloading through the training code to recover a model.
    """

    def __init__(self, *args, output_scale: _Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._output_scale = output_scale

    def _nam_stem(self, filepath: str) -> str:
        if not filepath.endswith(self.FILE_EXTENSION):
            raise ValueError(
                f"Checkpoint filepath {filepath} doesn't end in expected extension "
                f"{self.FILE_EXTENSION}"
            )
        return filepath[: -len(self.FILE_EXTENSION)]

    def _save_checkpoint(self, trainer: _pl.Trainer, filepath: str) -> None:
        super()._save_checkpoint(trainer, filepath)
        stem = _Path(self._nam_stem(filepath))
        outdir, basename = stem.parent, stem.name
        net = _get_parametric_net(trainer.model)
        try:
            if isinstance(net, _HyperWaveNet):
                _bake(net, net.nominal_params, output_scale=self._output_scale).export(
                    outdir, basename=basename
                )
            _export_parametric(
                net,
                outdir,
                basename=f"{basename}_parametric",
                output_scale=self._output_scale,
            )
        except NotImplementedError:
            # Defensive: nothing to checkpoint here if a future ParametricNet subclass
            # doesn't implement .nam weight export.
            pass

    def _remove_checkpoint(self, trainer: _pl.Trainer, filepath: str) -> None:
        super()._remove_checkpoint(trainer, filepath)
        stem = _Path(self._nam_stem(filepath))
        outdir, basename = stem.parent, stem.name
        for nam_basename in (basename, f"{basename}_parametric"):
            nam_path = outdir / f"{nam_basename}{_Exportable.FILE_EXTENSION}"
            if nam_path.exists():
                nam_path.unlink()


def _create_parametric_checkpoint_callbacks(
    learning_config, output_scale: _Optional[float] = None
):
    """
    Same checkpoint cadence/naming as ``nam.train.full._create_callbacks(packed=False)``,
    but using ``_ParametricModelCheckpoint`` so each `.ckpt` also gets a `.nam` export.
    """
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

    checkpoint_best = _ParametricModelCheckpoint(
        filename="{epoch:04d}_{step}_{ESR:.3e}_{MSE:.3e}",
        save_top_k=3,
        monitor="val_loss",
        output_scale=output_scale,
        **kwargs,
    )
    checkpoint_epoch = _ParametricModelCheckpoint(
        filename="checkpoint_epoch_{epoch:04d}",
        every_n_epochs=1,
        output_scale=output_scale,
    )
    callbacks = [checkpoint_best]
    if not validate_inside_epoch:
        callbacks.append(checkpoint_epoch)
        return callbacks
    checkpoint_last = _ParametricModelCheckpoint(
        filename="checkpoint_last_{epoch:04d}_{step}",
        output_scale=output_scale,
        **kwargs,
    )
    callbacks.extend([checkpoint_last, checkpoint_epoch])
    return callbacks


def _create_parametric_callbacks(
    learning_config, output_scale: _Optional[float] = None
):
    callbacks = _create_parametric_checkpoint_callbacks(learning_config, output_scale)
    threshold_esr = learning_config.get("threshold_esr")
    if threshold_esr is not None:
        callbacks.append(
            _ValidationStopping(monitor="ESR", stopping_threshold=threshold_esr)
        )
    return callbacks


def _parametric_plot_label(ds: _ParametricDataset) -> str:
    y_path = getattr(ds.dataset, "_y_path", None)
    if y_path is not None:
        return _Path(y_path).name
    params = ", ".join(f"{value:.3g}" for value in ds.params.detach().cpu().tolist())
    return f"params=[{params}]"


def _plot_parametric(
    model: _LightningModule,
    ds,
    savefig=None,
    show=True,
    window_start: _Optional[int] = None,
    window_end: _Optional[int] = None,
):
    if isinstance(ds, _ConcatDataset):

        def extend_savefig(i, original):
            if original is None:
                return None
            original = _Path(original)
            return original.with_name(
                f"{original.stem}_{i}{original.suffix}"
            )

        for i, ds_i in enumerate(ds.datasets):
            _plot_parametric(
                model,
                ds_i,
                savefig=extend_savefig(i, savefig),
                show=show and i == len(ds.datasets) - 1,
                window_start=window_start,
                window_end=window_end,
            )
        return
    if not isinstance(ds, _ParametricDataset):
        raise TypeError(
            "Expected a ParametricDataset or ConcatDataset of ParametricDataset; "
            f"got {type(ds).__name__}"
        )
    with _torch.no_grad():
        sample_rate = ds.sample_rate if ds.sample_rate is not None else 48_000
        tx = len(ds.dataset.x) / sample_rate
        print(f"Run (t={tx:.2f})")
        t0 = _time()
        output = model(ds.dataset.x, ds.params).cpu().numpy().flatten()
        t1 = _time()
        try:
            rt = f"{tx / (t1 - t0):.2f}"
        except ZeroDivisionError:
            rt = "???"
        print(f"Took {t1 - t0:.2f} ({rt}x)")
    target = ds.dataset.y.cpu().numpy()
    # Held-out validation clips can be short (a tail of unseen audio). If the requested
    # window starts past the end of this clip the default slice plots nothing, so fall back
    # to the whole clip rather than an empty figure. ESR below is over the full signal.
    if window_start is not None and window_start >= len(output):
        window_start, window_end = None, None
    _plt.figure(figsize=(16, 5))
    _plt.plot(output[window_start:window_end], label="Prediction")
    _plt.plot(target[window_start:window_end], linestyle="--", label="Target")
    nrmse = _rms(_torch.tensor(output) - ds.dataset.y) / _rms(ds.dataset.y)
    esr = nrmse**2
    _plt.title(f"{_parametric_plot_label(ds)}\nESR={esr:.3f}")
    _plt.legend()
    if savefig is not None:
        _plt.savefig(savefig)
    if show:
        _plt.show()


def main(
    data_config,
    model_config,
    learning_config,
    outdir: _Path,
    no_show: bool = False,
    make_plots=True,
):
    if not outdir.exists():
        raise RuntimeError(f"No output location found at {outdir}")
    for basename, config in (
        ("data", data_config),
        ("model", model_config),
        ("learning", learning_config),
    ):
        with open(_Path(outdir, f"config_{basename}.json"), "w") as fp:
            _json.dump(config, fp, indent=4)

    if model_config["net"]["name"] == "PackedWaveNet":
        raise ValueError("PackedWaveNet is not supported by the parametric trainer")

    data_config = _data_config_from_model(data_config, model_config)
    model = _ParametricLightningModule.init_from_config(model_config)
    net = _get_parametric_net(model)

    data_config["common"] = data_config.get("common", {})
    if "nx" in data_config["common"]:
        _warn(
            f"Overriding data nx={data_config['common']['nx']} with model required "
            f"{net.receptive_field}"
        )
    data_config["common"]["nx"] = net.receptive_field

    dataset_train = _init_dataset(data_config, _Split.TRAIN)
    dataset_validation = _init_dataset(data_config, _Split.VALIDATION)
    inner_train = _ConcatDataset(_iter_inner_datasets(dataset_train))
    inner_validation = _ConcatDataset(_iter_inner_datasets(dataset_validation))
    _apply_joint_dataset_hooks(
        dataset_train=inner_train,
        dataset_validation=inner_validation,
        hooks=_get_joint_dataset_hooks(data_config.get("joint", [])),
    )
    net.sample_rate = getattr(dataset_train, "sample_rate", None)
    _handshake_datasets(model, dataset_train, dataset_validation)
    output_scale = _output_scale_from_datasets((dataset_train, dataset_validation))

    train_dataloader = _make_parametric_dataloader(
        dataset_train, learning_config["train_dataloader"]
    )
    val_dataloader = _make_parametric_dataloader(
        dataset_validation, learning_config["val_dataloader"]
    )
    # Lets _EpochMetrics pool ESR/MSE/MRSTFT per capture instead of over the whole split;
    # a no-op (single-level fallback) if capture_grouped_batches was turned off for a loader.
    model.set_capture_batch_samplers(
        train_sampler=train_dataloader.batch_sampler,
        val_sampler=val_dataloader.batch_sampler,
    )
    # After the joint hooks, which rescale y in place: the splice source measures the
    # captures' DC and level, and would report the pre-scaling ones.
    model.set_splice_sources(
        train=_iter_parametric_datasets(dataset_train),
        validation=_iter_parametric_datasets(dataset_validation),
    )
    _enable_training_fast_paths(
        net,
        (train_dataloader, val_dataloader),
        learning_config.get("torch_compile", {}),
    )
    callbacks = _create_parametric_callbacks(learning_config, output_scale)
    trainer = _pl.Trainer(
        callbacks=callbacks,
        default_root_dir=outdir,
        **learning_config["trainer"],
    )

    try:
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
            _release_training_fast_paths(net)
        # Reached on normal completion or a user interrupt; in both cases we still want to
        # export the best model. A hard training error instead propagates past this block
        # (skipping export so a secondary bake/export failure can't mask the real error)
        # while the outer `finally` still tears the datasets down.
        checkpoint_callback = trainer.checkpoint_callback
        best_checkpoint = (
            checkpoint_callback.best_model_path
            if isinstance(checkpoint_callback, _ModelCheckpoint)
            else ""
        )
        if best_checkpoint != "":
            model = _ParametricLightningModule.load_from_checkpoint(
                best_checkpoint,
                **_ParametricLightningModule.parse_config(model_config),
            )
        net = _get_parametric_net(model)
        model.cpu()
        model.eval()
        net.sample_rate = getattr(dataset_train, "sample_rate", None)
        _handshake_datasets(model, dataset_train, dataset_validation)
        if make_plots:
            _plot_parametric(
                model,
                dataset_validation,
                savefig=_Path(outdir, "comparison.png"),
                window_start=100_000,
                window_end=110_000,
                show=False,
            )
            _plot_parametric(model, dataset_validation, show=not no_show)

        # Baking a fixed-setting stock WaveNet snapshot is a HyperWaveNet-only feature;
        # other parametric nets (e.g. ConcatWaveNet) only get the parametric export.
        if isinstance(net, _HyperWaveNet):
            _bake(
                net,
                net.nominal_params,
                output_scale=output_scale,
            ).export(outdir)
        _export_parametric(
            net,
            outdir,
            basename="model_parametric",
            output_scale=output_scale,
        )
    finally:
        dataset_train.teardown()
        dataset_validation.teardown()
