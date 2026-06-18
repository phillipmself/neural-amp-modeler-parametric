"""
ParametricDataset: a dataset for parametric A2 training.

Yields a 3-tuple (params, x, y) per item, where:
  - params: float32 tensor of shape (P,) — the parameter setting for this capture
  - x: float32 tensor of shape (NX+NY-1,) — audio input window
  - y: float32 tensor of shape (NY,) — audio output window

Standalone subclass of AbstractDataset (NOT Dataset) to sidestep
Dataset._ScaleOutputHook at data.py:339, which hard-raises on "ParametricWaveNet"
(AD-4/AD-6 in architecture-decisions.md).

Uses composition: holds an internal Dataset instance for all WAV loading, windowing,
start/stop, and length logic. __getitem__ delegates to the inner Dataset and prepends
the params tensor to form the 3-tuple.
"""

from copy import deepcopy as _deepcopy
from typing import Any as _Any
from typing import Dict as _Dict
from typing import List as _List
from typing import Optional as _Optional
from typing import Tuple as _Tuple

import torch as _torch

from nam._core import InitializableFromConfig as _InitializableFromConfig
from nam.data import AbstractDataset as _AbstractDataset
from nam.data import Dataset as _Dataset


class ParametricDataset(_AbstractDataset, _InitializableFromConfig):
    """
    Dataset for a single parametric capture: one WAV pair at one parameter setting.

    Wraps an ordinary Dataset via composition so all WAV I/O, windowing, start/stop,
    delay, and sample-rate handling are inherited from the proven Dataset implementation.
    The only addition is a fixed params tensor prepended to every item.

    Config keys (beyond the ordinary Dataset keys):
        param_names: list[str] — ordered list of parameter names (must match model)
        params:      list[float] — one scalar per param_name for this capture
        param_dim:   int (optional) — if present, validated against len(param_names)

    The 'params' list order must match the model's param_names order.
    """

    def __init__(
        self,
        inner: _Dataset,
        param_names: _List[str],
        params: _List[float],
        param_dim: _Optional[int] = None,
    ):
        """
        :param inner: Fully-constructed ordinary Dataset to delegate audio windowing to.
        :param param_names: Ordered list of parameter names for this capture's setting.
        :param params: Scalar value for each parameter; must be len(param_names).
        :param param_dim: If provided, validated against len(param_names). Useful for
            catching config mismatches early when the model config also specifies param_dim.
        """
        if len(params) != len(param_names):
            raise ValueError(
                f"'params' has {len(params)} values but 'param_names' has "
                f"{len(param_names)} entries; they must match. "
                f"param_names={param_names}, params={params}"
            )
        if param_dim is not None and param_dim != len(param_names):
            raise ValueError(
                f"'param_dim'={param_dim} does not match len(param_names)="
                f"{len(param_names)}. Either fix 'param_dim' or 'param_names'."
            )
        self._inner = inner
        self._param_names = list(param_names)
        # Store params as float32 tensor once — reused on every __getitem__ call
        self._params = _torch.tensor(
            [float(v) for v in params], dtype=_torch.float32
        )

    # ------------------------------------------------------------------
    # AbstractDataset contract
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int):  # type: ignore[override]
        """
        :return: (params, x, y)
            params: (P,) float32 — fixed parameter setting for this capture
            x:      (NX+NY-1,) float32 — audio input window (same as Dataset contract)
            y:      (NY,) float32 — audio output window (same as Dataset contract)

        The base AbstractDataset.__getitem__ has no return annotation (its body is `pass`),
        so pyright infers `None`. We broaden the return rather than narrowing it, so the
        # type: ignore[override] suppression is intentional — not a silenced bug.
        """
        x, y = self._inner[idx]
        return self._params, x, y

    def __len__(self) -> int:
        return len(self._inner)

    # ------------------------------------------------------------------
    # Convenience accessors (mirrors Dataset public surface for C2.2 validation)
    # ------------------------------------------------------------------

    @property
    def nx(self) -> int:
        return self._inner.nx

    @property
    def ny(self) -> int:
        return self._inner.ny

    @property
    def sample_rate(self) -> _Optional[float]:
        return self._inner.sample_rate

    @property
    def y(self) -> _torch.Tensor:
        return self._inner.y

    @property
    def param_names(self) -> _List[str]:
        return list(self._param_names)

    @property
    def param_dim(self) -> int:
        return len(self._param_names)

    def scale_output(self, gain: float):
        self._inner.scale_output(gain)

    def teardown(self):
        self._inner.teardown()

    # ------------------------------------------------------------------
    # InitializableFromConfig
    # ------------------------------------------------------------------

    @classmethod
    def parse_config(cls, config: _Dict[str, _Any]) -> _Dict[str, _Any]:
        """
        Extract parametric keys from config, then delegate WAV loading to Dataset.

        config must contain (beyond the ordinary Dataset keys):
            param_names: list[str]
            params:      list[float]
        config may contain:
            param_dim: int — validated if present
        """
        config = _deepcopy(config)

        # Extract parametric-only keys before passing to Dataset
        param_names = config.pop("param_names")
        params = config.pop("params")
        param_dim: _Optional[int] = config.pop("param_dim", None)

        if not isinstance(param_names, list) or len(param_names) == 0:
            raise ValueError(
                "'param_names' must be a non-empty list of strings in ParametricDataset config"
            )
        if not isinstance(params, list):
            raise ValueError(
                "'params' must be a list of floats in ParametricDataset config"
            )
        if len(params) != len(param_names):
            raise ValueError(
                f"'params' has {len(params)} values but 'param_names' has "
                f"{len(param_names)} entries; they must match. "
                f"param_names={param_names}, params={params}"
            )
        if param_dim is not None and param_dim != len(param_names):
            raise ValueError(
                f"'param_dim'={param_dim} does not match len(param_names)="
                f"{len(param_names)}. Either fix 'param_dim' or 'param_names'."
            )

        # Delegate WAV parsing to Dataset — this handles x_path/y_path resolution,
        # sample-rate checking, delay, start/stop, and length validation.
        # The FileNotFoundError from wav loading surfaces naturally with the path in
        # the message (Dataset.parse_config uses wav_to_tensor which opens the file).
        inner = _Dataset.init_from_config(config)

        return {
            "inner": inner,
            "param_names": param_names,
            "params": params,
            "param_dim": param_dim,
        }


class ParametricConcatDataset(_AbstractDataset):
    """
    Concatenates N per-capture ParametricDataset instances for joint training.

    Each subdataset represents one amplifier setting (one capture). All subdatasets
    must share the same ordered parameter schema, param_dim, nx, ny, and sample_rate
    — the concat would be meaningless otherwise and is likely a config mistake.

    __getitem__ routes to the correct subdataset using a precomputed dict lookup
    (same pattern as ConcatDataset._make_lookup in nam/data.py:905).

    This class intentionally does NOT implement parse_config; building the concat from
    config is deferred to the trainer layer which controls the list of captures.
    """

    def __init__(self, datasets: _List[ParametricDataset]) -> None:
        if len(datasets) == 0:
            raise ValueError(
                "ParametricConcatDataset requires at least one ParametricDataset; "
                "received an empty list."
            )
        self._datasets = list(datasets)
        self._validate_datasets()
        self._lookup = self._make_lookup()

    def _validate_datasets(self) -> None:
        """Validate schema, param_dim, nx, ny, and sample_rate consistency."""
        ref_param_names = self._datasets[0].param_names
        ref_param_dim = self._datasets[0].param_dim
        ref_nx = self._datasets[0].nx
        ref_ny = self._datasets[0].ny
        ref_sr = self._datasets[0].sample_rate

        for i, ds in enumerate(self._datasets[1:], start=1):
            if ds.param_dim != ref_param_dim:
                raise ValueError(
                    f"param_dim mismatch: dataset 0 has param_dim={ref_param_dim} "
                    f"but dataset {i} has param_dim={ds.param_dim}. "
                    "All subdatasets must share the same param_dim."
                )
            if ds.param_names != ref_param_names:
                raise ValueError(
                    f"param_names mismatch: dataset 0 has param_names={ref_param_names} "
                    f"but dataset {i} has param_names={ds.param_names}. "
                    "All subdatasets must share the same ordered param_names."
                )
            if ds.nx != ref_nx:
                raise ValueError(
                    f"nx mismatch: dataset 0 has nx={ref_nx} "
                    f"but dataset {i} has nx={ds.nx}."
                )
            if ds.ny != ref_ny:
                raise ValueError(
                    f"ny mismatch: dataset 0 has ny={ref_ny} "
                    f"but dataset {i} has ny={ds.ny}."
                )
            if ds.sample_rate != ref_sr:
                raise ValueError(
                    f"sample_rate mismatch: dataset 0 has sample_rate={ref_sr} "
                    f"but dataset {i} has sample_rate={ds.sample_rate}."
                )

    def _make_lookup(self) -> _Dict[int, _Tuple[int, int]]:
        """Map global index → (dataset_index, local_index)."""
        lookup: _Dict[int, _Tuple[int, int]] = {}
        offset = 0
        j = 0
        for i in range(len(self)):
            if offset == len(self._datasets[j]):
                offset -= len(self._datasets[j])  # resets offset to 0
                j += 1
            lookup[i] = (j, offset)
            offset += 1
        return lookup

    def __len__(self) -> int:
        return sum(len(ds) for ds in self._datasets)

    def __getitem__(self, idx: int):  # type: ignore[override]
        ds_idx, local_idx = self._lookup[idx]
        return self._datasets[ds_idx][local_idx]

    @property
    def param_dim(self) -> int:
        return self._datasets[0].param_dim

    @property
    def nx(self) -> int:
        return self._datasets[0].nx

    @property
    def ny(self) -> int:
        return self._datasets[0].ny

    @property
    def sample_rate(self) -> _Optional[float]:
        return self._datasets[0].sample_rate

    @property
    def datasets(self) -> _List[ParametricDataset]:
        return list(self._datasets)


def _build_parametric_concat(
    configs: _List[_Dict[str, _Any]],
) -> "ParametricConcatDataset":
    """Factory for list-based parametric configs; called by register_concat_dataset_initializer.

    Each dict in configs is a fully-merged per-capture config (common keys already merged
    by init_dataset before this function is called).
    """
    datasets = [ParametricDataset.init_from_config(c) for c in configs]
    return ParametricConcatDataset(datasets)
