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
    def param_names(self) -> _List[str]:
        return list(self._param_names)

    @property
    def param_dim(self) -> int:
        return len(self._param_names)

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
