# File: catnets.py
# Created Date: Wednesday June 22nd 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
"Cat nets" -- parametric models where the parametric input is concatenated to the
input samples
"""

import logging
from enum import Enum
from typing import Any, Dict

import torch

from ..base import ModelDatasetHandshakeError, ParametricBaseNet
from ..recurrent import LSTM
from ..wavenet import WaveNet

logger = logging.getLogger(__name__)


class _ShapeType(Enum):
    CONV = "conv"  # (B,C,L)
    RNN = "rnn"  # (B,L,D)


class _CatMixin(ParametricBaseNet):
    """
    Parametric nets that concatenate the params with the input at each time point.
    """

    @property
    def _expected_param_dim(self) -> int:
        if self._shape_type == _ShapeType.RNN:
            return self._input_size - 1  # type: ignore[attr-defined]
        return self._net._layers[0]._config["input_size"] - 1  # type: ignore[attr-defined]

    @property
    def _shape_type(self) -> _ShapeType:
        raise NotImplementedError()

    @property
    def _single_class(self):
        """
        The class for the non-parametric model that this is extending.
        """
        raise NotImplementedError()

    def handshake(self, dataset: "nam.data.AbstractDataset"):  # noqa: F821
        super().handshake(dataset)
        if hasattr(dataset, "keys") and len(dataset.keys) != self._expected_param_dim:
            raise ModelDatasetHandshakeError(
                f"Parametric dimension mismatch: model expects {self._expected_param_dim} "
                f"parameters but dataset provides {len(dataset.keys)}"
            )

    def _forward(self, params, x):
        sequence_length = x.shape[1]
        if self._shape_type == _ShapeType.RNN:
            params_tiled = torch.tile(params[:, None, :], (1, sequence_length, 1))
            x_augmented = torch.cat([x[..., None], params_tiled], dim=2)
        else:
            params_tiled = torch.tile(params[..., None], (1, 1, sequence_length))
            x_augmented = torch.cat([x[:, None, :], params_tiled], dim=1)
        return self._single_class._forward(self, x_augmented)

    def _export_input_output_args(self):
        return (self._get_param_defaults(device=self.device),)


class CatLSTM(_CatMixin, LSTM):
    @property
    def _shape_type(self) -> _ShapeType:
        return _ShapeType.RNN

    @property
    def _single_class(self):
        return LSTM

    def _at_nominal_settings(self, x: torch.Tensor) -> torch.Tensor:
        params = self._get_param_defaults(device=x.device)
        return self(params, x)


class CatWaveNet(_CatMixin, WaveNet):
    @property
    def _shape_type(self) -> _ShapeType:
        return _ShapeType.CONV

    @property
    def _single_class(self):
        return WaveNet

    def _at_nominal_settings(self, x: torch.Tensor) -> torch.Tensor:
        params = self._get_param_defaults(device=x.device)
        return self(params, x)
