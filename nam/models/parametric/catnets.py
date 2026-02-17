# File: catnets.py
# Created Date: Wednesday June 22nd 2022
# Author: Steven Atkinson (steven@atkinson.mn)

"""
"Cat nets" -- parametric models where the parametric input is concatenated to the
input samples
"""

import logging

import torch

from ..base import ModelDatasetHandshakeError, ParametricBaseNet
from ..wavenet import WaveNet

logger = logging.getLogger(__name__)


class _CatMixin(ParametricBaseNet):
    """
    Parametric nets that concatenate the params with the input at each time point.
    """

    def __init__(self, *args, **kwargs):
        layers_configs = kwargs.get("layers_configs", args[0] if len(args) > 0 else None)
        if layers_configs is None or len(layers_configs) == 0:
            raise ValueError("Expected non-empty WaveNet `layers_configs` for CatWaveNet.")
        first_input_size = layers_configs[0].get("input_size", None)
        if first_input_size is None:
            raise KeyError("Expected `input_size` in first WaveNet layer config.")
        self._expected_param_dim_cache = int(first_input_size) - 1
        super().__init__(*args, **kwargs)

    @property
    def _expected_param_dim(self) -> int:
        return self._expected_param_dim_cache

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
        params_tiled = torch.tile(params[..., None], (1, 1, sequence_length))
        x_augmented = torch.cat([x[:, None, :], params_tiled], dim=1)
        return self._single_class._forward(self, x_augmented)

    def _export_input_output_args(self):
        return (self._get_param_defaults(device=self.device),)


class CatWaveNet(_CatMixin, WaveNet):
    @property
    def _single_class(self):
        return WaveNet

    def _at_nominal_settings(self, x: torch.Tensor) -> torch.Tensor:
        params = self._get_param_defaults(device=x.device)
        return self(params, x)
