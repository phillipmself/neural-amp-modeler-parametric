"""
ParametricLightningModule: thin LightningModule subclass that overrides _shared_step
to route a 3-tuple parametric batch (params, x, y) correctly.
"""
from typing import Dict as _Dict
from typing import Tuple as _Tuple

import torch as _torch

from .lightning_module import LightningModule as _LightningModule
from .lightning_module import _LossItem


class ParametricLightningModule(_LightningModule):
    """
    Overrides _shared_step to unpack (params, x, y) and pass params as kwarg.

    Base _shared_step binds batch[:-1] as positional args, which would pass params
    as x and x as params on a 3-tuple batch. This override corrects the routing.
    """

    def _shared_step(
        self, batch
    ) -> _Tuple[_torch.Tensor, _torch.Tensor, _Dict[str, _LossItem]]:
        params, x, y = batch
        preds = self.net(x, params=params, pad_start=False)
        return preds, y, self._get_loss_dict(preds, y)
