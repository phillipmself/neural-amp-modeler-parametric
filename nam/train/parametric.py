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

    def configure_optimizers(self):
        adapter_lr = self._optimizer_config.get("adapter_lr")
        adapter_weight_decay = self._optimizer_config.get("adapter_weight_decay")
        if adapter_lr is None and adapter_weight_decay is None:
            return super().configure_optimizers()

        optimizer_config = dict(self._optimizer_config)
        optimizer_config.pop("adapter_lr", None)
        optimizer_config.pop("adapter_weight_decay", None)
        adapter_params = list(self.net._adapter.parameters())
        adapter_param_ids = {id(param) for param in adapter_params}
        trunk_params = [
            param for param in self.parameters() if id(param) not in adapter_param_ids
        ]
        adapter_group = {"params": adapter_params}
        if adapter_lr is not None:
            adapter_group["lr"] = adapter_lr
        if adapter_weight_decay is not None:
            adapter_group["weight_decay"] = adapter_weight_decay
        optimizer = _torch.optim.Adam(
            [
                {"params": trunk_params},
                adapter_group,
            ],
            **optimizer_config,
        )
        if self._scheduler_config is None:
            return optimizer

        lr_scheduler = getattr(
            _torch.optim.lr_scheduler, self._scheduler_config["class"]
        )(optimizer, **self._scheduler_config["kwargs"])
        lr_scheduler_config = {"scheduler": lr_scheduler}
        for key in ("interval", "frequency", "monitor"):
            if key in self._scheduler_config:
                lr_scheduler_config[key] = self._scheduler_config[key]
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_config}

    def _shared_step(
        self, batch
    ) -> _Tuple[_torch.Tensor, _torch.Tensor, _Dict[str, _LossItem]]:
        params, x, y = batch
        preds = self.net(x, params=params, pad_start=False)
        return preds, y, self._get_loss_dict(preds, y)
