"""
C3.1 training seam tests: PA8, PA8b.

PA8  — ParametricLightningModule._shared_step routes (params, x, y) correctly;
        loss is scalar-computable.
PA8b — Base LightningModule._shared_step mis-routes a (params, x, y) batch when
        wrapping ParametricWaveNet (confirms the override is load-bearing).
"""

import pytest
import torch

from nam.models.parametric import ParametricWaveNet
from nam.train.lightning_module import LightningModule, LossConfig
from nam.train.parametric import ParametricLightningModule

# ---------------------------------------------------------------------------
# Tiny ParametricWaveNet config (P=2, single layer array, small channels)
# ---------------------------------------------------------------------------

_PARAM_CONFIG = {
    "layers_configs": [
        {
            "input_size": 1,
            "condition_size": 1,
            "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
            "channels": 4,
            "kernel_size": 2,
            "dilations": [1, 2],
            "activation": "Tanh",
        }
    ],
    "head_scale": 1.0,
    "params": [
        {"name": "gain",   "min": 0.0, "max": 1.0, "default": 0.5},
        {"name": "treble", "min": 0.0, "max": 1.0, "default": 0.3},
    ],
}

_LOSS_CONFIG = LossConfig(mse_weight=1.0, mrstft_weight=None)


def _build_param_net() -> ParametricWaveNet:
    return ParametricWaveNet.init_from_config(_PARAM_CONFIG)


def _make_batch(model: ParametricWaveNet, batch_size: int = 2, extra: int = 64):
    """Return (params, x, y) with shapes (B, P), (B, L), (B, L-RF+1)."""
    rf = model.receptive_field
    seq_len = rf + extra
    target_len = seq_len - rf + 1
    params = torch.randn(batch_size, 2)
    x = torch.randn(batch_size, seq_len)
    y = torch.randn(batch_size, target_len)
    return params, x, y


# ---------------------------------------------------------------------------
# PA8 — ParametricLightningModule routes 3-tuple correctly
# ---------------------------------------------------------------------------


def test_pa8_parametric_shared_step_correct_routing():
    """PA8: ParametricLightningModule._shared_step routes (params, x, y) correctly.
    preds has the right shape and loss is scalar-computable."""
    net = _build_param_net()
    module = ParametricLightningModule(net, loss_config=_LOSS_CONFIG)
    module.eval()

    batch = _make_batch(net)
    params, x, y = batch

    with torch.no_grad():
        preds, targets, loss_dict = module._shared_step(batch)

    assert isinstance(preds, torch.Tensor)
    assert preds.shape == y.shape, f"preds shape {preds.shape} != y shape {y.shape}"
    # Scalar loss sanity check (same pattern as training_step)
    loss: torch.Tensor = torch.zeros(())
    for v in loss_dict.values():
        if v.weight is not None and v.weight > 0.0 and v.value is not None:
            loss = loss + v.weight * v.value
    assert loss.ndim == 0, "Loss must be a scalar (0-dim tensor)"


# ---------------------------------------------------------------------------
# PA8b — Base LightningModule mis-routes a 3-tuple batch (override is load-bearing)
# ---------------------------------------------------------------------------


def test_pa8b_base_shared_step_misroutes_3tuple():
    """PA8b: Base LightningModule._shared_step mis-routes a (params, x, y) batch
    when wrapping ParametricWaveNet, confirming the override is not redundant.

    Base does self(params, x, pad_start=False) which calls
    ParametricWaveNet.forward(x=params, params=x, ...) — swapping them.
    This should raise ValueError because x (the audio) has wrong shape when
    treated as params, or params has wrong shape when treated as audio."""
    net = _build_param_net()
    module = LightningModule(net, loss_config=_LOSS_CONFIG)  # BASE class, not parametric
    module.eval()

    batch = _make_batch(net)

    with torch.no_grad():
        with pytest.raises(ValueError):
            module._shared_step(batch)


def test_pa8c_parametric_optimizer_uses_single_group_by_default():
    net = _build_param_net()
    module = ParametricLightningModule(
        net,
        optimizer_config={"lr": 1.0e-3},
        loss_config=_LOSS_CONFIG,
    )

    optimizer = module.configure_optimizers()

    assert isinstance(optimizer, torch.optim.Adam)
    assert len(optimizer.param_groups) == 1
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3)


def test_pa8d_parametric_optimizer_can_split_adapter_learning_rate():
    net = _build_param_net()
    net.register_parameter(
        "_test_wrapper_parameter", torch.nn.Parameter(torch.tensor(1.0))
    )
    module = ParametricLightningModule(
        net,
        optimizer_config={
            "lr": 1.0e-3,
            "adapter_lr": 5.0e-4,
            "weight_decay": 1.0e-6,
        },
        loss_config=_LOSS_CONFIG,
    )

    optimizer = module.configure_optimizers()

    assert isinstance(optimizer, torch.optim.Adam)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0e-3)
    assert optimizer.param_groups[1]["lr"] == pytest.approx(5.0e-4)

    all_param_ids = {id(param) for param in module.parameters()}
    adapter_param_ids = {id(param) for param in module.net._adapter.parameters()}
    trunk_param_ids = all_param_ids - adapter_param_ids
    optimizer_trunk_ids = {id(param) for param in optimizer.param_groups[0]["params"]}
    optimizer_adapter_ids = {id(param) for param in optimizer.param_groups[1]["params"]}

    assert optimizer_trunk_ids == trunk_param_ids
    assert optimizer_adapter_ids == adapter_param_ids
    assert optimizer_trunk_ids.isdisjoint(optimizer_adapter_ids)
