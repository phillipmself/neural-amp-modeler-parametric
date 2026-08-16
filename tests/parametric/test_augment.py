import pytest as _pytest
import torch as _torch

from nam.models.parametric import ConcatLSTM as ConcatLSTM
from nam.models.parametric import ParamSpec as _ParamSpec
from nam.models.parametric import ConcatWaveNet as _ConcatWaveNet
from nam.models.parametric import HyperWaveNet as _HyperWaveNet
from nam.models.parametric._augment import (
    sample_landed_trajectories as _sample_landed_trajectories,
)
from nam.train.parametric import _LandedMoveConfig as _LandedMoveConfig
from nam.train.parametric import _ParametricLossConfig as _ParametricLossConfig
from nam.train.parametric import _ParametricLightningModule as _ParametricLightningModule
from tests.parametric.test_concat_wavenet import _concat_wavenet_config
from tests.parametric.test_hyperwavenet import _hyperwavenet_config

_SAMPLE_RATE = 48000.0
# Long enough that a gesture landing tens of milliseconds early still has room to move
# inside the window; the shipping models have ~132 ms of history.
_LAND_BY = 8192
_NY = 64
_LENGTH = _LAND_BY + _NY


def _net() -> _ConcatWaveNet:
    net = _ConcatWaveNet.init_from_config(_concat_wavenet_config())
    net.sample_rate = _SAMPLE_RATE
    return net


def _destinations(n=64) -> _torch.Tensor:
    _torch.manual_seed(0)
    return _torch.stack(
        [_torch.rand(n) * 10.0, _torch.randint(0, 3, (n,)).float()], dim=1
    )


def _landed(destination, **kwargs):
    defaults = dict(
        min_ramp_seconds=0.05,
        max_ramp_seconds=2.0,
        min_margin_seconds=0.02,
        max_margin_seconds=0.1,
    )
    defaults.update(kwargs)
    return _sample_landed_trajectories(
        _net().param_specs, destination, _LENGTH, _LAND_BY, _SAMPLE_RATE, **defaults
    )


def test_control_is_parked_at_the_destination_across_the_scored_window():
    """The capture is only a valid target if the move has finished before scoring starts.

    This is the whole basis of the augmentation: the device's output at a setting does not
    depend on where the knob was beforehand, so any leakage into the scored window would
    be training against a target that no longer describes the input.
    """
    destination = _destinations()
    trajectory = _landed(destination)

    scored = trajectory[:, _LAND_BY:, :]
    assert _torch.allclose(scored, destination[:, None, :].expand_as(scored), atol=1e-5)


def test_gesture_lands_within_the_configured_margin():
    destination = _destinations()
    min_margin, max_margin = 0.02, 0.1
    trajectory = _landed(
        destination, min_margin_seconds=min_margin, max_margin_seconds=max_margin
    )

    # The continuous control, whose ramp is what the margin is measured against. A switch
    # steps at the gesture's start instead, so it settles a whole ramp earlier.
    differs = (trajectory[..., 0] - destination[:, None, 0]).abs() > 1e-5
    # A row whose drawn origin happens to equal its destination never moves; that is a
    # legitimate draw, so score the margin only over the rows that do.
    moves = differs.any(dim=1)
    assert moves.sum() > len(destination) // 2
    # Last index that still differs from the destination, per row.
    last = _LENGTH - 1 - differs.flip(1).float().argmax(1)
    margin_seconds = ((_LAND_BY - (last + 1)) / _SAMPLE_RATE)[moves]

    assert (margin_seconds >= min_margin - 1e-6).all()
    assert (margin_seconds <= max_margin + 1e-6).all()


def test_switches_step_rather_than_blend():
    """A blended one-hot is a conditioning vector no model was ever trained on."""
    destination = _destinations()
    trajectory = _landed(destination)

    switch = trajectory[..., 1]
    assert _torch.equal(switch, switch.round())


def test_zero_probability_leaves_the_forward_untouched():
    """The plain constant-control path must stay bit-identical, not merely close."""
    net = _net()
    destination = _destinations(4)
    trajectory = _landed(destination, probability=0.0)
    x = _torch.randn(4, _LENGTH)

    with _torch.no_grad():
        assert _torch.equal(
            net(x, trajectory, pad_start=False), net(x, destination, pad_start=False)
        )


def test_probability_selects_a_share_of_the_batch():
    destination = _destinations(512)
    trajectory = _landed(destination, probability=0.5)

    moved = ((trajectory - destination[:, None, :]).abs().sum(-1) > 1e-5).any(dim=1)
    assert 0 < int(moved.sum()) < 512


def _module(**kwargs) -> _ParametricLightningModule:
    return _ParametricLightningModule(
        _net(), landed_move_config=_LandedMoveConfig(**kwargs)
    )


def _batch(net, n=2):
    length = net.receptive_field - 1 + 16
    return (
        _torch.randn(n, length),
        _torch.tensor([[7.5, 2.0]]).expand(n, -1).contiguous(),
        _torch.randn(n, 16),
    )


def test_augmentation_is_skipped_during_validation(mocker):
    """Validation must keep measuring the parked-control case the model ships for."""
    module = _module(min_margin_seconds=0.0, max_margin_seconds=0.0)
    module.eval()
    spy = mocker.spy(module, "_landed_move_batch")

    module._shared_step(_batch(module.net))

    spy.assert_not_called()


def test_augmentation_runs_during_training(mocker):
    module = _module(min_margin_seconds=0.0, max_margin_seconds=0.0)
    module.train()
    spy = mocker.spy(module, "_landed_move_batch")

    module._shared_step(_batch(module.net))

    spy.assert_called_once()


def test_absent_by_default(mocker):
    module = _ParametricLightningModule(_net())
    module.train()
    assert module._landed_move_config is None

    preds, _, _ = module._shared_step(_batch(module.net))

    assert preds.shape == (2, 16)


def test_margin_that_cannot_fit_the_history_is_rejected():
    """Silently doing nothing is the worst outcome for a training-time flag."""
    module = _module(min_margin_seconds=0.5)
    module.train()

    with _pytest.raises(ValueError, match="ahead of this model's first"):
        module._shared_step(_batch(module.net))


def test_requires_a_net_that_accepts_a_trajectory():
    net = _HyperWaveNet.init_from_config(_hyperwavenet_config())
    with _pytest.raises(ValueError, match="landed_move requires"):
        _ParametricLightningModule(net, landed_move_config=_LandedMoveConfig())


@_pytest.mark.parametrize(
    "config",
    [
        {"probability": 1.5},
        {"min_ramp_seconds": 0.0},
        {"min_ramp_seconds": 2.0, "max_ramp_seconds": 1.0},
        {"min_margin_seconds": 0.5, "max_margin_seconds": 0.1},
        {"rail_probability": -0.1},
        {"nonsense": 1.0},
    ],
)
def test_config_rejects_invalid(config):
    with _pytest.raises(ValueError):
        _LandedMoveConfig.from_config(config)


def test_config_absent_by_default():
    assert _LandedMoveConfig.from_config(None) is None


def test_land_by_counts_the_loss_mask_as_room():
    """A masked prefix is unscored, so a gesture may land in it.

    This is the only room a recurrent net has -- its receptive field describes per-step
    arithmetic, not memory -- and it must be added to the convolutional prefix rather than
    replacing it.
    """
    net = _net()
    # Big enough that the whole configured margin range fits inside the mask alone -- the
    # toy net's receptive field contributes only a handful of samples.
    mask = 8192
    module = _ParametricLightningModule(
        _net(),
        loss_config=_ParametricLossConfig(mask_first=mask),
        landed_move_config=_LandedMoveConfig(probability=1.0),
    )
    module.train()
    scored = 64
    length = net.receptive_field - 1 + scored + mask
    destination = _torch.tensor([[7.5, 2.0]]).expand(8, -1).contiguous()

    _, trajectory = module._landed_move_batch(
        _torch.randn(8, length), destination, scored + mask
    )

    # Nothing may still be moving once the scored region opens.
    opens = length - scored
    still = trajectory[:, opens:, :]
    assert _torch.allclose(still, destination[:, None, :].expand_as(still), atol=1e-5)


def test_a_recurrent_net_can_land_a_move_in_its_mask():
    """Landed-move needs no frozen reference, so recurrence is no obstacle -- unlike the
    quasi-static anchor, which cannot build a clean reference on a recurrent net."""
    specs = [
        _ParamSpec(name="a", min=0.0, max=10.0, default=5.0, type="continuous"),
        _ParamSpec(name="b", min=0.0, max=10.0, default=5.0, type="continuous"),
    ]
    net = ConcatLSTM(param_specs=specs, hidden_size=4, num_layers=1)
    net.sample_rate = _SAMPLE_RATE
    mask = 8192
    module = _ParametricLightningModule(
        net,
        loss_config=_ParametricLossConfig(mask_first=mask),
        landed_move_config=_LandedMoveConfig(probability=1.0),
    )
    module.train()
    length = mask + 512
    destination = _torch.tensor([[4.0, 4.5]]).expand(4, -1).contiguous()

    _, trajectory = module._landed_move_batch(
        _torch.randn(4, length), destination, length
    )

    still = trajectory[:, mask:, :]
    assert _torch.allclose(still, destination[:, None, :].expand_as(still), atol=1e-5)
