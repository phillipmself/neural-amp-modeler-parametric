import pytest as _pytest
import torch as _torch

from nam.models.parametric import ConcatWaveNet as _ConcatWaveNet
from nam.models.parametric import HyperWaveNet as _HyperWaveNet
from nam.models.parametric._anchors import as_held_trajectory as _as_held_trajectory
from nam.models.parametric._anchors import (
    sample_param_trajectories as _sample_param_trajectories,
)
from nam.models.parametric._quasi_static import (
    quasi_static_loss as _quasi_static_loss,
)
from nam.models.parametric._quasi_static import (
    quasi_static_reference as _quasi_static_reference,
)
from nam.train.parametric import _ParametricLossConfig as _ParametricLossConfig
from nam.train.parametric import _ParametricLightningModule as _ParametricLightningModule
from nam.train.parametric import (
    _QuasiStaticAnchorConfig as _QuasiStaticAnchorConfig,
)
from tests.parametric.test_concat_wavenet import _concat_wavenet_config
from tests.parametric.test_hyperwavenet import _hyperwavenet_config

_SAMPLE_RATE = 48000.0
_NY = 32
_BLOCK = 8


def _net() -> _ConcatWaveNet:
    net = _ConcatWaveNet.init_from_config(_concat_wavenet_config())
    net.sample_rate = _SAMPLE_RATE
    return net


def _window(net, batch=3):
    _torch.manual_seed(0)
    return _torch.randn(batch, net.receptive_field - 1 + _NY) * 0.2


def _held(batch=3):
    return _torch.tensor([[7.5, 2.0], [1.0, 0.0], [4.0, 1.0]])[:batch]


def _moving(net, length, batch=3):
    return _sample_param_trajectories(
        net.param_specs,
        batch,
        length,
        _SAMPLE_RATE,
        min_ramp_seconds=0.002,
        max_ramp_seconds=0.02,
    )


def test_a_control_that_never_moves_scores_exactly_zero():
    """The anchor's whole claim is that its residual is dynamic artifact and nothing else.

    With no control motion the moving render and the frozen reference are the same
    computation, so any non-zero value here would be static mismatch leaking in.
    """
    net = _net()
    x = _window(net)
    constant = _as_held_trajectory(_held(), x.shape[1])

    assert _quasi_static_loss(net, x, constant, _NY, _BLOCK) == 0.0


def test_reference_matches_a_plain_static_render():
    """The reference must be the model's own static response, not an approximation of it."""
    net = _net()
    x = _window(net)
    constant = _as_held_trajectory(_held(), x.shape[1])

    with _torch.no_grad():
        reference = _quasi_static_reference(net, x, constant, _NY, _BLOCK)
        plain = net(x, _held(), pad_start=False)

    assert _torch.equal(reference, plain)


def test_a_moving_control_scores_above_zero_and_carries_gradient():
    net = _net()
    x = _window(net)
    trajectory = _moving(net, x.shape[1])

    loss = _quasi_static_loss(net, x, trajectory, _NY, _BLOCK)
    loss.backward()

    assert loss > 0.0
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0.0 for g in grads)


def test_reference_carries_no_gradient():
    """The reference is a target; letting it pull would let the model move the goalposts."""
    net = _net()
    x = _window(net)
    trajectory = _moving(net, x.shape[1])

    reference = _quasi_static_reference(net, x, trajectory, _NY, _BLOCK)

    assert not reference.requires_grad


def test_anchor_still_varies_with_the_controls():
    """It must not be satisfiable by killing knob authority.

    The reference tracks the controls too, so a model that ignored them entirely would
    score zero here -- but the capture loss pins that variation, and this test guards the
    reference against being accidentally decoupled from the controls.
    """
    net = _net()
    x = _window(net, batch=1)
    low = _as_held_trajectory(_torch.tensor([[0.0, 0.0]]), x.shape[1])
    high = _as_held_trajectory(_torch.tensor([[10.0, 2.0]]), x.shape[1])

    with _torch.no_grad():
        assert not _torch.allclose(
            _quasi_static_reference(net, x, low, _NY, _BLOCK),
            _quasi_static_reference(net, x, high, _NY, _BLOCK),
        )


@_pytest.mark.parametrize("block", [1, 2, 4, 8, 16, 32])
def test_every_block_width_that_divides_ny_is_accepted(block):
    net = _net()
    x = _window(net)
    trajectory = _moving(net, x.shape[1])

    assert _quasi_static_loss(net, x, trajectory, _NY, block) >= 0.0


def test_block_must_divide_ny():
    net = _net()
    x = _window(net)
    trajectory = _moving(net, x.shape[1])

    with _pytest.raises(ValueError, match="divide"):
        _quasi_static_loss(net, x, trajectory, _NY, 7)


def test_input_length_must_match_ny():
    net = _net()
    x = _window(net)
    trajectory = _moving(net, x.shape[1])

    # Divisible by the block, so the length check is what has to catch this.
    with _pytest.raises(ValueError, match="samples for ny"):
        _quasi_static_reference(net, x, trajectory, _NY * 2, _BLOCK)


def _module(**kwargs):
    config = dict(weight=0.5, batch_size=2, ny=_NY, block=_BLOCK)
    config.update(kwargs)
    return _ParametricLightningModule(
        _net(),
        loss_config=_ParametricLossConfig(
            quasi_static_anchor=_QuasiStaticAnchorConfig(**config)
        ),
    )


def _batch(net, batch=2):
    length = net.receptive_field - 1 + _NY
    return (
        _torch.randn(batch, length),
        _torch.tensor([[7.5, 2.0]]).expand(batch, -1).contiguous(),
        _torch.randn(batch, _NY),
    )


def test_anchor_appears_in_the_training_loss_dict():
    module = _module()
    module.train()

    _, _, loss_dict = module._shared_step(_batch(module.net))

    item = loss_dict["Quasi-Static Anchor"]
    assert item.weight == _pytest.approx(0.5)
    assert item.value.ndim == 0 and item.value >= 0.0


def test_anchor_is_skipped_during_validation():
    module = _module()
    module.eval()

    _, _, loss_dict = module._shared_step(_batch(module.net))

    assert "Quasi-Static Anchor" not in loss_dict


def test_zero_weight_anchor_is_not_run(mocker):
    module = _module(weight=0.0)
    module.train()
    spy = mocker.spy(module, "_quasi_static_anchor_loss")

    module._shared_step(_batch(module.net))

    spy.assert_not_called()


def test_batch_input_is_released_after_the_step():
    """Holding the batch would keep its graph alive for the whole epoch."""
    module = _module()
    module.train()

    module._shared_step(_batch(module.net))

    assert module._batch_input is None


def test_a_batch_too_small_for_the_anchor_is_rejected():
    module = _module(batch_size=8)
    module.train()

    with _pytest.raises(ValueError, match="lower its"):
        module._shared_step(_batch(module.net, batch=2))


def test_requires_a_net_that_accepts_a_trajectory():
    net = _HyperWaveNet.init_from_config(_hyperwavenet_config())
    with _pytest.raises(ValueError, match="quasi_static_anchor requires"):
        _ParametricLightningModule(
            net,
            loss_config=_ParametricLossConfig(
                quasi_static_anchor=_QuasiStaticAnchorConfig(weight=0.5)
            ),
        )


def test_absent_by_default():
    assert _ParametricLossConfig.parse_config({})["quasi_static_anchor"] is None


def test_config_parsing():
    parsed = _QuasiStaticAnchorConfig.from_config(
        {"weight": 0.25, "batch_size": 4, "ny": 128, "block": 32}
    )
    assert parsed.weight == _pytest.approx(0.25)
    assert parsed.block == 32
    # Inherited from the ramp anchor, since the gesture model is the same.
    assert parsed.max_ramp_seconds > parsed.min_ramp_seconds > 0.0


@_pytest.mark.parametrize(
    "config",
    [
        {"weight": 1.0, "ny": 128, "block": 7},
        {"weight": 1.0, "block": 0},
        {"weight": -1.0},
        {"weight": 1.0, "nonsense": 1},
    ],
)
def test_config_rejects_invalid(config):
    with _pytest.raises(ValueError):
        _QuasiStaticAnchorConfig.from_config(config)
