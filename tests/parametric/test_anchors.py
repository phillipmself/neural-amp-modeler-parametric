import pytest as _pytest
import torch as _torch

from nam.models.parametric import ConcatWaveNet as _ConcatWaveNet
from nam.models.parametric import FiLMWaveNet as _FiLMWaveNet
from nam.models.parametric import HyperWaveNet as _HyperWaveNet
from nam.models.parametric._anchors import anchor_output as _anchor_output
from nam.models.parametric._anchors import (
    sample_param_trajectories as _sample_param_trajectories,
)
from nam.models.parametric._anchors import _sample_endpoints as _sample_endpoints
from nam.models.parametric._anchors import sample_raw_params as _sample_raw_params
from nam.train.parametric import _ParametricLightningModule as _ParametricLightningModule
from nam.train.parametric import _ParametricLossConfig as _ParametricLossConfig
from nam.train.parametric import _SilenceAnchorConfig as _SilenceAnchorConfig
from nam.train.parametric import (
    _SilenceAnchorRampConfig as _SilenceAnchorRampConfig,
)
from nam.train.parametric import _silence_anchor_norm as _silence_anchor_norm
from tests.parametric.test_concat_wavenet import _concat_wavenet_config
from tests.parametric.test_film_wavenet import _film_wavenet_config
from tests.parametric.test_hyperwavenet import _hyperwavenet_config


def _net() -> _ConcatWaveNet:
    return _ConcatWaveNet.init_from_config(_concat_wavenet_config())


def test_sample_raw_params_respects_spec_ranges():
    net = _net()
    params = _sample_raw_params(net.param_specs, 256)

    assert params.shape == (256, net.param_dim)
    for i, spec in enumerate(net.param_specs):
        column = params[:, i]
        assert (column >= spec.min).all() and (column <= spec.max).all()
        if spec.type == "switch":
            # Switch values index a one-hot, so anything non-integral would fail encoding.
            assert (column == column.round()).all()

    # Sampled values must actually vary; a constant draw would anchor one setting only.
    assert params[:, 0].std() > 0.0


def test_sample_raw_params_encodes():
    net = _net()
    params = _sample_raw_params(net.param_specs, 8)
    encoded = net._encode_params(params)
    assert encoded.shape == (8, net.encoded_param_dim)
    assert _torch.isfinite(encoded).all()


def test_anchor_input_is_silence(mocker):
    net = _net()
    spy = mocker.spy(net, "forward")
    _anchor_output(net, _sample_raw_params(net.param_specs, 3), ny=16)

    x = spy.call_args.args[0]
    assert (x == 0.0).all()
    # The receptive-field history is part of the window, not padded on afterwards.
    assert x.shape == (3, net.receptive_field - 1 + 16)
    assert spy.call_args.kwargs["pad_start"] is False


def test_anchor_output_shape_and_grad():
    net = _net()
    params = _sample_raw_params(net.param_specs, 4)
    y = _anchor_output(net, params, ny=32)

    assert y.shape == (4, 32)
    y.square().mean().backward()
    assert any(p.grad is not None and _torch.isfinite(p.grad).all() for p in net.parameters())


def test_anchor_output_depends_on_params():
    """The anchor would be vacuous if silence-in output did not vary with the controls."""
    net = _net()
    low = _torch.tensor([[0.0, 0.0]])
    high = _torch.tensor([[10.0, 2.0]])
    assert not _torch.allclose(
        _anchor_output(net, low, ny=32), _anchor_output(net, high, ny=32)
    )


def test_anchor_row_wise_under_uniform_batch_promise():
    """A net running under the uniform-batch promise must not collapse distinct settings
    onto the first row's setting."""
    net = _HyperWaveNet.init_from_config(_hyperwavenet_config())
    params = _sample_raw_params(net.param_specs, 3)
    reference = _torch.cat(
        [net(_torch.zeros((1, net.receptive_field - 1 + 8)), params[i], pad_start=False)
         for i in range(3)],
        dim=0,
    )

    net.set_uniform_batch_params(True)
    assert net.requires_uniform_batch_params
    y = _anchor_output(net, params, ny=8)

    assert _torch.allclose(y, reference)
    # The per-row calls are unbatched, so the promise is still unverified and remains
    # available to catch a genuinely mixed training batch.
    assert not net._uniform_batch_params_checked


def test_silence_anchor_absent_by_default():
    config = _ParametricLossConfig.parse_config({})
    assert config["silence_anchor"] is None


def test_silence_anchor_not_computed_when_unconfigured(mocker):
    module = _ParametricLightningModule(_net(), loss_config=_ParametricLossConfig())
    mocked = mocker.patch.object(module, "_silence_anchor_loss")
    module.train()

    loss_dict = module._get_loss_dict(_torch.randn((2, 64)), _torch.randn((2, 64)))

    mocked.assert_not_called()
    assert "Silence Anchor" not in loss_dict


def test_silence_anchor_in_training_loss_dict():
    module = _ParametricLightningModule(
        _net(),
        loss_config=_ParametricLossConfig(
            silence_anchor=_SilenceAnchorConfig(weight=0.5, batch_size=2, ny=16)
        ),
    )
    module.train()

    loss_dict = module._get_loss_dict(_torch.randn((2, 64)), _torch.randn((2, 64)))

    item = loss_dict["Silence Anchor"]
    assert item.weight == _pytest.approx(0.5)
    assert item.value.ndim == 0 and item.value >= 0.0


def test_silence_anchor_skipped_during_validation():
    module = _ParametricLightningModule(
        _net(),
        loss_config=_ParametricLossConfig(
            silence_anchor=_SilenceAnchorConfig(weight=0.5, batch_size=2, ny=16)
        ),
    )
    module.eval()

    loss_dict = module._get_loss_dict(_torch.randn((2, 64)), _torch.randn((2, 64)))

    assert "Silence Anchor" not in loss_dict


def _both_anchors(ny=16, ramp_ny=None, weight=0.5, ramp_weight=0.25):
    return _ParametricLossConfig(
        silence_anchor=_SilenceAnchorConfig(weight=weight, batch_size=2, ny=ny),
        silence_anchor_ramp=_SilenceAnchorRampConfig(
            weight=ramp_weight, batch_size=2, ny=ny if ramp_ny is None else ramp_ny
        ),
    )


def test_merged_anchors_match_running_them_separately():
    """The shared forward is an optimization, so it must not move either term.

    Both anchors feed the same silence over the same window and differ only in the
    control tensor, so concatenating them into one forward is exact. Seeding the two
    routes identically makes them draw the same controls.
    """
    config = _both_anchors()
    module = _ParametricLightningModule(_net(), loss_config=config)
    module.train()

    _torch.manual_seed(0)
    merged = {key: item.value for key, item in module._silence_anchor_items().items()}
    _torch.manual_seed(0)
    separate = {
        "Silence Anchor": module._silence_anchor_loss(config.silence_anchor),
        "Silence Anchor Ramp": module._silence_anchor_ramp_loss(
            config.silence_anchor_ramp
        ),
    }

    assert set(merged) == set(separate)
    for key, value in merged.items():
        assert value == separate[key], key


def test_merged_anchors_run_one_forward(mocker):
    module = _ParametricLightningModule(_net(), loss_config=_both_anchors())
    module.train()
    spy = mocker.spy(module.net, "forward")

    module._silence_anchor_items()

    assert spy.call_count == 1


def test_anchors_fall_back_to_separate_forwards_when_windows_differ(mocker):
    """A shared forward needs a shared window; mismatched ny must still score both."""
    module = _ParametricLightningModule(
        _net(), loss_config=_both_anchors(ny=16, ramp_ny=24)
    )
    module.train()
    spy = mocker.spy(module.net, "forward")

    items = module._silence_anchor_items()

    assert sorted(items) == ["Silence Anchor", "Silence Anchor Ramp"]
    assert spy.call_count == 2


def test_zero_weight_anchor_is_not_run(mocker):
    """A zero weight cancels the term, so paying for its forward is pure waste."""
    module = _ParametricLightningModule(
        _net(), loss_config=_both_anchors(weight=0.0)
    )
    module.train()
    spy = mocker.spy(module.net, "forward")

    items = module._silence_anchor_items()

    assert list(items) == ["Silence Anchor Ramp"]
    assert spy.call_count == 1


def test_silence_anchor_norm_does_not_go_quiet_as_the_offset_shrinks():
    """
    The whole job of the anchor is to keep pulling once the residual is already small, so
    the term must stay first-order in the offset. A squared term halves its gradient every
    time the offset halves, which is why it cannot finish the job.
    """
    offsets = [1e-1, 1e-2, 1e-3, 1e-4]
    values = []
    grads = []
    for offset in offsets:
        y = _torch.full((4, 64), offset, requires_grad=True)
        loss = _silence_anchor_norm(y)
        loss.backward()
        values.append(loss.item())
        grads.append(y.grad.abs().mean().item())

    # Loss tracks the offset itself, not its square.
    for offset, value in zip(offsets, values):
        assert value == _pytest.approx(offset, rel=1e-5)
    # And the pull per sample is the same at 1e-4 as at 1e-1.
    assert grads[0] == _pytest.approx(grads[-1], rel=1e-5)


def test_silence_anchor_norm_penalises_a_zero_mean_transient():
    """``mean(|y|)``, not ``|mean(y)|``: a moving control produces an excursion whose mean
    can cancel while the sound of it does not."""
    zero_mean = _torch.cat([_torch.full((32,), 0.02), _torch.full((32,), -0.02)])[None]

    assert zero_mean.mean().abs().item() == _pytest.approx(0.0, abs=1e-7)
    assert _silence_anchor_norm(zero_mean).item() == _pytest.approx(0.02, rel=1e-5)


def test_silence_anchor_config_parsing():
    parsed = _ParametricLossConfig.parse_config(
        {"silence_anchor": {"weight": 2.0, "batch_size": 4, "ny": 128}}
    )
    anchor = parsed["silence_anchor"]
    assert anchor.weight == _pytest.approx(2.0)
    assert anchor.batch_size == 4
    assert anchor.ny == 128


@_pytest.mark.parametrize(
    "config",
    [
        {},
        {"weight": 1.0, "batch_size": 0},
        {"weight": 1.0, "ny": 0},
        {"weight": -1.0},
        {"weight": 1.0, "typo": 3},
    ],
)
def test_silence_anchor_config_rejects_invalid(config):
    with _pytest.raises(ValueError):
        _SilenceAnchorConfig.from_config(config)


# --- Moving controls -------------------------------------------------------------------


def _trajectories(n=64, length=256, **kwargs):
    net = _net()
    defaults = dict(min_ramp_seconds=0.001, max_ramp_seconds=0.02)
    defaults.update(kwargs)
    return net, _sample_param_trajectories(
        net.param_specs,
        n,
        length,
        48_000.0,
        generator=_torch.Generator().manual_seed(0),
        **defaults,
    )


def _film_net() -> _FiLMWaveNet:
    return _FiLMWaveNet.init_from_config(_film_wavenet_config(encoder=True))


def test_film_wavenet_accepts_a_trajectory():
    """FiLM reads the controls through 1x1 convolutions, which have no time extent, so a
    control that moves within the window needs no special handling -- and the anchors need
    that to be true in order to score a knob being turned."""
    net = _film_net()
    assert net.supports_param_trajectory


def test_film_wavenet_constant_trajectory_matches_the_static_call():
    """Same tie-down as the concat case: the moving-control route must reduce to the
    established held-setting route when nothing actually moves."""
    net = _film_net()
    net.eval()
    length = net.receptive_field + 32
    x = _torch.randn((2, length))
    params = _torch.tensor([[2.0, 0.0], [9.0, 2.0]])
    trajectory = params[:, None, :].expand(-1, length, -1)

    with _torch.no_grad():
        assert _torch.allclose(
            net(x, params, pad_start=False),
            net(x, trajectory, pad_start=False),
            atol=1e-6,
        )


@_pytest.mark.parametrize("moving", [False, True])
def test_film_wavenet_anchors_run_and_carry_gradient(moving):
    """The anchor is the whole point of the trajectory support for this net: silence in at
    a control setting -- held or moving -- must produce a scored output that the model can
    actually be trained against."""
    net = _film_net()
    net.train()
    ny = 16
    if moving:
        params = _sample_param_trajectories(
            net.param_specs,
            2,
            net.receptive_field - 1 + ny,
            net.sample_rate,
            min_ramp_seconds=0.05,
            max_ramp_seconds=2.0,
        )
    else:
        params = _sample_raw_params(net.param_specs, 2)

    loss = _anchor_output(net, params, ny).abs().mean()
    loss.backward()

    assert loss.requires_grad
    assert any(
        p.grad is not None and _torch.any(p.grad != 0.0) for p in net.parameters()
    )


def test_constant_trajectory_matches_the_static_call():
    """A trajectory that never moves must be the (B, P) path exactly; this is what ties
    the new conditioning route to the established one."""
    net = _net()
    net.eval()
    x = _torch.randn((3, 64))
    params = _torch.tensor([[2.0, 0.0], [7.0, 1.0], [9.0, 2.0]])
    trajectory = params[:, None, :].expand(-1, 64, -1)

    with _torch.no_grad():
        for pad_start in (False, True):
            assert _torch.allclose(
                net(x, params, pad_start=pad_start),
                net(x, trajectory, pad_start=pad_start),
                atol=1e-6,
            )


def test_pad_start_extends_a_trajectory_with_its_first_frame():
    """The control was sitting at its starting value before the window opened, so the
    receptive-field history must not see some other value."""
    net = _net()
    net.eval()
    history = net.receptive_field - 1
    x = _torch.randn((2, 64))
    _, trajectory = _trajectories(n=2, length=64)

    padded_x = _torch.cat([_torch.zeros((2, history)), x], dim=1)
    padded_params = _torch.cat(
        [trajectory[:, :1].expand(-1, history, -1), trajectory], dim=1
    )
    with _torch.no_grad():
        assert _torch.allclose(
            net(x, trajectory, pad_start=True),
            net(padded_x, padded_params, pad_start=False),
            atol=1e-6,
        )


def test_trajectory_shape_and_ranges():
    net, trajectory = _trajectories()
    assert trajectory.shape == (64, 256, net.param_dim)
    for i, spec in enumerate(net.param_specs):
        column = trajectory[:, :, i]
        assert (column >= spec.min).all() and (column <= spec.max).all()


def test_continuous_controls_ramp_monotonically_between_endpoints():
    _, trajectory = _trajectories()
    continuous = trajectory[:, :, 0]
    delta = continuous[:, 1:] - continuous[:, :-1]
    # A single gesture per window: each control travels one way only.
    assert ((delta >= -1e-6).all(dim=1) | (delta <= 1e-6).all(dim=1)).all()
    # And some of them actually travel, or the anchor would be the static one.
    assert (continuous[:, 0] != continuous[:, -1]).any()


def test_switches_step_rather_than_blend():
    """The runtime excludes switch channels from smoothing, so a switch must jump between
    integral indices and never take an intermediate value."""
    _, trajectory = _trajectories()
    switch = trajectory[:, :, 1]

    assert (switch == switch.round()).all()
    transitions = (switch[:, 1:] != switch[:, :-1]).sum(dim=1)
    assert (transitions <= 1).all()
    assert (transitions == 1).any()


def test_switch_step_and_continuous_ramp_share_one_commit_instant():
    """The runtime commits the whole control vector at once, so a gesture that moves a
    switch and a knob starts both at the same sample."""
    _, trajectory = _trajectories(n=200, length=300)
    switch, continuous = trajectory[:, :, 1], trajectory[:, :, 0]

    compared = 0
    for row in range(trajectory.shape[0]):
        stepped = (switch[row, 1:] != switch[row, :-1]).nonzero()
        departed = (continuous[row] != continuous[row, 0]).nonzero()
        if len(stepped) == 0 or len(departed) == 0:
            continue
        compared += 1
        assert stepped[0, 0].item() + 1 == departed[0, 0].item()
    assert compared > 0


def test_some_gestures_settle_inside_the_window():
    """The scored tail after a ramp lands is where the settling artifact lives; if no
    sampled gesture ever finished in-window it would never be trained on."""
    _, trajectory = _trajectories(n=256, length=256)
    continuous = trajectory[:, :, 0]
    moving = continuous[:, 0] != continuous[:, -1]
    settled = (continuous[:, -1] == continuous[:, -2]) & moving
    assert settled.any()


def test_gestures_start_and_end_on_the_rails():
    """Users park knobs fully off and fully up; a purely uniform draw would almost never
    put a gesture endpoint there. Tested on the endpoints themselves rather than on the
    trajectory, whose first frame is mid-gesture whenever the move began before the
    window opened."""
    net = _net()
    spec = net.param_specs[0]
    generator = _torch.Generator().manual_seed(0)

    on_rails = _sample_endpoints(net.param_specs, 256, 1.0, None, generator)[:, 0]
    assert ((on_rails == spec.min) | (on_rails == spec.max)).all()
    assert (on_rails == spec.min).any() and (on_rails == spec.max).any()

    off_rails = _sample_endpoints(net.param_specs, 256, 0.0, None, generator)[:, 0]
    assert not ((off_rails == spec.min) | (off_rails == spec.max)).any()


def test_windows_cover_both_moving_and_settled_gestures():
    """Every gesture moves at least one control, but a window can open after the move
    finished or close before it starts -- both are things the runtime does, and the
    settled case is what keeps a moving-control anchor from drifting off the static one.
    """
    _, trajectory = _trajectories(n=256)
    moving = (trajectory[:, 0, :] != trajectory[:, -1, :]).any(dim=1)

    assert moving.float().mean() > 0.5
    assert (~moving).any()


@_pytest.mark.parametrize(
    "kwargs",
    [
        {"min_ramp_seconds": 0.0},
        {"min_ramp_seconds": 1.0, "max_ramp_seconds": 0.5},
        {"rail_probability": 1.5},
    ],
)
def test_trajectory_sampler_rejects_invalid(kwargs):
    with _pytest.raises(ValueError):
        _trajectories(n=2, length=8, **kwargs)


def test_trajectory_rejected_by_a_net_that_cannot_use_it():
    net = _HyperWaveNet.init_from_config(_hyperwavenet_config())
    assert not net.supports_param_trajectory
    with _pytest.raises(NotImplementedError):
        net(
            _torch.zeros((1, net.receptive_field)),
            _torch.zeros((1, net.receptive_field, net.param_dim)),
            pad_start=False,
        )


def test_ramp_anchor_rejected_at_construction_for_such_a_net():
    with _pytest.raises(ValueError, match="param trajectory"):
        _ParametricLightningModule(
            _HyperWaveNet.init_from_config(_hyperwavenet_config()),
            loss_config=_ParametricLossConfig(
                silence_anchor_ramp=_SilenceAnchorRampConfig(weight=1.0)
            ),
        )


def test_ramp_anchor_absent_by_default():
    assert _ParametricLossConfig.parse_config({})["silence_anchor_ramp"] is None


def test_ramp_anchor_not_computed_when_unconfigured(mocker):
    module = _ParametricLightningModule(_net(), loss_config=_ParametricLossConfig())
    mocked = mocker.patch.object(module, "_silence_anchor_ramp_loss")
    module.train()

    loss_dict = module._get_loss_dict(_torch.randn((2, 64)), _torch.randn((2, 64)))

    mocked.assert_not_called()
    assert "Silence Anchor Ramp" not in loss_dict


def test_ramp_anchor_in_training_loss_dict():
    net = _net()
    net.sample_rate = 48_000.0
    module = _ParametricLightningModule(
        net,
        loss_config=_ParametricLossConfig(
            silence_anchor_ramp=_SilenceAnchorRampConfig(
                weight=0.25, batch_size=2, ny=16
            )
        ),
    )
    module.train()

    loss_dict = module._get_loss_dict(_torch.randn((2, 64)), _torch.randn((2, 64)))

    item = loss_dict["Silence Anchor Ramp"]
    assert item.weight == _pytest.approx(0.25)
    assert item.value.ndim == 0 and item.value >= 0.0
    item.value.backward()
    assert any(p.grad is not None and _torch.isfinite(p.grad).all() for p in net.parameters())


def test_ramp_anchor_skipped_during_validation():
    net = _net()
    net.sample_rate = 48_000.0
    module = _ParametricLightningModule(
        net,
        loss_config=_ParametricLossConfig(
            silence_anchor_ramp=_SilenceAnchorRampConfig(weight=0.25, batch_size=2, ny=16)
        ),
    )
    module.eval()

    assert "Silence Anchor Ramp" not in module._get_loss_dict(
        _torch.randn((2, 64)), _torch.randn((2, 64))
    )


def test_ramp_anchor_config_parsing():
    parsed = _ParametricLossConfig.parse_config(
        {
            "silence_anchor_ramp": {
                "weight": 1.5,
                "batch_size": 4,
                "ny": 64,
                "min_ramp_seconds": 0.01,
                "max_ramp_seconds": 3.0,
                "rail_probability": 0.5,
            }
        }
    )["silence_anchor_ramp"]

    assert parsed.weight == _pytest.approx(1.5)
    assert parsed.batch_size == 4 and parsed.ny == 64
    assert parsed.min_ramp_seconds == _pytest.approx(0.01)
    assert parsed.max_ramp_seconds == _pytest.approx(3.0)
    assert parsed.rail_probability == _pytest.approx(0.5)


@_pytest.mark.parametrize(
    "config",
    [
        {"weight": 1.0, "min_ramp_seconds": 0.0},
        {"weight": 1.0, "min_ramp_seconds": 2.0, "max_ramp_seconds": 1.0},
        {"weight": 1.0, "rail_probability": -0.1},
        {"weight": 1.0, "typo": 1},
    ],
)
def test_ramp_anchor_config_rejects_invalid(config):
    with _pytest.raises(ValueError):
        _SilenceAnchorRampConfig.from_config(config)
