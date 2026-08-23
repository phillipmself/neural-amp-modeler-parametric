import math as _math

import pytest as _pytest
import torch as _torch

from nam.models.parametric import ConcatLSTM as _ConcatLSTM
from nam.models.parametric import ConcatWaveNet as _ConcatWaveNet
from nam.models.parametric import HyperWaveNet as _HyperWaveNet
from nam.models.parametric._anchors import anchor_output as _anchor_output
from nam.models.parametric._anchors import anchor_window as _anchor_window
from nam.models.parametric._anchors import (
    as_held_trajectory as _as_held_trajectory,
)
from nam.models.parametric._anchors import (
    sample_param_trajectories as _sample_param_trajectories,
)
from nam.models.parametric._anchors import sample_raw_params as _sample_raw_params
from nam.train.parametric import _ParametricLightningModule as _ParametricLightningModule
from nam.train.parametric import _ParametricLossConfig as _ParametricLossConfig
from nam.train.parametric import _SilenceAnchorConfig as _SilenceAnchorConfig
from nam.train.parametric import _SilenceAnchorRampConfig as _SilenceAnchorRampConfig
from nam.train.parametric import _silence_anchor_norm as _silence_anchor_norm
from tests.parametric.test_concat_lstm import _concat_lstm_config
from tests.parametric.test_concat_wavenet import _concat_wavenet_config
from tests.parametric.test_hyperwavenet import _hyperwavenet_config


def _net() -> _ConcatWaveNet:
    return _ConcatWaveNet.init_from_config(_concat_wavenet_config())


def _concat_nets():
    return [
        _ConcatWaveNet.init_from_config(_concat_wavenet_config()),
        _ConcatLSTM.init_from_config(_concat_lstm_config()),
    ]


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


def test_anchor_input_is_silence(mocker):
    net = _net()
    spy = mocker.spy(net, "forward")
    _anchor_output(net, _sample_raw_params(net.param_specs, 3), ny=16)

    x = spy.call_args.args[0]
    assert (x == 0.0).all()
    # The receptive-field history is part of the window, not padded on afterwards.
    assert x.shape == (3, net.receptive_field - 1 + 16)
    assert spy.call_args.kwargs["pad_start"] is False


def _burn_in_lstm() -> _ConcatLSTM:
    """A ConcatLSTM shaped like the shipping config: a burn-in long enough to swallow a
    naively sized anchor window."""
    config = _concat_lstm_config()
    config["train_burn_in"] = 64
    config["train_truncate"] = 64
    return _ConcatLSTM.init_from_config(config)


def test_anchor_window_covers_a_detached_burn_in():
    """
    A recurrent net detaches its burn-in, so an anchor window sized off the receptive
    field alone leaves nothing scored: the whole window is consumed by the burn-in, and
    the only gradient that survives is the output head's, downstream of the detach. The
    window has to budget for the warmup, and only the scored tail may be returned.
    """
    net = _burn_in_lstm()
    net.train()
    ny = 32
    assert net.training_warmup >= ny  # Otherwise this asserts nothing.

    y = _anchor_output(net, _sample_raw_params(net.param_specs, 2), ny=ny)
    assert y.shape == (2, ny)
    _silence_anchor_norm(y).backward()

    recurrent = {
        name: parameter
        for name, parameter in net.named_parameters()
        if not name.startswith("_head")
    }
    assert recurrent
    assert any(
        parameter.grad is not None and (parameter.grad != 0.0).any()
        for parameter in recurrent.values()
    )


def test_anchor_scores_only_the_post_warmup_tail():
    """The burn-in is warmup, not signal; scoring it would average gradient-free samples
    into the number the anchor reports."""
    net = _burn_in_lstm()
    net.eval()
    params = _sample_raw_params(net.param_specs, 2)
    full = net(
        _torch.zeros((2, net.training_warmup + 32)), params, pad_start=False
    )

    assert _torch.allclose(_anchor_output(net, params, ny=32), full[:, -32:])


@_pytest.mark.parametrize("net", _concat_nets())
def test_anchor_gradient_reaches_the_control_conditioning(net):
    """
    The anchor exists to shape how the net responds to its controls, so the term has to
    be differentiable all the way back to the control values. Scoring the loss against
    the net's weights alone would still pass if conditioning were detached.
    """
    params = _sample_raw_params(net.param_specs, 4).requires_grad_(True)
    y = _anchor_output(net, params, ny=32)

    assert y.shape == (4, 32)
    _silence_anchor_norm(y).backward()
    assert params.grad is not None and _torch.isfinite(params.grad).all()
    # Switches encode through a non-differentiable one-hot, so only the continuous
    # controls can carry a gradient.
    continuous = [i for i, spec in enumerate(net.param_specs) if spec.type != "switch"]
    assert (params.grad[:, continuous] != 0.0).any()


@_pytest.mark.parametrize("net", _concat_nets())
def test_anchor_output_depends_on_params(net):
    """The anchor would be vacuous if silence-in output did not vary with the controls."""
    lows = _torch.tensor([[spec.min for spec in net.param_specs]])
    highs = _torch.tensor([[spec.max for spec in net.param_specs]])
    assert not _torch.allclose(
        _anchor_output(net, lows, ny=32), _anchor_output(net, highs, ny=32)
    )


def test_anchor_row_wise_under_uniform_batch_promise():
    """A net running under the uniform-batch promise must not collapse distinct settings
    onto the first row's setting."""
    net = _HyperWaveNet.init_from_config(_hyperwavenet_config())
    params = _sample_raw_params(net.param_specs, 3)
    reference = _torch.cat(
        [
            net(
                _torch.zeros((1, net.receptive_field - 1 + 8)),
                params[i],
                pad_start=False,
            )
            for i in range(3)
        ],
        dim=0,
    )

    net.set_uniform_batch_params(True)
    assert net.requires_uniform_batch_params
    y = _anchor_output(net, params, ny=8)

    assert _torch.allclose(y, reference)
    # The per-row calls are unbatched, so the promise is still unverified and remains
    # available to catch a genuinely mixed training batch.
    assert not net._uniform_batch_params_checked


def test_silence_anchor_norm_does_not_go_quiet_as_the_offset_shrinks():
    """
    The anchor's job is to keep pulling once the residual is already small, so the term
    must stay first-order in the offset. A squared term halves its gradient every time
    the offset halves, which is why it cannot finish the job.
    """
    grads = []
    for offset in (1e-1, 1e-2, 1e-3, 1e-4):
        y = _torch.full((4, 64), offset, requires_grad=True)
        _silence_anchor_norm(y).backward()
        grads.append(y.grad.abs().mean().item())

    assert all(g == _pytest.approx(grads[0]) for g in grads)


def test_silence_anchor_absent_by_default():
    config = _ParametricLossConfig.parse_config({})
    assert config["silence_anchor"] is None


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


def test_silence_anchor_reported_as_an_epoch_metric():
    """An anchor whose value is never logged is an anchor whose failure is invisible."""
    from nam.train.parametric import _BUCKET_MEAN_METRICS, _bucket_means

    assert "Silence Anchor" in _BUCKET_MEAN_METRICS
    module = _ParametricLightningModule(
        _net(),
        loss_config=_ParametricLossConfig(
            silence_anchor=_SilenceAnchorConfig(weight=0.5, batch_size=2, ny=16)
        ),
    )
    module.train()

    loss_dict = module._get_loss_dict(_torch.randn((2, 64)), _torch.randn((2, 64)))
    assert "Silence Anchor" in _bucket_means(loss_dict)


def test_silence_anchor_not_computed_at_zero_weight():
    """Weight zero is how a config disables the term; it must not still cost a forward."""
    module = _ParametricLightningModule(
        _net(),
        loss_config=_ParametricLossConfig(
            silence_anchor=_SilenceAnchorConfig(weight=0.0, batch_size=2, ny=16)
        ),
    )
    module.train()

    loss_dict = module._get_loss_dict(_torch.randn((2, 64)), _torch.randn((2, 64)))

    assert "Silence Anchor" not in loss_dict


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


# The shipped defaults, so the sampler tests exercise the regime training runs in.
_DEFAULT_TRAVEL = {
    "min_window_travel": _SilenceAnchorRampConfig.min_window_travel,
    "max_window_travel": _SilenceAnchorRampConfig.max_window_travel,
}


def _travel_fraction(trajectories, param_specs, window):
    """Per row, the largest fraction of any control's full range that the row traverses
    where the loss can see it.

    Measured over the scored tail, not the whole window: a gesture that resolved during
    a recurrent net's burn-in traverses its range where nothing is scored, and counting
    that would report a healthy sampler on the rows that prove it is not. The tail is
    also what gesture speed is drawn against, so a row's travel here is the number the
    sampler was asked for, less whatever the window's edges cut off.

    Continuous controls only: a switch either steps or does not, so including it would
    report a full traversal for a row whose knobs never budged.
    """
    columns = [i for i, spec in enumerate(param_specs) if spec.type != "switch"]
    ranges = _torch.tensor([param_specs[i].max - param_specs[i].min for i in columns])
    scored = trajectories[:, -window.ny :, columns]
    return ((scored.amax(dim=1) - scored.amin(dim=1)) / ranges).amax(dim=1)


@_pytest.mark.parametrize("training", [False, True])
@_pytest.mark.parametrize("net", _concat_nets())
def test_a_constant_trajectory_matches_the_held_setting(net, training):
    """
    A control that never moves is a trajectory like any other. If the two paths disagree
    the trajectory plumbing is laying the controls out wrong -- transposed, misaligned in
    time, or broadcast where it should be strict -- and the moving anchor would then be
    scoring a window the net never really saw.

    Run in both modes: training is where a recurrent net takes its truncated-BPTT block
    loop, which is the path that could misalign a trajectory across block boundaries.
    """
    net.train(training)
    params = _sample_raw_params(net.param_specs, 3)
    window = _anchor_window(net, 16)
    held = params[:, None, :].expand(-1, window.length, -1)

    assert _torch.allclose(
        _anchor_output(net, held, ny=16), _anchor_output(net, params, ny=16), atol=1e-6
    )


def test_hyperwavenet_refuses_a_trajectory():
    """This net generates its weights from the control vector as a whole, so a value that
    moves within one forward has no meaning to give it."""
    net = _HyperWaveNet.init_from_config(_hyperwavenet_config())
    assert not net.supports_param_trajectory
    length = net.receptive_field - 1 + 8

    with _pytest.raises(NotImplementedError):
        net(_torch.zeros((2, length)), _torch.zeros((2, length, net.param_dim)))


def test_silence_anchor_ramp_refused_at_construction_on_such_a_net():
    """Fail while the config is being read, not on the first training step, since the
    setting cannot be honoured at all."""
    with _pytest.raises(ValueError):
        _ParametricLightningModule(
            _HyperWaveNet.init_from_config(_hyperwavenet_config()),
            loss_config=_ParametricLossConfig(
                silence_anchor_ramp=_SilenceAnchorRampConfig(weight=1.0)
            ),
        )


def test_sampled_trajectories_are_valid_control_values():
    """Every frame is encoded on its way into the net, so an out-of-range value or a
    fractional switch index would fail there rather than here."""
    net = _net()
    window = _anchor_window(net, 32)
    trajectories = _sample_param_trajectories(
        net.param_specs, 64, window, **_DEFAULT_TRAVEL
    )

    assert trajectories.shape == (64, window.length, net.param_dim)
    for i, spec in enumerate(net.param_specs):
        values = trajectories[..., i]
        assert (values >= spec.min).all() and (values <= spec.max).all()
        if spec.type == "switch":
            # Switches step between indices rather than blending: a fractional index is
            # a conditioning vector no model was ever trained on.
            assert (values == values.round()).all()
    net._encode_params(trajectories)


def test_every_sampled_trajectory_moves_a_control_unless_it_is_a_hold():
    """A row that does not move must be one that was chosen not to, not an accident.

    ``hold_probability`` rows are deliberately still -- they are what pins the
    DC-vs-control map. Any other still row is the term paying for a trajectory forward
    and scoring a setting the static anchor already covers.
    """
    net = _net()
    window = _anchor_window(net, 4096)
    trajectories = _sample_param_trajectories(
        net.param_specs,
        4096,
        window,
        min_window_travel=8.0,
        max_window_travel=8.0,
        hold_probability=0.25,
        generator=_torch.Generator().manual_seed(0),
    )

    def still_fraction(rows):
        return (rows[:, 0, :] == rows[:, -1, :]).all(dim=1).float().mean().item()

    assert still_fraction(trajectories) == _pytest.approx(0.25, abs=0.03)

    no_holds = _sample_param_trajectories(
        net.param_specs,
        4096,
        window,
        min_window_travel=8.0,
        max_window_travel=8.0,
        hold_probability=0.0,
        generator=_torch.Generator().manual_seed(0),
    )
    # Not exactly zero: a gesture may commit past the last sample of the window, which
    # is a still row arrived at honestly. It has to stay rare enough that the hold rate
    # is what sets how much of the batch holds still.
    assert still_fraction(no_holds) < 0.01


def test_gesture_speed_is_the_same_for_every_architecture_and_window():
    """
    One config has to mean one gesture, whatever net it is training and however wide its
    window. Speed is drawn against the scored tail, which is ``ny`` samples for every
    architecture; drawing it against the whole influencing span instead would hand a
    convolutional net ``receptive_field - 1 + ny`` where a recurrent net gets ``ny``, and
    the same numbers would train the two on gestures an order of magnitude apart.

    Without a common yardstick the term cannot be compared between architectures at all:
    a run that reduced knob noise would be reporting how fast its net's gestures happened
    to be as much as how well it learned them.
    """
    medians = {}
    for name, factory in (("wavenet", _net), ("recurrent", _burn_in_lstm)):
        net = factory()
        net.train()
        for ny in (512, 8192):
            window = _anchor_window(net, ny)
            trajectories = _sample_param_trajectories(
                net.param_specs,
                2048,
                window,
                hold_probability=0.0,
                generator=_torch.Generator().manual_seed(0),
                **_DEFAULT_TRAVEL,
            )
            moved = _travel_fraction(trajectories, net.param_specs, window)
            # Rows whose continuous controls were all held still are the `held` mask and
            # the switch-only gestures doing their job, not slow gestures. Speed is only
            # a question where something is actually moving.
            medians[(name, ny)] = moved[moved > 0.0].median().item()

    reference = medians[("wavenet", 512)]
    for key, median in medians.items():
        assert median == _pytest.approx(reference, rel=0.1), key

    # Floors, so "the same everywhere" cannot be satisfied by moving nothing. Both are
    # derived from the configured bounds rather than fixed, so they track the defaults.
    # The median lands below the drawn geometric mean because a commit instant near the
    # window's edge leaves part of its gesture outside the tail.
    geometric_mean = _math.sqrt(
        _DEFAULT_TRAVEL["min_window_travel"] * _DEFAULT_TRAVEL["max_window_travel"]
    )
    assert geometric_mean / 10.0 < reference < geometric_mean


def test_silence_anchor_ramp_in_training_loss_dict_and_epoch_metrics():
    from nam.train.parametric import _BUCKET_MEAN_METRICS, _bucket_means

    assert "Silence Anchor Ramp" in _BUCKET_MEAN_METRICS
    module = _ParametricLightningModule(
        _net(),
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
    assert "Silence Anchor Ramp" in _bucket_means(loss_dict)


@_pytest.mark.parametrize("training", [False, True])
def test_silence_anchor_ramp_skipped_when_it_should_be(training):
    """Validation and zero weight, the two ways the term must cost nothing."""
    module = _ParametricLightningModule(
        _net(),
        loss_config=_ParametricLossConfig(
            silence_anchor_ramp=_SilenceAnchorRampConfig(
                weight=0.0 if training else 1.0, batch_size=2, ny=16
            )
        ),
    )
    module.train(training)

    loss_dict = module._get_loss_dict(_torch.randn((2, 64)), _torch.randn((2, 64)))

    assert "Silence Anchor Ramp" not in loss_dict


def test_silence_anchor_ramp_config_parsing():
    parsed = _ParametricLossConfig.parse_config(
        {
            "silence_anchor_ramp": {
                "weight": 1.5,
                "ny": 64,
                "min_window_travel": 0.01,
                "max_window_travel": 0.5,
                "hold_probability": 0.5,
                "rail_probability": 0.5,
            }
        }
    )
    ramp = parsed["silence_anchor_ramp"]
    assert ramp.weight == _pytest.approx(1.5)
    assert ramp.ny == 64
    assert ramp.min_window_travel == _pytest.approx(0.01)
    assert ramp.max_window_travel == _pytest.approx(0.5)
    assert ramp.hold_probability == _pytest.approx(0.5)
    assert ramp.rail_probability == _pytest.approx(0.5)
    # Absent by default, so an existing project trains exactly as it did before.
    assert _ParametricLossConfig.parse_config({})["silence_anchor_ramp"] is None


@_pytest.mark.parametrize(
    "config",
    [
        {"weight": 1.0, "min_window_travel": 0.0},
        {"weight": 1.0, "min_window_travel": 1.0, "max_window_travel": 0.5},
        {"weight": 1.0, "hold_probability": 1.5},
        {"weight": 1.0, "rail_probability": 1.5},
        {"weight": 1.0, "typo": 3},
    ],
)
def test_silence_anchor_ramp_config_rejects_invalid(config):
    with _pytest.raises(ValueError):
        _SilenceAnchorRampConfig.from_config(config)


@_pytest.mark.parametrize("net", _concat_nets())
def test_stacking_anchor_batches_does_not_change_a_row(net):
    """
    Sharing a forward is only sound if a row's output does not depend on its neighbours.
    That holds for both concat architectures -- neither convolution nor recurrence mixes
    across the batch -- and it is what lets the two anchors be scored from one pass.
    """
    net.eval()
    window = _anchor_window(net, 16)
    held = _as_held_trajectory(_sample_raw_params(net.param_specs, 3), window.length)
    moving = _sample_param_trajectories(
        net.param_specs, 2, window, **_DEFAULT_TRAVEL
    )

    stacked = _anchor_output(net, _torch.cat([held, moving], dim=0), window.ny)
    held_out = _anchor_output(net, held, window.ny)
    moving_out = _anchor_output(net, moving, window.ny)

    # Recurrence's matmuls aren't bit-exact across batch sizes on every torch build --
    # exercised across the Python-version matrix, so an ULP-level allclose is the
    # meaningful bar here. Convolution stays bit-exact, so hold it to that.
    if isinstance(net, _ConcatLSTM):
        assert _torch.allclose(stacked[:3], held_out, atol=1e-6)
        assert _torch.allclose(stacked[3:], moving_out, atol=1e-6)
    else:
        assert _torch.equal(stacked[:3], held_out)
        assert _torch.equal(stacked[3:], moving_out)


def _both_anchors(ramp_ny: int = 16) -> _ParametricLossConfig:
    return _ParametricLossConfig(
        silence_anchor=_SilenceAnchorConfig(weight=1.0, batch_size=3, ny=16),
        silence_anchor_ramp=_SilenceAnchorRampConfig(
            weight=1.0, batch_size=2, ny=ramp_ny
        ),
    )


@_pytest.mark.parametrize(
    "ramp_ny, forwards", [(16, 1), (32, 2)], ids=["merged", "windows differ"]
)
def test_both_anchors_share_one_forward_when_their_windows_agree(
    mocker, ramp_ny, forwards
):
    """The point of the merge is the launch it saves, so the forward count is the thing
    to assert. Anchors whose windows differ cannot stack and fall back to one each."""
    net = _net()
    module = _ParametricLightningModule(net, loss_config=_both_anchors(ramp_ny))
    module.train()
    spy = mocker.spy(net, "forward")

    loss_dict = module._get_loss_dict(_torch.randn((2, 64)), _torch.randn((2, 64)))

    assert spy.call_count == forwards
    for name in ("Silence Anchor", "Silence Anchor Ramp"):
        assert loss_dict[name].value.ndim == 0


def test_a_zero_weight_anchor_costs_no_forward(mocker):
    """Weight zero is how a config disables one of two terms; the other must still run,
    and must not be dragged into a stacked forward with it."""
    net = _net()
    loss_config = _both_anchors()
    loss_config.silence_anchor.weight = 0.0
    module = _ParametricLightningModule(net, loss_config=loss_config)
    module.train()
    spy = mocker.spy(net, "forward")

    loss_dict = module._get_loss_dict(_torch.randn((2, 64)), _torch.randn((2, 64)))

    assert spy.call_count == 1
    assert "Silence Anchor" not in loss_dict
    assert spy.call_args.args[0].shape[0] == 2  # The ramp's rows only.
