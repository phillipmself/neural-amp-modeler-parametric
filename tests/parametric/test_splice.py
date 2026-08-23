import pytest as _pytest
import torch as _torch

from nam.data import Dataset as _Dataset
from nam.models.parametric import ConcatLSTM as _ConcatLSTM
from nam.models.parametric import ConcatWaveNet as _ConcatWaveNet
from nam.models.parametric import HyperWaveNet as _HyperWaveNet
from nam.models.parametric import ParametricDataset as _ParametricDataset
from nam.models.parametric._splice import SpliceSource as _SpliceSource
from nam.models.parametric._splice import splice_window as _splice_window
from nam.train.parametric import _ParametricLightningModule as _ParametricLightningModule
from nam.train.parametric import _ParametricLossConfig as _ParametricLossConfig
from nam.train.parametric import _ParamStepSpliceConfig as _ParamStepSpliceConfig
from tests.parametric.test_concat_lstm import _concat_lstm_config
from tests.parametric.test_concat_wavenet import _concat_wavenet_config
from tests.parametric.test_hyperwavenet import _hyperwavenet_config

_LENGTH = 20_000
# Fixed noise: no two windows of it are alike, so a window or a target taken from the
# wrong moment shows up in the values rather than only in a loss that goes the wrong way.
_SOURCE = 0.4 * (
    2.0 * _torch.rand((_LENGTH,), generator=_torch.Generator().manual_seed(0)) - 1.0
)
# A fixture this short has a mean that is only statistically near zero, so scaled copies
# of it differ in DC by more than a real capture pair would. Only the DC test uses the
# shipping threshold.
_DC_REPORT_DB = -20.0


def _net(kind: str = "wavenet"):
    if kind == "wavenet":
        return _ConcatWaveNet.init_from_config(_concat_wavenet_config())
    return _ConcatLSTM.init_from_config(_concat_lstm_config())


def _delayed(y: _torch.Tensor, delay: int) -> _torch.Tensor:
    """The output file a capture with this declared ``delay`` would have produced.

    A positive delay means y lags x, which is what the data set's correction shifts back
    out; a fixture that declares one without shifting is simply misaligned.
    """
    if delay > 0:
        return _torch.cat([_torch.zeros(delay), y[:-delay]])
    if delay < 0:
        return _torch.cat([y[-delay:], _torch.zeros(-delay)])
    return y


def _capture(
    net,
    params,
    *,
    start=None,
    delay=None,
    gain=0.0,
    offset=0.0,
    scale=1.0,
) -> _ParametricDataset:
    """One capture of ``_SOURCE`` at ``scale``, aligned to the shared input.

    Captures are told apart by their output gain rather than by a DC offset so the fixture
    does not trip the DC report that has a test of its own.
    """
    return _ParametricDataset(
        _Dataset(
            _SOURCE.clone(),
            _delayed(_SOURCE * scale, delay or 0) + offset,
            nx=net.receptive_field,
            ny=None,
            start_samples=start,
            delay=delay,
            input_gain=gain,
            require_input_pre_silence=None,
        ),
        _torch.tensor(params, dtype=_torch.float32),
    )


def _assert_segments_match(source_scales, cases, window, target):
    """Every scored segment is its capture's answer to the source samples the window's
    output covers."""
    bounds = list(window.segment_starts) + [window.ny]
    scored_start = window.length - window.ny
    for row in range(target.shape[0]):
        first = int(cases.start[row]) + scored_start
        for i in range(window.steps):
            scale = source_scales[int(cases.captures[row, i + 1])]
            expected = _SOURCE[first + bounds[i] : first + bounds[i + 1]] * scale
            assert _torch.allclose(target[row, bounds[i] : bounds[i + 1]], expected, atol=1e-6)


def _source(captures, net) -> _SpliceSource:
    return _SpliceSource.from_datasets(captures, net, dc_report_db=_DC_REPORT_DB)


def _window(net, **overrides):
    kwargs = {"steps": 1, "spacing": 128, "min_tail": 256, "settle": 256}
    kwargs.update(overrides)
    return _splice_window(net, **kwargs)


def _splice_config(**overrides) -> _ParamStepSpliceConfig:
    kwargs = {
        "weight": 1.0,
        "batch_size": 2,
        "spacing": 128,
        "min_tail": 256,
        "settle": 256,
        "validation_windows": 0,
        "dc_report_db": _DC_REPORT_DB,
    }
    kwargs.update(overrides)
    return _ParamStepSpliceConfig(**kwargs)


@_pytest.mark.parametrize("start", [None, 0, 300, -900])
@_pytest.mark.parametrize("delay", [None, 0, 7, -11])
def test_dataset_input_start_locates_the_source_sample(start, delay):
    """Every splice target is fetched through this offset; if it is wrong the model is
    trained against audio from the wrong moment and the metrics still look sane."""
    dataset = _Dataset(
        _SOURCE.clone(),
        _SOURCE.clone(),
        nx=1,
        ny=None,
        start_samples=start,
        delay=delay,
        require_input_pre_silence=None,
    )

    assert _torch.equal(
        dataset.x, _SOURCE[dataset.input_start : dataset.input_start + len(dataset.x)]
    )


@_pytest.mark.parametrize("kind", ["wavenet", "lstm"])
def test_target_comes_from_the_source_times_the_window_covers(kind):
    """Captures may be trimmed and delay-corrected differently, so a local index says
    nothing on its own about what moment it is."""
    net = _net(kind)
    scales = [1.0, 0.4]
    captures = [
        _capture(net, [2.0, 0.0], start=300, delay=5, scale=scales[0]),
        _capture(net, [9.0, 2.0], start=-9_000, delay=-11, scale=scales[1]),
    ]
    source = _source(captures, net)
    window = _window(net)
    cases = source.enumerate_cases(4, window)

    x, _, target = source.build(cases, window)

    for row in range(x.shape[0]):
        first = int(cases.start[row])
        assert _torch.allclose(x[row], _SOURCE[first : first + window.length], atol=1e-7)
    _assert_segments_match(scales, cases, window, target)


@_pytest.mark.parametrize("kind", ["wavenet", "lstm"])
@_pytest.mark.parametrize("steps", [1, 2])
def test_the_controls_step_where_the_target_does(kind, steps):
    """The trajectory is indexed in the input domain and the target in the output domain,
    which are ``receptive_field - 1 + warmup`` apart. Getting that wrong teaches the model
    to anticipate a knob move, and trains perfectly happily while doing it."""
    net = _net(kind)
    scales = [1.0, 0.4]
    captures = [
        _capture(net, [2.0, 0.0], scale=scales[0]),
        _capture(net, [9.0, 2.0], scale=scales[1]),
    ]
    source = _source(captures, net)
    window = _window(net, steps=steps)
    cases = source.enumerate_cases(3, window)

    _, params, target = source.build(cases, window)

    assert window.length - window.ny == net.receptive_field - 1 + window.warmup
    for row in range(params.shape[0]):
        control_steps = (
            (params[row, 1:] != params[row, :-1]).any(dim=-1).nonzero()[:, 0] + 1
        )
        assert control_steps.tolist() == list(window.step_offsets)
    _assert_segments_match(scales, cases, window, target)


def test_captures_driven_at_different_levels_are_refused():
    """Two captures at different input gains answered different signals, so neither is a
    target for the other's window."""
    net = _net()
    captures = [
        _capture(net, [2.0, 0.0]),
        _capture(net, [9.0, 2.0], gain=-6.0),
    ]

    with _pytest.raises(ValueError, match="does not share an input"):
        _SpliceSource.from_datasets(captures, net)


def test_a_dc_difference_between_captures_is_reported():
    """It would be spliced in as an instantaneous DC step at every control change, which
    no device makes."""
    net = _net()
    captures = [
        _capture(net, [2.0, 0.0]),
        _capture(net, [9.0, 2.0], offset=0.2),
    ]

    with _pytest.warns(UserWarning, match="DC offset"):
        _SpliceSource.from_datasets(captures, net)


def test_one_capture_cannot_be_stepped_between():
    net = _net()

    with _pytest.raises(ValueError, match="at least two captures"):
        _SpliceSource.from_datasets([_capture(net, [2.0, 0.0])], net)


def test_absent_by_default():
    assert _ParametricLossConfig.parse_config({})["param_step_splice"] is None


def test_config_parsing_rejects_unknown_fields():
    with _pytest.raises(ValueError):
        _ParametricLossConfig.parse_config(
            {"param_step_splice": {"weight": 1.0, "nope": 1}}
        )


def test_refused_at_construction_on_a_net_that_cannot_take_a_trajectory():
    """A net that conditions on the control vector as a whole has no meaning to give a
    control that moves within one forward, so fail while the config is being read."""
    with _pytest.raises(ValueError, match="param trajectory"):
        _ParametricLightningModule(
            _HyperWaveNet.init_from_config(_hyperwavenet_config()),
            loss_config=_ParametricLossConfig(param_step_splice=_splice_config()),
        )


def _module(net, **overrides):
    module = _ParametricLightningModule(
        net, loss_config=_ParametricLossConfig(param_step_splice=_splice_config(**overrides))
    )
    module.set_splice_sources(
        train=[_capture(net, [2.0, 0.0]), _capture(net, [9.0, 2.0], scale=0.5)]
    )
    return module


def test_in_training_loss_dict():
    net = _net()
    module = _module(net)
    module.train()

    item = module._get_loss_dict(_torch.randn((2, 64)), _torch.randn((2, 64)))["Param Step"]

    assert item.weight == _pytest.approx(1.0)
    assert item.value.ndim == 0 and item.value >= 0.0


def test_skipped_during_validation():
    """A regularizer on the model, not a measurement of held-out data."""
    module = _module(_net())
    module.eval()

    loss_dict = module._get_loss_dict(_torch.randn((2, 64)), _torch.randn((2, 64)))

    assert "Param Step" not in loss_dict


def test_zero_weight_costs_no_forward(mocker):
    """Weight zero is how the validation metric gets read off a model without changing
    what it is trained on."""
    net = _net()
    module = _module(net, weight=0.0)
    module.train()
    spy = mocker.spy(net, "forward")

    loss_dict = module._get_loss_dict(_torch.randn((2, 64)), _torch.randn((2, 64)))

    assert spy.call_count == 0
    assert "Param Step" not in loss_dict


def test_setup_is_a_no_op_without_the_term(mocker):
    """The shared-input, DC and alignment scans run over whole captures; a run that never
    asked for splices must not pay for them."""
    module = _ParametricLightningModule(_net(), loss_config=_ParametricLossConfig())
    spy = mocker.spy(_SpliceSource, "from_datasets")

    module.set_splice_sources(train=[], validation=[])

    assert spy.call_count == 0


def test_an_enabled_term_without_captures_fails_rather_than_doing_nothing():
    """Silently training without a term that was asked for is worse than not starting:
    the run looks normal and the model comes out unchanged."""
    module = _ParametricLightningModule(
        _net(), loss_config=_ParametricLossConfig(param_step_splice=_splice_config())
    )
    module.train()

    with _pytest.raises(RuntimeError, match="set_splice_sources"):
        module._get_loss_dict(_torch.randn((2, 64)), _torch.randn((2, 64)))


def test_a_validation_split_with_nothing_to_step_between_still_trains():
    """The metric is a reading, not the thing that was asked for."""
    net = _net()
    module = _ParametricLightningModule(
        net,
        loss_config=_ParametricLossConfig(
            param_step_splice=_splice_config(validation_windows=4)
        ),
    )

    with _pytest.warns(UserWarning, match="no validation metric"):
        module.set_splice_sources(
            train=[_capture(net, [2.0, 0.0]), _capture(net, [9.0, 2.0], scale=0.5)],
            validation=[_capture(net, [5.0, 1.0], scale=0.7)],
        )

    assert module._param_step_splice_metrics() == {}
    module.train()
    assert "Param Step" in module._get_loss_dict(
        _torch.randn((2, 64)), _torch.randn((2, 64))
    )


@_pytest.mark.parametrize("kind", ["wavenet", "lstm"])
def test_validation_metric_reports_a_penalty_against_the_held_control(kind):
    net = _net(kind)
    module = _ParametricLightningModule(
        net,
        loss_config=_ParametricLossConfig(
            param_step_splice=_splice_config(weight=0.0, validation_windows=4)
        ),
    )
    module.set_splice_sources(
        validation=[_capture(net, [2.0, 0.0]), _capture(net, [9.0, 2.0], scale=0.5)]
    )

    metrics = module._param_step_splice_metrics()

    assert metrics["Param Step Penalty"] == _pytest.approx(
        metrics["Param Step ESR"] / metrics["Param Step Static ESR"]
    )


def test_enumerated_cases_do_not_move_between_calls():
    """A metric compared epoch to epoch cannot be drawn fresh each time or the draw is
    most of what moves."""
    net = _net()
    source = _source([_capture(net, [2.0, 0.0]), _capture(net, [9.0, 2.0], scale=0.5)], net)
    window = _window(net)

    first = source.enumerate_cases(8, window)
    second = source.enumerate_cases(8, window)

    assert _torch.equal(first.start, second.start)
    assert _torch.equal(first.captures, second.captures)


def test_sampled_windows_stay_inside_the_shared_region_and_always_move():
    net = _net()
    source = _source(
        [
            _capture(net, [2.0, 0.0], start=300),
            _capture(net, [9.0, 2.0], start=-9_000, scale=0.5),
            _capture(net, [5.0, 1.0], delay=6, scale=0.25),
        ],
        net,
    )
    window = _window(net, steps=2)

    cases = source.sample_cases(64, window, neighbors=2, neighbor_probability=0.5)

    low, high = source.shared_range
    assert (cases.start >= low).all()
    assert (cases.start + window.length <= high).all()
    # A step onto the setting already in force pays for a trajectory and scores a held
    # setting.
    assert (cases.captures[:, 1:] != cases.captures[:, :-1]).all()
