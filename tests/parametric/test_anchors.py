import pytest as _pytest
import torch as _torch

from nam.models.parametric import ConcatLSTM as _ConcatLSTM
from nam.models.parametric import ConcatWaveNet as _ConcatWaveNet
from nam.models.parametric import HyperWaveNet as _HyperWaveNet
from nam.models.parametric._anchors import anchor_output as _anchor_output
from nam.models.parametric._anchors import sample_raw_params as _sample_raw_params
from nam.train.parametric import _ParametricLightningModule as _ParametricLightningModule
from nam.train.parametric import _ParametricLossConfig as _ParametricLossConfig
from nam.train.parametric import _SilenceAnchorConfig as _SilenceAnchorConfig
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
