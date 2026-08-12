import pytest as _pytest
import torch as _torch

from nam.models.parametric import ConcatWaveNet as _ConcatWaveNet
from nam.models.parametric import HyperWaveNet as _HyperWaveNet
from nam.models.parametric._anchors import anchor_output as _anchor_output
from nam.models.parametric._anchors import sample_raw_params as _sample_raw_params
from nam.train.parametric import _ParametricLightningModule as _ParametricLightningModule
from nam.train.parametric import _ParametricLossConfig as _ParametricLossConfig
from nam.train.parametric import _SilenceAnchorConfig as _SilenceAnchorConfig
from tests.parametric.test_concat_wavenet import _concat_wavenet_config
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
