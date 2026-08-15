import json as _json
from typing import cast as _cast

import pytest as _pytest
import torch as _torch

from nam.models import factory as _factory
from nam.models.parametric import FiLMWaveNet as _FiLMWaveNet
from nam.models.parametric import ParamSpec as _ParamSpec
from nam.models.parametric import export_parametric as _export_parametric
from nam.models.parametric._anchors import (
    sample_param_trajectories as _sample_param_trajectories,
)
from nam.models.parametric._film_wavenet import _film_condition_size
from nam.models.parametric._film_wavenet import _film_input_dim
from nam.models.wavenet._film import FiLM as _FiLM
from nam.models.wavenet._wavenet import WaveNet as _InnerWaveNet


def _film_wavenet_config(*, encoder: bool = False) -> dict:
    config = {
        "sample_rate": 48_000.0,
        "layers": [
            {
                "head": {"out_channels": 2, "kernel_size": 1, "bias": True},
                "channels": 4,
                "kernel_size": 2,
                "dilations": [1, 2],
                "activation": "Tanh",
                "activation_post_film": {"active": True, "shift": True, "groups": 1},
                "conv_post_film": {"active": True, "shift": False, "groups": 1},
            },
            {
                "input_size": 4,  # = previous layer array's channels
                "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
                "channels": 2,
                "kernel_size": 2,
                "dilations": [1],
                "activation": "Tanh",
                "input_mixin_pre_film": {"active": True, "shift": True, "groups": 1},
            },
        ],
        "head_scale": 0.5,
        "params": [
            {"name": "gain", "min": 0.0, "max": 10.0, "default": 5.0, "type": "continuous"},
            {"name": "mode", "min": 0, "max": 2, "default": 1, "type": "switch",
             "enum_names": ["clean", "crunch", "lead"]},
        ],
    }
    if encoder:
        config["param_encoder"] = {
            "hidden_sizes": [6],
            "out_features": 3,
            "activation": "ReLU",
        }
    return config


def _param_specs() -> tuple[_ParamSpec, ...]:
    return tuple(
        _ParamSpec.from_dict(spec) for spec in _film_wavenet_config()["params"]
    )


def _init(config: dict) -> _FiLMWaveNet:
    return _cast(_FiLMWaveNet, _factory.init("FiLMWaveNet", args=(config,)))


def _perturbed(config: dict, seed: int = 0) -> _FiLMWaveNet:
    """A model whose FiLM modules are no longer the identity, so controls do something."""
    model = _init(config)
    generator = _torch.Generator().manual_seed(seed)
    with _torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(
                0.2 * _torch.randn(parameter.shape, generator=generator)
            )
    model.eval()
    return model


def _stock_inner_config() -> dict:
    return {
        "layers_configs": [
            {
                "input_size": 1,
                "condition_size": 1,
                "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
                "channels": 2,
                "kernel_size": 2,
                "dilations": [1],
                "activation": "Tanh",
            }
        ],
        "head_scale": 1.0,
    }


def _inner_wavenet_internal_config(*, slimmable: bool = False) -> dict:
    layer = {
        "input_size": 1,
        "condition_size": 1,
        "film_condition_size": 4,
        "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
        "channels": 4,
        "kernel_size": 2,
        "dilations": [1, 2],
        "activation": "Tanh",
        "film_params": {
            "activation_post_film": {"active": True, "shift": True, "groups": 1}
        },
    }
    if slimmable:
        layer["slimmable"] = {"method": "slice_channels_uniform", "kwargs": {}}
    return {"layers_configs": [layer], "head_scale": 1.0}


def test_inner_wavenet_is_shaped_like_a_stock_one():
    # The controls reach the layers only through FiLM, so the audio path must stay
    # single-channel -- that is what makes the inner net interchangeable with a stock
    # capture and keeps the controls out of every time-convolved operator.
    model = _init(_film_wavenet_config())

    exported = model._export_config()
    for layer in exported["layers"]:
        assert layer["condition_size"] == 1
    assert exported["layers"][0]["input_size"] == 1


def test_export_declares_film_condition_size_on_every_layer_array():
    # The runtime sizes its cached FiLM condition from this field.
    model = _init(_film_wavenet_config(encoder=True))

    exported = model._export_config()
    assert model.film_condition_size == 3
    for layer in exported["layers"]:
        assert layer["film_condition_size"] == 3


def test_film_condition_size_follows_the_params_without_an_encoder():
    model = _init(_film_wavenet_config())

    # 1 continuous + a 3-way switch
    assert model.film_condition_size == 4
    for layer in model._export_config()["layers"]:
        assert layer["film_condition_size"] == 4


@_pytest.mark.parametrize(
    "key,value", [("input_size", 2), ("condition_size", 5), ("film_condition_size", 9)]
)
def test_wrong_derived_layer_field_is_rejected(key, value):
    config = _film_wavenet_config()
    config["layers"][0][key] = value
    with _pytest.raises(ValueError, match=key):
        _init(config)


def test_config_without_any_active_film_is_rejected():
    config = _film_wavenet_config()
    for layer in config["layers"]:
        for film_key in [k for k in layer if k.endswith("_film")]:
            del layer[film_key]
    with _pytest.raises(ValueError, match="FiLM"):
        _init(config)


def test_condition_dsp_config_is_rejected():
    config = _film_wavenet_config()
    config["condition_dsp"] = {"name": "WaveNet", "config": {}}
    with _pytest.raises(NotImplementedError, match="condition_dsp"):
        _init(config)


def test_slimmable_layer_array_is_rejected():
    config = _film_wavenet_config()
    config["layers"][0]["slimmable"] = {
        "method": "slice_channels_uniform",
        "kwargs": {},
    }
    with _pytest.raises(NotImplementedError, match="slimmable"):
        _init(config)


def test_condition_dsp_inner_wavenet_is_rejected():
    tiny = _inner_wavenet_internal_config()
    inner = _InnerWaveNet.init_from_config(
        {**tiny, "condition_dsp": {"name": "WaveNet", "config": _stock_inner_config()}}
    )
    with _pytest.raises(NotImplementedError, match="condition_dsp"):
        _FiLMWaveNet(wavenet=inner, param_specs=_param_specs(), sample_rate=48_000.0)


def test_packed_layer_array_is_rejected():
    config = _film_wavenet_config()
    config["layers"][0]["packing"] = {"num_models": 2}
    with _pytest.raises(NotImplementedError, match="pack"):
        _init(config)


def test_identity_init_makes_the_model_a_plain_wavenet():
    # Training starts from the unmodulated network, so an existing single-setting capture
    # can be loaded straight into it.
    model = _init(_film_wavenet_config())
    model.eval()
    films = [m for m in model._wavenet.modules() if isinstance(m, _FiLM)]
    # FiLM is per layer, not per layer array: 2 sites x 2 layers, then 1 site x 1 layer.
    assert len(films) == 5

    for film in films:
        t = _torch.randn(2, _film_input_dim(film), 7)
        c = _torch.randn(2, _film_condition_size(film), 7)
        assert _torch.allclose(film(t, c), t, atol=1e-6)

    x = _torch.randn(model.receptive_field + 32)
    low = model(x, _torch.tensor([0.0, 0.0]), pad_start=False)
    high = model(x, _torch.tensor([10.0, 2.0]), pad_start=False)
    assert _torch.allclose(low, high, atol=1e-6)


def test_identity_init_can_be_turned_off():
    config = _film_wavenet_config()
    config["identity_init"] = False
    model = _init(config)

    films = [m for m in model._wavenet.modules() if isinstance(m, _FiLM)]
    assert any(film._film.weight.abs().max() > 0.0 for film in films)


@_pytest.mark.parametrize("encoder", [False, True])
def test_params_condition_the_output(encoder):
    model = _perturbed(_film_wavenet_config(encoder=encoder))
    x = _torch.randn(model.receptive_field + 32)

    low = model(x, _torch.tensor([0.0, 0.0]), pad_start=False)
    high = model(x, _torch.tensor([10.0, 2.0]), pad_start=False)

    assert not _torch.allclose(low, high)


def test_forward_shape_contract():
    model = _init(_film_wavenet_config())
    rf = model.receptive_field
    length = rf + 10
    out_length = length - rf + 1
    batch_params = _torch.tensor([[5.0, 1.0], [2.0, 0.0]])
    shared_params = _torch.tensor([5.0, 1.0])

    x_batched = _torch.randn(2, length)
    x_flat = _torch.randn(length)

    assert model(x_batched, batch_params, pad_start=False).shape == (2, out_length)
    assert model(x_batched, shared_params, pad_start=False).shape == (2, out_length)
    assert model(x_flat, batch_params, pad_start=False).shape == (2, out_length)
    assert model(x_flat, shared_params, pad_start=False).shape == (out_length,)


@_pytest.mark.parametrize("encoder", [False, True])
def test_weight_round_trip_and_reload_from_export(encoder):
    config = _film_wavenet_config(encoder=encoder)
    model = _perturbed(config)
    x = _torch.randn(2, model.receptive_field + 24)
    params = _torch.tensor([[1.0, 0.0], [9.0, 2.0]])
    expected = model(x, params, pad_start=False)

    weights = model._export_weights()
    exported = model._export_config()
    exported["sample_rate"] = 48_000.0

    reloaded = _init(_json.loads(_json.dumps(exported)))
    consumed = reloaded.import_weights(_torch.tensor(weights))
    reloaded.eval()

    assert consumed == len(weights)
    assert _torch.allclose(reloaded(x, params, pad_start=False), expected, atol=1e-5)


def test_encoder_weights_come_before_the_wavenet_weights():
    # The runtime reads the encoder first so it can size the rest from the config.
    config = _film_wavenet_config(encoder=True)
    model = _perturbed(config)

    encoder_weights = model.param_encoder.export_weights().detach()
    weights = model._export_weights()

    assert _torch.allclose(
        _torch.tensor(weights[: len(encoder_weights)]), encoder_weights, atol=1e-6
    )
    assert len(weights) == len(encoder_weights) + len(model._wavenet.export_weights())


@_pytest.mark.parametrize("encoder", [False, True])
def test_a_control_move_settles_within_exactly_the_receptive_field(encoder):
    # The point of the architecture: the controls are read by 1x1 convolutions, so they
    # have no time extent of their own. Outside one receptive field of a move the model is
    # exactly the model for the current setting; inside it, the only thing resolving is
    # audio state left over from the old setting.
    model = _perturbed(_film_wavenet_config(encoder=encoder))
    rf = model.receptive_field
    length = 8 * rf
    step = 4 * rf
    x = _torch.randn(length) * 0.3

    def held(value: float) -> _torch.Tensor:
        return _torch.tensor([[value], [1.0]]).expand(-1, length).clone()

    moving = held(1.0)
    moving[0, step:] = 9.0

    with _torch.no_grad():
        low = model.forward_control_sequence(x, held(1.0), pad_start=False)
        high = model.forward_control_sequence(x, held(9.0), pad_start=False)
        move = model.forward_control_sequence(x, moving, pad_start=False)

    # A held sequence must agree with the plain constant-setting forward.
    assert _torch.allclose(
        low, model(x, _torch.tensor([1.0, 1.0]), pad_start=False), atol=1e-6
    )
    # Before the move, and again one receptive field after it, the moving run *is* the
    # corresponding steady state -- no control memory outside that window.
    assert _torch.allclose(move[: step - rf], low[: step - rf], atol=1e-6)
    assert _torch.allclose(move[step:], high[step:], atol=1e-6)
    assert not _torch.allclose(
        move[step - rf : step], high[step - rf : step], atol=1e-6
    )


def test_control_sequence_rejects_a_length_mismatch():
    model = _init(_film_wavenet_config())
    length = model.receptive_field + 8
    x = _torch.randn(length)
    with _pytest.raises(ValueError, match="length"):
        model.forward_control_sequence(
            x, _torch.ones(2, length - 1), pad_start=False
        )


def test_export_parametric_writes_a_loadable_file(tmp_path):
    model = _perturbed(_film_wavenet_config(encoder=True))
    x = _torch.randn(2, model.receptive_field + 24)
    params = _torch.tensor([[1.0, 0.0], [9.0, 2.0]])
    expected = model(x, params, pad_start=False)

    _export_parametric(model, tmp_path, basename="film")

    with open(tmp_path / "film.nam") as fp:
        exported = _json.load(fp)

    assert exported["architecture"] == "FiLMWaveNet"
    assert exported["version"] == "1.0.0"
    assert exported["config"]["param_encoder"]["out_features"] == 3

    # The file has to be enough on its own to rebuild the model that wrote it.
    reloaded = _init(exported["config"])
    consumed = reloaded.import_weights(_torch.tensor(exported["weights"]))
    reloaded.eval()

    assert consumed == len(exported["weights"])
    assert _torch.allclose(reloaded(x, params, pad_start=False), expected, atol=1e-5)


def test_film_condition_width_is_independent_of_the_knob_count():
    # Every per-layer gain is a linear combination of the FiLM condition, so this width is
    # what bounds how independently the channels can respond -- it is a free design choice,
    # not a function of how many knobs the model has.
    config = _film_wavenet_config(encoder=True)
    config["param_encoder"]["out_features"] = 16
    model = _perturbed(config)

    assert model.film_condition_size == 16
    for film in (m for m in model._wavenet.modules() if isinstance(m, _FiLM)):
        assert _film_condition_size(film) == 16

    x = _torch.randn(model.receptive_field + 32)
    low = model(x, _torch.tensor([0.0, 0.0]), pad_start=False)
    high = model(x, _torch.tensor([10.0, 2.0]), pad_start=False)
    assert not _torch.allclose(low, high)


def test_control_sequence_rejects_an_input_shorter_than_the_receptive_field():
    model = _init(_film_wavenet_config())
    length = model.receptive_field - 1
    with _pytest.raises(ValueError, match="receptive field"):
        model.forward_control_sequence(
            _torch.randn(length), _torch.ones(2, length), pad_start=False
        )


def test_inner_wavenet_rejects_a_film_condition_that_neither_broadcasts_nor_aligns():
    # The layers right-align the condition against tensors of varying length, so a length
    # in between silently misbroadcasts inside a FiLM instead of failing legibly.
    model = _init(_film_wavenet_config())
    length = model.receptive_field + 8
    x = _torch.randn(1, 1, length)
    p = _torch.randn(1, model.film_condition_size, 3)
    with _pytest.raises(ValueError, match="FiLM condition length"):
        model._wavenet(x, p)


def test_moving_control_runs_through_the_compiled_step():
    """A moving control is the anchors' and the augmentation's hot path.

    It used to bypass `_run_step`, which left it eager while every other forward was
    compiled -- expensive on a model whose cost is dominated by kernel launches.
    """
    model = _init(_film_wavenet_config())
    length = model.receptive_field + 8
    x = _torch.randn(2, length)
    trajectory = _sample_param_trajectories(
        model.param_specs, 2, length, 48000.0,
        min_ramp_seconds=0.01, max_ramp_seconds=0.1,
    )

    eager = model(x, trajectory, pad_start=False)
    model.set_compiled(True, backend="eager")

    calls = []
    compiled = model._compiled_step

    def _counting(*args, **kwargs):
        calls.append(1)
        return _cast(object, compiled)(*args, **kwargs)

    model._compiled_step = _counting
    out = model(x, trajectory, pad_start=False)

    assert calls == [1]
    assert _torch.allclose(out, eager, atol=1e-6)
