import torch as _torch

from nam import data as _data
from nam.models.parametric.catnets import CatWaveNet as _CatWaveNet


def test_handshake_uses_param_dim_from_wavenet_layout():
    config = {
        "layers_configs": [
            {
                "input_size": 2,
                "condition_size": 2,
                "channels": 4,
                "head_size": 1,
                "kernel_size": 3,
                "dilations": [1, 2],
                "activation": "Tanh",
            }
        ],
        "head_scale": 0.02,
        "sample_rate": 48_000,
    }
    model = _CatWaveNet.init_from_config(config)

    x = _torch.zeros(64)
    y = _torch.zeros(64)
    dataset = _data.ParametricDataset(
        params={"gain": 1.0},
        x=x,
        y=y,
        nx=3,
        ny=8,
        sample_rate=48_000,
        require_input_pre_silence=None,
    )

    dataset.handshake(model)


def test_export_param_config_includes_min_max_when_provided():
    config = {
        "layers_configs": [
            {
                "input_size": 2,
                "condition_size": 2,
                "channels": 4,
                "head_size": 1,
                "kernel_size": 3,
                "dilations": [1, 2],
                "activation": "Tanh",
            }
        ],
        "head_scale": 0.02,
        "sample_rate": 48_000,
    }
    model = _CatWaveNet.init_from_config(config)

    x = _torch.zeros(64)
    y = _torch.zeros(64)
    dataset = _data.ParametricDataset(
        params={"gain": 1.0},
        x=x,
        y=y,
        nx=3,
        ny=8,
        sample_rate=48_000,
        require_input_pre_silence=None,
        common_params={
            "gain": {
                "type": "continuous",
                "default_value": 2.0,
                "minval": 1.0,
                "maxval": 3.0,
            }
        },
    )

    model.handshake(dataset)

    exported = model._export_parametric_config()
    assert exported == {
        "gain": {
            "type": "continuous",
            "default_value": 2.0,
            "minval": 1.0,
            "maxval": 3.0,
        }
    }
