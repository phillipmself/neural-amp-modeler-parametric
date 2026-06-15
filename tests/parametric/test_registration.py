"""
PA9 — factory registration test.

Importing nam.models.parametric registers "ParametricWaveNet" in the model
factory. This test confirms that factory.init("ParametricWaveNet", config)
returns a ParametricWaveNet instance without any edits to factory.py.
"""

import nam.models.parametric  # noqa: F401 — side-effect: registers "ParametricWaveNet"

from nam.models import factory
from nam.models.parametric import ParametricWaveNet

_CONFIG = {
    "layers_configs": [
        {
            "input_size": 1,
            "condition_size": 1,
            "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
            "channels": 2,
            "kernel_size": 2,
            "dilations": [1, 2],
            "activation": "Tanh",
        }
    ],
    "head_scale": 1.0,
    "param_names": ["gain"],
    "param_dim": 1,
}


def test_pa9_factory_init_returns_parametric_wavenet():
    """PA9: factory.init('ParametricWaveNet', config) returns a ParametricWaveNet."""
    model = factory.init("ParametricWaveNet", kwargs={"config": _CONFIG})
    assert isinstance(model, ParametricWaveNet), (
        f"Expected ParametricWaveNet, got {type(model)}"
    )


def test_pa9_factory_registered_key():
    """PA9: 'ParametricWaveNet' key is present in the model registry after import."""
    # The registry is a module-level dict; peek via a direct init to confirm key exists
    # (factory.init would raise KeyError if not registered)
    model = factory.init("ParametricWaveNet", kwargs={"config": _CONFIG})
    assert model is not None
