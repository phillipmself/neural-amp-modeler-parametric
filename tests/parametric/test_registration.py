"""
PA9 — factory registration test.
PA10 — dataset registry test.

Importing nam.models.parametric registers:
  - "ParametricWaveNet" in the model factory (PA9)
  - "parametric" in the dataset registry (PA10)

Both confirmed here without any edits to factory.py or data.py.
"""

from pathlib import Path as _Path

import numpy as _np
import pytest as _pytest

import nam.models.parametric  # noqa: F401 — side-effect: registers both types

from nam.data import Split, init_dataset, np_to_wav as _np_to_wav
from nam.models import factory
from nam.models.parametric import ParametricDataset, ParametricWaveNet

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
    # nominal_params is required since C1.2 (AD-5)
    "nominal_params": [0.5],
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


# ---------------------------------------------------------------------------
# PA10 — dataset registry: init_dataset returns ParametricDataset
# ---------------------------------------------------------------------------

_RATE = 48_000
_PRE_SILENCE = int(0.5 * _RATE)
_AUDIO = 100


def _write_pair(tmp_path: _Path):
    """Write a tiny mono WAV pair and return (x_path_str, y_path_str)."""
    rng = _np.random.default_rng(99)
    n = _PRE_SILENCE + _AUDIO
    x = _np.concatenate([_np.zeros(_PRE_SILENCE), rng.uniform(-0.4, 0.4, _AUDIO)])
    y = rng.uniform(-0.4, 0.4, n)
    x_path = tmp_path / "x.wav"
    y_path = tmp_path / "y.wav"
    _np_to_wav(x, x_path, rate=_RATE)
    _np_to_wav(y, y_path, rate=_RATE)
    return str(x_path), str(y_path)


def test_pa10_dataset_registry_returns_parametric_dataset(tmp_path):
    """PA10: init_dataset({'type':'parametric', ...}, split) returns ParametricDataset."""
    x_path, y_path = _write_pair(tmp_path)
    # init_dataset dispatches on config["type"] at data.py:977; if "parametric" is
    # registered, it calls ParametricDataset.init_from_config with the split sub-config.
    config = {
        "type": "parametric",
        "train": {
            "x_path": x_path,
            "y_path": y_path,
            "nx": 4,
            "ny": 8,
            "sample_rate": _RATE,
            "param_names": ["gain"],
            "params": [0.5],
        },
    }
    ds = init_dataset(config, Split.TRAIN)
    assert isinstance(ds, ParametricDataset), (
        f"init_dataset with type='parametric' must return ParametricDataset, got {type(ds)}"
    )
