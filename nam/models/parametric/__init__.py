"""
Parametric A2 models.

Importing this package registers "ParametricWaveNet" in the model factory so that
factory.init("ParametricWaveNet", config) works without any edits to factory.py.
"""

from ._model import ParametricWaveNet, ResidualAffineAdapter
from ..factory import register as _register

# Register on package import — no factory.py edit needed.
_register("ParametricWaveNet", ParametricWaveNet.init_from_config)

__all__ = ["ParametricWaveNet", "ResidualAffineAdapter"]
