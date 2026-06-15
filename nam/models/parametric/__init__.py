"""
Parametric A2 models.

Importing this package registers:
- "ParametricWaveNet" in the model factory (factory.init)
- "parametric" in the dataset registry (init_dataset)

Both registrations happen on package import with no edits to factory.py or data.py.
"""

from ._dataset import ParametricConcatDataset, ParametricDataset
from ._model import ParametricWaveNet, ResidualAffineAdapter
from ..factory import register as _register
from nam.data import register_dataset_initializer as _register_dataset

# Register on package import — no factory.py edit needed.
_register("ParametricWaveNet", ParametricWaveNet.init_from_config)

# Register the parametric dataset type — no data.py edit needed.
_register_dataset("parametric", ParametricDataset.init_from_config)

__all__ = ["ParametricWaveNet", "ResidualAffineAdapter", "ParametricDataset", "ParametricConcatDataset"]
