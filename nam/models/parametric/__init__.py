"""
Parametric A2 models.

Importing this package registers:
- "ParametricWaveNet" in the model factory (factory.init)
- "parametric" in the dataset registry (init_dataset)

Both registrations happen on package import with no edits to factory.py or data.py.
"""

from ._dataset import ParametricConcatDataset, ParametricDataset, _build_parametric_concat
from ._loader import load_parametric_nam
from ._model import ParametricWaveNet, ResidualAffineAdapter
from ._spec import ParamSpec
from ..factory import register as _register
from nam.data import register_dataset_initializer as _register_dataset
from nam.data import register_concat_dataset_initializer as _register_concat

# Register on package import — no factory.py edit needed.
_register("ParametricWaveNet", ParametricWaveNet.init_from_config)

# Register the parametric dataset type — no data.py edit needed.
_register_dataset("parametric", ParametricDataset.init_from_config)

# Register the multi-capture (list-based) parametric concat factory so that
# init_dataset dispatches to ParametricConcatDataset instead of ConcatDataset
# when type=="parametric" and the split value is a list.
_register_concat("parametric", _build_parametric_concat)

__all__ = [
    "ParamSpec",
    "ParametricWaveNet",
    "ResidualAffineAdapter",
    "ParametricDataset",
    "ParametricConcatDataset",
    "load_parametric_nam",
]
