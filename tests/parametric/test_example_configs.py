import json
from pathlib import Path

from nam.models.parametric import ParametricWaveNet


_ROOT = Path(__file__).resolve().parents[2]
_PACKED_MODEL_PATH = _ROOT / "nam" / "train" / "_resources" / "config_model_packed.json"
_EXAMPLE_MODEL_PATH = _ROOT / "nam_full_configs" / "parametric" / "model.json"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def test_example_model_matches_channels_8_topology_and_hyperparameters():
    packed = _load_json(_PACKED_MODEL_PATH)
    example = _load_json(_EXAMPLE_MODEL_PATH)

    channels_8 = next(
        submodel
        for submodel in packed["net"]["config"]["submodels"]
        if submodel["name"] == "channels_8"
    )

    assert example["net"]["name"] == "ParametricWaveNet"
    assert example["net"]["config"]["layers_configs"] == channels_8["config"]["layers_configs"]
    assert example["net"]["config"]["head_scale"] == channels_8["config"]["head_scale"]
    assert example["loss"] == packed["loss"]
    assert example["optimizer"]["lr"] == packed["optimizer"]["lr"]
    assert example["optimizer"]["weight_decay"] == packed["optimizer"]["weight_decay"]
    assert example["optimizer"]["adapter_lr"] == 5.0e-4
    assert (
        example["optimizer"]["adapter_weight_decay"]
        == packed["optimizer"]["weight_decay"]
    )
    assert set(example["optimizer"]) == set(packed["optimizer"]) | {
        "adapter_lr",
        "adapter_weight_decay",
    }
    assert example["lr_scheduler"] == packed["lr_scheduler"]
    assert "params" in example["net"]["config"]
    assert example["net"]["config"]["adapter_hidden_dim"] == 8
    assert "adapter_activation" in example["net"]["config"]


def test_example_model_net_config_initializes_parametric_wavenet():
    example = _load_json(_EXAMPLE_MODEL_PATH)

    model = ParametricWaveNet.init_from_config(example["net"]["config"])

    assert isinstance(model, ParametricWaveNet)


def test_example_model_adapter_param_count_matches_shared_encoder_design():
    example = _load_json(_EXAMPLE_MODEL_PATH)
    model = ParametricWaveNet.init_from_config(example["net"]["config"])

    adapter_param_count = sum(param.numel() for param in model._adapter.parameters())

    assert adapter_param_count == 3336
