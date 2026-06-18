"""Generate the UI-like parametric WaveNet fixture for loader/plugin smoke tests.

This script intentionally uses the example config that mirrors the local UI trainer's
single-submodel "channels_8" topology, then exports the resulting model in two forms:

* ``parametric_wavenet_standard.nam``  - canonical NAM export for plugin/runtime tests
* ``parametric_wavenet_standard.json`` - pretty-printed JSON copy for inspection

The exported model is untrained; it is meant to verify loader behavior, parameter UI
population, and basic audio-path execution without requiring identity behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from nam.models.parametric import ParametricWaveNet

_DEFAULT_CONFIG_PATH = (
    _ROOT
    / "docs"
    / "parametric-a2"
    / "example-configs"
    / "ui-like-3capture-gain-bright"
    / "model.json"
)
_DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent
_DEFAULT_BASENAME = "parametric_wavenet_standard"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the UI-like ParametricWaveNet fixture used for loader/plugin "
            "smoke tests."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG_PATH,
        help="Path to the trainer-style example config JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_DEFAULT_OUTPUT_DIR,
        help="Directory where the .nam and .json fixture files will be written.",
    )
    parser.add_argument(
        "--basename",
        default=_DEFAULT_BASENAME,
        help="Basename for the generated output files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Torch random seed for deterministic fixture regeneration.",
    )
    return parser.parse_args()


def _load_model_config(config_path: Path) -> dict:
    config_dict = json.loads(config_path.read_text())
    return config_dict["net"]["config"]


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    model_config = _load_model_config(args.config)
    model = ParametricWaveNet.init_from_config(model_config)
    model.export(args.output_dir, basename=args.basename)

    nam_path = args.output_dir / f"{args.basename}.nam"
    export_dict = json.loads(nam_path.read_text())
    json_path = args.output_dir / f"{args.basename}.json"
    json_path.write_text(json.dumps(export_dict, indent=4) + "\n")

    print(f"config:   {args.config}")
    print(f"nam:      {nam_path}")
    print(f"json:     {json_path}")
    print(f"params:   {[p['name'] for p in export_dict['config']['params']]}")
    print(f"weights:  {len(export_dict['weights'])}")


if __name__ == "__main__":
    main()
