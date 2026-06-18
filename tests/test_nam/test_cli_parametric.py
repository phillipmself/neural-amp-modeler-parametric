import importlib
from pathlib import Path


def test_cli_exports_parametric_gui_entrypoint():
    cli = importlib.import_module("nam.cli")

    assert callable(cli.nam_parametric_gui)


def test_pyproject_declares_parametric_script_and_package():
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    text = pyproject.read_text()

    assert 'nam-parametric = "nam.cli:nam_parametric_gui"' in text
    assert '"nam.models.parametric"' in text


def test_parametric_package_importable():
    module = importlib.import_module("nam.models.parametric")

    assert hasattr(module, "ParametricWaveNet")
