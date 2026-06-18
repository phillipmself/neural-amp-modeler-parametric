import importlib

from nam.train import gui as legacy_gui


def test_parametric_gui_import_does_not_change_legacy_gui_surface():
    parametric_gui = importlib.import_module("nam.train.gui.parametric")

    assert hasattr(legacy_gui, "GUI")
    assert hasattr(parametric_gui, "GUI")
    assert callable(parametric_gui.run)
