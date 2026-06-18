import builtins
import importlib

from nam.train import gui as legacy_gui


def test_parametric_gui_import_does_not_change_legacy_gui_surface():
    parametric_gui = importlib.import_module("nam.train.gui.parametric")

    assert hasattr(legacy_gui, "GUI")
    assert hasattr(parametric_gui, "GUI")
    assert callable(parametric_gui.run)


def test_parametric_gui_run_falls_back_on_install_failure(monkeypatch):
    parametric_gui = importlib.import_module("nam.train.gui.parametric")
    original_import = builtins.__import__
    install_error_calls = []

    try:
        with monkeypatch.context() as context:
            def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name == "nam.train.gui" and "_parametric" in fromlist:
                    raise ImportError("Simulated install failure")
                return original_import(name, globals, locals, fromlist, level)

            context.setattr(builtins, "__import__", fake_import)
            importlib.reload(parametric_gui)
            assert parametric_gui._install_is_valid is False
            context.setattr(
                parametric_gui,
                "_install_error",
                lambda: install_error_calls.append(True),
            )
            context.setattr(
                parametric_gui,
                "GUI",
                lambda: (_ for _ in ()).throw(AssertionError("GUI should not be constructed")),
            )

            parametric_gui.run()

        assert install_error_calls == [True]
    finally:
        importlib.reload(parametric_gui)
    assert parametric_gui._install_is_valid is True
