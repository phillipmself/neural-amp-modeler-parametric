import os as _os
import sys as _sys

import pytest as _pytest

from nam.capture.audio import _enable_asio_on_windows
from nam.capture.audio import _raise_on_dropout
from nam.capture.audio import asio_com_apartment as _asio_com_apartment
from nam.capture.audio import current_device_sample_rates as _current_device_sample_rates
from nam.capture.audio import AudioDeviceError as _AudioDeviceError
from nam.capture.audio import AudioDropoutError as _AudioDropoutError
from nam.capture.audio import LATENCY_CHOICES as _LATENCY_CHOICES


class _Status:
    """Stands in for PortAudio's ``CallbackFlags``."""

    def __init__(self, **flags):
        self.input_underflow = flags.get("input_underflow", False)
        self.input_overflow = flags.get("input_overflow", False)
        self.output_underflow = flags.get("output_underflow", False)
        self.output_overflow = flags.get("output_overflow", False)
        self.priming_output = flags.get("priming_output", False)


def test_clean_stream_status_is_not_a_dropout():
    _raise_on_dropout(_Status(), latency="low", blocksize=0)


def test_blocking_api_flags_are_not_dropouts():
    """
    ``input_underflow``/``output_overflow`` belong to PortAudio's blocking read/write
    API and never mean lost audio on the callback stream the recorder uses; treating
    them as failures would refuse good captures.
    """
    _raise_on_dropout(
        _Status(input_underflow=True, output_overflow=True, priming_output=True),
        latency="low",
        blocksize=0,
    )


@_pytest.mark.parametrize("flag", ["input_overflow", "output_underflow"])
def test_lost_audio_raises(flag):
    with _pytest.raises(_AudioDropoutError) as excinfo:
        _raise_on_dropout(_Status(**{flag: True}), latency=0.002, blocksize=64)
    message = str(excinfo.value)
    assert "was not saved" in message
    # The message has to name the two settings that fix it, since the whole point of
    # the low-latency option is that the user can back off when it does not hold.
    assert "Stream latency" in message
    assert "0.002" in message
    assert "64" in message


def test_dropout_is_an_audio_device_error():
    # SessionWorker catches AudioDeviceError to surface engine failures in the GUI.
    assert issubclass(_AudioDropoutError, _AudioDeviceError)


def test_asio_is_enabled_on_windows(monkeypatch):
    monkeypatch.setattr(_sys, "platform", "win32")
    monkeypatch.delenv("SD_ENABLE_ASIO", raising=False)
    _enable_asio_on_windows()
    assert _os.environ["SD_ENABLE_ASIO"] == "1"


@_pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_asio_is_not_enabled_off_windows(monkeypatch, platform):
    """
    ASIO is Windows-only; setting this anywhere else would at best do nothing and at
    worst send ``sounddevice`` looking for a DLL variant that does not exist there.
    """
    monkeypatch.setattr(_sys, "platform", platform)
    monkeypatch.delenv("SD_ENABLE_ASIO", raising=False)
    _enable_asio_on_windows()
    assert "SD_ENABLE_ASIO" not in _os.environ


def test_an_explicit_asio_setting_is_left_alone(monkeypatch):
    """
    ``setdefault`` leaves a value the user set themselves untouched.

    This is not an off switch, and deliberately not tested as one: ``sounddevice``
    checks only whether ``SD_ENABLE_ASIO`` exists, never its value, so "0" still
    selects the ASIO DLL. On Windows this app is ASIO or nothing.
    """
    monkeypatch.setattr(_sys, "platform", "win32")
    monkeypatch.setenv("SD_ENABLE_ASIO", "0")
    _enable_asio_on_windows()
    assert _os.environ["SD_ENABLE_ASIO"] == "0"


@_pytest.mark.parametrize("platform", ["win32", "linux"])
@_pytest.mark.parametrize("allow_reinit", [True, False])
def test_sample_rate_poll_never_reinitialises_portaudio(
    monkeypatch, platform, allow_reinit
):
    """
    The rate poll runs on a timer, and off macOS a PortAudio reinit loads every
    installed ASIO driver. Even asked directly for one, this must not do it.
    """
    import sounddevice as _sd

    monkeypatch.setattr(_sys, "platform", platform)

    def _explode(*args, **kwargs):
        raise AssertionError("PortAudio was reinitialised on the rate poll path")

    monkeypatch.setattr(_sd, "_terminate", _explode)
    monkeypatch.setattr(_sd, "_initialize", _explode)

    # Empty, so callers fall back to the cached DeviceInfo.default_samplerate.
    assert _current_device_sample_rates(allow_reinit=allow_reinit) == {}


@_pytest.mark.parametrize("allow_reinit", [True, False])
def test_darwin_still_reads_coreaudio_and_ignores_the_flag(monkeypatch, allow_reinit):
    """
    macOS must keep its live CoreAudio read, which is what makes the sample-rate
    warning work there, and must keep ignoring ``allow_reinit`` as it always has.
    """
    import nam.capture.audio as _audio

    monkeypatch.setattr(_sys, "platform", "darwin")
    monkeypatch.setattr(
        _audio, "_coreaudio_sample_rates", lambda: {"Audient iD44": 96000.0}
    )
    assert _current_device_sample_rates(allow_reinit=allow_reinit) == {
        "Audient iD44": 96000.0
    }


def test_darwin_falls_back_to_empty_when_coreaudio_fails(monkeypatch):
    import nam.capture.audio as _audio

    monkeypatch.setattr(_sys, "platform", "darwin")

    def _boom():
        raise OSError("CoreAudio unavailable")

    monkeypatch.setattr(_audio, "_coreaudio_sample_rates", _boom)
    assert _current_device_sample_rates(allow_reinit=True) == {}


class _FakeOle32:
    """Stands in for ``ctypes.windll.ole32`` so both branches run on any platform."""

    # CoInitializeEx hands back an *unsigned* HRESULT through ctypes.
    S_OK = 0
    S_FALSE = 1
    RPC_E_CHANGED_MODE = 0x80010106

    def __init__(self, result=S_OK):
        self._result = result
        self.calls = []

    def CoInitializeEx(self, reserved, flags):
        self.calls.append(("CoInitializeEx", reserved, flags))
        return self._result

    def CoUninitialize(self):
        self.calls.append(("CoUninitialize",))


class _ExplodingWindll:
    def __getattr__(self, name):
        raise AssertionError(f"COM was touched off Windows (windll.{name})")


def _install_fake_windll(monkeypatch, ole32):
    import ctypes

    class _Windll:
        pass

    windll = _Windll()
    windll.ole32 = ole32
    # raising=False: there is no ctypes.windll at all on macOS or Linux.
    monkeypatch.setattr(ctypes, "windll", windll, raising=False)


@_pytest.mark.parametrize("result", [_FakeOle32.S_OK, _FakeOle32.S_FALSE])
def test_worker_thread_enters_a_single_threaded_apartment_on_windows(
    monkeypatch, result
):
    """
    ASIO drivers are in-process COM servers loaded on whichever thread opens the
    stream, so the capture worker has to be in an apartment or the open fails with
    "Failed to load ASIO driver".
    """
    ole32 = _FakeOle32(result)
    monkeypatch.setattr(_sys, "platform", "win32")
    _install_fake_windll(monkeypatch, ole32)

    with _asio_com_apartment():
        assert ole32.calls == [("CoInitializeEx", None, 0x2)]

    # Whatever we entered, we leave -- a leaked apartment outlives the QThread.
    assert ole32.calls[-1] == ("CoUninitialize",)


def test_apartment_is_left_even_if_the_capture_raises(monkeypatch):
    ole32 = _FakeOle32()
    monkeypatch.setattr(_sys, "platform", "win32")
    _install_fake_windll(monkeypatch, ole32)

    with _pytest.raises(RuntimeError):
        with _asio_com_apartment():
            raise RuntimeError("capture blew up")

    assert ole32.calls[-1] == ("CoUninitialize",)


def test_an_existing_mta_is_not_torn_down(monkeypatch):
    """
    RPC_E_CHANGED_MODE means someone else put this thread in a different apartment and
    still owns it; calling CoUninitialize against a apartment we never entered would
    unbalance their refcount.
    """
    ole32 = _FakeOle32(_FakeOle32.RPC_E_CHANGED_MODE)
    monkeypatch.setattr(_sys, "platform", "win32")
    _install_fake_windll(monkeypatch, ole32)

    with _asio_com_apartment():
        pass

    assert ("CoUninitialize",) not in ole32.calls


@_pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_no_com_anywhere_but_windows(monkeypatch, platform):
    """
    macOS has no COM and no ASIO; this must stay a plain pass-through there.
    """
    import ctypes

    monkeypatch.setattr(_sys, "platform", platform)
    monkeypatch.setattr(ctypes, "windll", _ExplodingWindll(), raising=False)

    ran = False
    with _asio_com_apartment():
        ran = True
    assert ran


def test_missing_ole32_does_not_take_down_a_capture(monkeypatch):
    """
    The apartment is a precondition, not the job. If COM cannot be reached at all the
    capture should still be attempted rather than failing before it starts.
    """
    import ctypes

    monkeypatch.setattr(_sys, "platform", "win32")
    # A bare object has no ``.ole32``, which is the AttributeError the code guards on.
    monkeypatch.setattr(ctypes, "windll", object(), raising=False)

    ran = False
    with _asio_com_apartment():
        ran = True
    assert ran


def test_latency_choices_run_from_safest_to_tightest():
    assert _LATENCY_CHOICES[0][1] == "high"
    assert _LATENCY_CHOICES[1][1] == "low"
    seconds = [value for _, value in _LATENCY_CHOICES if isinstance(value, float)]
    assert seconds == sorted(seconds, reverse=True)
