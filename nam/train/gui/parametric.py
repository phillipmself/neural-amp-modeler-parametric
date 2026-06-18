"""
Standalone GUI for parametric NAM training.
"""

from __future__ import annotations

import tkinter as _tk
from pathlib import Path as _Path
from tkinter import filedialog as _filedialog
from tkinter import messagebox as _messagebox
from typing import Any as _Any
from typing import Dict as _Dict
from typing import List as _List

from nam import __version__
from nam.train import core as _core
from nam.train import full as _full
from nam.train.gui import AdvancedOptions as _AdvancedOptions
from nam.train.gui import _parametric as _helpers
from nam.train.gui._resources import settings as _settings
from nam.util import timestamp as _timestamp

_BUTTON_WIDTH = 18
_PATH_LABEL_WIDTH = 80


def _non_negative_int_or_default(val: str, default: int) -> int:
    try:
        parsed = int(val.strip())
    except ValueError:
        return default
    return max(parsed, 0)


def _optional_int_or_default(val: str, default: int | None) -> int | None:
    stripped = val.strip()
    if stripped == "":
        return None
    try:
        return int(stripped)
    except ValueError:
        return default


def _optional_float_or_default(val: str, default: float | None) -> float | None:
    stripped = val.strip()
    if stripped == "":
        return None
    try:
        return float(stripped)
    except ValueError:
        return default


class _AdvancedOptionsWindow(object):
    def __init__(self, resume_main, parent: "GUI"):
        self._parent = parent
        self._resume_main = resume_main
        self._closed = False
        self._root = _tk.Toplevel(parent._root)
        self._root.title("Advanced Options")
        self._root.transient(parent._root)
        self._root.protocol("WM_DELETE_WINDOW", self._close)

        self._num_epochs = _tk.StringVar(
            value=str(self._parent.advanced_options.num_epochs)
        )
        self._latency = _tk.StringVar(
            value=""
            if self._parent.advanced_options.latency is None
            else str(self._parent.advanced_options.latency)
        )
        self._threshold_esr = _tk.StringVar(
            value=""
            if self._parent.advanced_options.threshold_esr is None
            else str(self._parent.advanced_options.threshold_esr)
        )

        self._build_layout()
        self._root.grab_set()

    def _build_layout(self):
        frame = _tk.Frame(self._root)
        frame.pack(padx=12, pady=12)

        self._make_row(frame, row=0, label="Epochs", variable=self._num_epochs)
        self._make_row(frame, row=1, label="Reamp latency", variable=self._latency)
        self._make_row(
            frame,
            row=2,
            label="Threshold ESR",
            variable=self._threshold_esr,
        )

        button_frame = _tk.Frame(self._root)
        button_frame.pack(anchor="e", padx=12, pady=(0, 12))
        _tk.Button(
            button_frame,
            text="Ok",
            width=_BUTTON_WIDTH,
            command=self._apply_and_close,
        ).pack(side=_tk.LEFT)

    def _make_row(
        self,
        frame: _tk.Frame,
        *,
        row: int,
        label: str,
        variable: _tk.StringVar,
    ):
        _tk.Label(frame, text=label, anchor="w", width=16).grid(
            row=row,
            column=0,
            sticky="w",
            padx=(0, 8),
            pady=4,
        )
        _tk.Entry(frame, textvariable=variable, width=16).grid(
            row=row,
            column=1,
            sticky="ew",
            pady=4,
        )

    def _apply_and_close(self):
        self._parent.advanced_options.num_epochs = _non_negative_int_or_default(
            self._num_epochs.get(),
            self._parent.advanced_options.num_epochs,
        )
        self._parent.advanced_options.latency = _optional_int_or_default(
            self._latency.get(),
            self._parent.advanced_options.latency,
        )
        self._parent.advanced_options.threshold_esr = _optional_float_or_default(
            self._threshold_esr.get(),
            self._parent.advanced_options.threshold_esr,
        )
        self._close()

    def _close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self._root.grab_release()
        except _tk.TclError:
            pass
        self._root.destroy()
        self._resume_main()


class GUI(object):
    def __init__(self):
        self._root = _tk.Tk()
        self._root.title(f"NAM Parametric Trainer - v{__version__}")

        self._input_path: str | None = None
        self._training_destination: str | None = None
        self._param_rows: _List[_Dict[str, _tk.StringVar]] = []
        self._capture_rows: _List[_Dict[str, _Any]] = []
        self.advanced_options = _AdvancedOptions(
            num_epochs=_helpers.default_num_epochs(),
            latency=None,
            ignore_checks=False,
            threshold_esr=None,
        )

        self._build_layout()
        self._add_param_row()
        self._render_param_rows()
        self._render_capture_rows()
        self._update_train_button_state()

    def mainloop(self):
        self._root.mainloop()

    def _build_layout(self):
        self._frame_paths = _tk.Frame(self._root)
        self._frame_paths.pack(anchor="w", padx=12, pady=8)

        self._input_label = self._make_path_row(
            self._frame_paths,
            button_text="Input Audio",
            path_type="file",
            path_key=_settings.PathKey.INPUT_FILE,
            row=0,
        )
        self._destination_label = self._make_path_row(
            self._frame_paths,
            button_text="Train Destination",
            path_type="directory",
            path_key=_settings.PathKey.TRAINING_DESTINATION,
            row=1,
        )

        self._frame_params = _tk.LabelFrame(self._root, text="Parameters")
        self._frame_params.pack(fill="x", padx=12, pady=8)
        self._params_grid = _tk.Frame(self._frame_params)
        self._params_grid.pack(fill="x", padx=8, pady=8)
        self._add_param_button = _tk.Button(
            self._frame_params,
            text="Add Parameter",
            width=_BUTTON_WIDTH,
            command=self._add_param_row,
        )
        self._add_param_button.pack(anchor="w", padx=8, pady=(0, 8))

        self._frame_captures = _tk.LabelFrame(self._root, text="Output Captures")
        self._frame_captures.pack(fill="both", expand=True, padx=12, pady=8)
        self._add_capture_button = _tk.Button(
            self._frame_captures,
            text="Add Output Files",
            width=_BUTTON_WIDTH,
            command=self._add_output_files,
        )
        self._add_capture_button.pack(anchor="w", padx=8, pady=(8, 0))
        self._captures_grid = _tk.Frame(self._frame_captures)
        self._captures_grid.pack(fill="both", expand=True, padx=8, pady=8)

        self._frame_options = _tk.Frame(self._root)
        self._frame_options.pack(anchor="w", padx=12, pady=(0, 8))
        self._silent_training_var = _tk.BooleanVar(value=False)
        self._save_plot_var = _tk.BooleanVar(value=True)
        _tk.Checkbutton(
            self._frame_options,
            text="Silent run",
            variable=self._silent_training_var,
        ).pack(anchor="w")
        _tk.Checkbutton(
            self._frame_options,
            text="Save ESR plot automatically",
            variable=self._save_plot_var,
        ).pack(anchor="w")

        self._frame_actions = _tk.Frame(self._root)
        self._frame_actions.pack(anchor="e", padx=12, pady=(0, 12))
        self._advanced_options_button = _tk.Button(
            self._frame_actions,
            text="Advanced options...",
            width=_BUTTON_WIDTH,
            command=self._open_advanced_options,
        )
        self._advanced_options_button.pack(side=_tk.LEFT, padx=(0, 8))
        self._train_button = _tk.Button(
            self._frame_actions,
            text="Train",
            width=_BUTTON_WIDTH,
            command=self._train,
        )
        self._train_button.pack(side=_tk.LEFT)

    def _make_path_row(
        self,
        frame: _tk.Frame,
        *,
        button_text: str,
        path_type: str,
        path_key: _settings.PathKey,
        row: int,
    ) -> _tk.Label:
        _tk.Button(
            frame,
            text=button_text,
            width=_BUTTON_WIDTH,
            command=lambda: self._set_path(path_type=path_type, path_key=path_key),
        ).grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
        label = _tk.Label(
            frame,
            width=_PATH_LABEL_WIDTH,
            anchor="w",
            justify="left",
            text=f"Select {button_text.lower()}",
        )
        label.grid(row=row, column=1, sticky="w")
        return label

    def _default_capture_values(self) -> _List[str]:
        return [row["default"].get() for row in self._param_rows]

    def _silent_training(self) -> bool:
        variable = getattr(self, "_silent_training_var", None)
        return False if variable is None else bool(variable.get())

    def _save_plot(self) -> bool:
        variable = getattr(self, "_save_plot_var", None)
        return True if variable is None else bool(variable.get())

    def _raw_param_rows(self) -> _List[_Dict[str, str]]:
        return [
            {
                "name": row["name"].get(),
                "min": row["min"].get(),
                "max": row["max"].get(),
                "default": row["default"].get(),
            }
            for row in self._param_rows
        ]

    def _raw_capture_rows(self) -> _List[_Dict[str, _Any]]:
        return [
            {
                "output_path": row["output_path"],
                "values": [var.get() for var in row["values"]],
            }
            for row in self._capture_rows
        ]

    def _replace_capture_rows(self, rows: _List[_Dict[str, _Any]]):
        self._capture_rows = [
            {
                "output_path": row["output_path"],
                "values": [self._make_bound_var(value) for value in row["values"]],
            }
            for row in rows
        ]
        self._render_capture_rows()
        self._update_train_button_state()

    def _make_bound_var(self, value: _Any, on_write=None) -> _tk.StringVar:
        var = _tk.StringVar(value=str(value))
        var.trace_add("write", lambda *_args: self._update_train_button_state())
        if on_write is not None:
            var.trace_add("write", lambda *_args: on_write())
        return var

    def _add_param_row(self):
        self._param_rows.append(
            {
                "name": self._make_bound_var("", on_write=self._render_capture_rows),
                "min": self._make_bound_var("0.0"),
                "max": self._make_bound_var("1.0"),
                "default": self._make_bound_var("0.0"),
            }
        )
        synchronized = _helpers.synchronize_capture_rows(
            self._raw_capture_rows(),
            self._default_capture_values(),
        )
        self._replace_capture_rows(synchronized)
        self._render_param_rows()

    def _remove_param_row(self, index: int):
        self._param_rows.pop(index)
        synchronized = _helpers.synchronize_capture_rows(
            self._raw_capture_rows(),
            self._default_capture_values(),
            removed_index=index,
        )
        self._replace_capture_rows(synchronized)
        self._render_param_rows()

    def _render_param_rows(self):
        for child in self._params_grid.winfo_children():
            child.destroy()
        headers = ("Name", "Min", "Max", "Default", "")
        for column, header in enumerate(headers):
            _tk.Label(self._params_grid, text=header, anchor="w").grid(
                row=0, column=column, sticky="w", padx=4, pady=2
            )
        for row_index, row in enumerate(self._param_rows, start=1):
            for column, key in enumerate(("name", "min", "max", "default")):
                _tk.Entry(
                    self._params_grid,
                    textvariable=row[key],
                    width=16 if key == "name" else 10,
                ).grid(row=row_index, column=column, sticky="ew", padx=4, pady=2)
            _tk.Button(
                self._params_grid,
                text="Remove",
                command=lambda index=row_index - 1: self._remove_param_row(index),
            ).grid(row=row_index, column=4, sticky="w", padx=4, pady=2)
        self._update_train_button_state()

    def _render_capture_rows(self):
        for child in self._captures_grid.winfo_children():
            child.destroy()
        _tk.Label(self._captures_grid, text="Output").grid(
            row=0, column=0, sticky="w", padx=4, pady=2
        )
        for i, row in enumerate(self._param_rows, start=1):
            label_text = row["name"].get().strip() or f"Param {i}"
            _tk.Label(self._captures_grid, text=label_text).grid(
                row=0, column=i, sticky="w", padx=4, pady=2
            )
        _tk.Label(self._captures_grid, text="").grid(
            row=0, column=len(self._param_rows) + 1, sticky="w"
        )

        for row_index, row in enumerate(self._capture_rows, start=1):
            _tk.Label(
                self._captures_grid,
                text=_Path(row["output_path"]).name,
                anchor="w",
                justify="left",
            ).grid(row=row_index, column=0, sticky="w", padx=4, pady=2)
            for column, value_var in enumerate(row["values"], start=1):
                _tk.Entry(
                    self._captures_grid,
                    textvariable=value_var,
                    width=10,
                ).grid(row=row_index, column=column, sticky="ew", padx=4, pady=2)
            _tk.Button(
                self._captures_grid,
                text="Remove",
                command=lambda index=row_index - 1: self._remove_capture_row(index),
            ).grid(
                row=row_index,
                column=len(self._param_rows) + 1,
                sticky="w",
                padx=4,
                pady=2,
            )

    def _remove_capture_row(self, index: int):
        self._capture_rows.pop(index)
        self._render_capture_rows()
        self._update_train_button_state()

    def _set_path(self, *, path_type: str, path_key: _settings.PathKey):
        last_path = _settings.get_last_path(path_key)
        initial_dir = None if last_path is None else (
            str(last_path if last_path.is_dir() else last_path.parent)
        )
        if path_type == "file":
            result = _filedialog.askopenfilename(initialdir=initial_dir)
        elif path_type == "directory":
            result = _filedialog.askdirectory(initialdir=initial_dir)
        else:
            raise ValueError(path_type)
        if result == "":
            return
        _settings.set_last_path(path_key, _Path(result))
        if path_key == _settings.PathKey.INPUT_FILE:
            self._input_path = result
            self._input_label["text"] = result
        else:
            self._training_destination = result
            self._destination_label["text"] = result
        self._update_train_button_state()

    def _add_output_files(self):
        last_path = _settings.get_last_path(_settings.PathKey.OUTPUT_FILE)
        initial_dir = None if last_path is None else (
            str(last_path if last_path.is_dir() else last_path.parent)
        )
        result = _filedialog.askopenfilenames(initialdir=initial_dir)
        if not result:
            return
        _settings.set_last_path(_settings.PathKey.OUTPUT_FILE, _Path(result[0]))
        merged_rows = _helpers.add_unique_capture_rows(
            self._raw_capture_rows(),
            result,
            self._default_capture_values(),
        )
        self._replace_capture_rows(merged_rows)

    def _update_train_button_state(self, *_args):
        ready = self._input_path is not None and self._training_destination is not None
        if ready:
            try:
                param_specs = _helpers.build_param_specs(self._raw_param_rows())
                _helpers.validate_capture_rows(self._raw_capture_rows(), param_specs)
            except Exception:
                ready = False
        self._train_button["state"] = _tk.NORMAL if ready else _tk.DISABLED

    def _set_all_widget_states_to(self, state):
        def set_state(widget: _Any):
            try:
                widget.configure(state=state)
            except (_tk.TclError, TypeError):
                pass
            for child in widget.winfo_children():
                set_state(child)

        set_state(self._root)

    def _disable(self):
        self._set_all_widget_states_to(_tk.DISABLED)

    def _resume(self):
        self._set_all_widget_states_to(_tk.NORMAL)
        self._update_train_button_state()

    def _wait_while_func(self, func, *args, **kwargs):
        self._disable()
        func(self._resume, *args, **kwargs)

    def _open_advanced_options(self):
        self._wait_while_func(lambda resume: _AdvancedOptionsWindow(resume, self))

    def _build_failed_validation_message(self, validations: _Dict[str, _core.DataValidationOutput]) -> str:
        def make_message(output_path: str, validation: _core.DataValidationOutput) -> str:
            msg = f"{_Path(output_path).name}:\n"
            if not validation.sample_rate.passed:
                msg += (
                    f"  Different sample rates for input ({validation.sample_rate.input}) "
                    f"and output ({validation.sample_rate.output}).\n"
                )
            if not validation.length.passed:
                msg += (
                    "  Input and output lengths do not match closely enough "
                    f"(delta {validation.length.delta_seconds:.2f} s).\n"
                )
            if validation.latency.manual is None:
                if validation.latency.calibration.warnings.matches_lookahead:
                    msg += "  Latency calibration hit the lookahead limit.\n"
                if validation.latency.calibration.warnings.disagreement_too_high:
                    msg += "  Latency calibration estimates disagree too much.\n"
                if validation.latency.calibration.warnings.not_detected:
                    msg += "  Latency calibration impulses were not detected.\n"
            if not validation.checks.passed:
                msg += "  A standardized-input data check failed.\n"
            if not validation.pytorch.passed:
                msg += "  Dataset initialization failed:\n"
                if not validation.pytorch.train.passed:
                    msg += f"    train: {validation.pytorch.train.msg}\n"
                if not validation.pytorch.validation.passed:
                    msg += f"    validation: {validation.pytorch.validation.msg}\n"
            return msg

        body = "".join(
            make_message(output_path, validation)
            for output_path, validation in validations.items()
            if not validation.passed
        )
        return "The following output files failed checks:\n\n" + body

    def _validate_for_training(self):
        if self._input_path is None:
            raise RuntimeError("Input path was not selected.")
        input_path = _Path(self._input_path)
        if not _core.validate_input(input_path).passed:
            _messagebox.showerror(
                "Invalid Input",
                f"Input file {input_path} is not recognized as a standardized input file.",
            )
            return None

        param_specs = _helpers.build_param_specs(self._raw_param_rows())
        capture_rows = self._raw_capture_rows()
        user_latency = self.advanced_options.latency
        validation_outputs: _Dict[str, _core.DataValidationOutput] = {}
        validated_rows: _List[tuple[_Dict[str, _Any], _Path, _core.DataValidationOutput]] = []
        for row in capture_rows:
            output_path = _Path(row["output_path"])
            validation = _core.validate_data(
                input_path,
                output_path,
                user_latency=user_latency,
                silent=self._silent_training(),
            )
            validation_outputs[str(output_path)] = validation
            validated_rows.append((row, output_path, validation))

        if any(not validation.passed for validation in validation_outputs.values()):
            msg = self._build_failed_validation_message(validation_outputs)
            if all(validation.passed_critical for validation in validation_outputs.values()):
                if not _messagebox.askyesno("Output Warning", msg + "\nContinue anyway?"):
                    return None
            else:
                _messagebox.showerror("Output Validation Failed", msg + "\nCritical errors found.")
                return None

        validated_captures: _List[_Dict[str, _Any]] = []
        latency_failures: _List[str] = []
        for row, output_path, validation in validated_rows:
            try:
                delay = _core.get_final_latency(validation.latency)
            except Exception as e:
                latency_failures.append(f"{output_path.name}: {e}")
                continue
            validated_captures.append(
                {
                    "output_path": str(output_path),
                    "values": row["values"],
                    "delay": delay,
                }
            )
        if latency_failures:
            _messagebox.showerror(
                "Latency Validation Failed",
                "Could not determine a usable latency for:\n\n"
                + "\n".join(latency_failures),
            )
            return None

        captures = _helpers.validate_capture_rows(validated_captures, param_specs)
        gaps = _helpers.find_missing_param_extrema(param_specs, captures)
        if gaps and not _messagebox.askyesno(
            "Coverage Warning", _helpers.format_coverage_message(gaps)
        ):
            return None
        return param_specs, captures

    def _train(self):
        try:
            validated = self._validate_for_training()
            if validated is None:
                return
            param_specs, captures = validated
            assert self._input_path is not None
            assert self._training_destination is not None
            outdir = _Path(self._training_destination, _timestamp())
            outdir.mkdir(parents=True, exist_ok=False)

            data_config = _helpers.build_parametric_data_config(
                self._input_path, param_specs, captures
            )
            model_config = _helpers.build_parametric_model_config(param_specs)
            learning_config = _helpers.build_learning_config(
                num_epochs=self.advanced_options.num_epochs,
                batch_size=_helpers.default_batch_size(),
                threshold_esr=self.advanced_options.threshold_esr,
            )
            silent = self._silent_training()
            save_plot = self._save_plot()

            self._train_button["state"] = _tk.DISABLED
            self._root.update_idletasks()
            _full.main(
                data_config,
                model_config,
                learning_config,
                outdir,
                no_show=silent,
                make_plots=not silent,
                save_plot=save_plot,
            )
            _messagebox.showinfo(
                "Training Complete",
                f"Parametric model exported to:\n{outdir / 'model.nam'}",
            )
        except Exception as e:
            _messagebox.showerror("Training Failed", str(e))
        finally:
            self._update_train_button_state()


def run():
    gui = GUI()
    gui.mainloop()


if __name__ == "__main__":
    run()
