"""Small desktop launcher with isolated, logged and stoppable acquisition jobs."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any

import yaml


def checkout_root() -> Path:
    root = Path(__file__).resolve().parents[2]
    if not (root / "configs/dm_50.yaml").is_file():
        raise RuntimeError("Install evil_rig from its checkout with the supplied installer")
    return root


@dataclass
class LaunchOptions:
    subject: str = "bench"
    profile: str = "dm_50"
    mode: str = "bench"
    modality: str = "force"
    free_trials: int = 10
    forced_trials: int = 0
    lick_source: str = "optical"
    optical_clock: str = "streamdio"
    data_dir: str = ""
    max_minutes: float = 60.0

    def validate(self) -> None:
        import math
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", self.subject):
            raise ValueError("Subject must use letters, numbers, underscores, dots or hyphens")
        if self.profile not in {"dm_50", "dm_100", "dm_200", "centering_50"}:
            raise ValueError("Unknown task profile")
        if self.mode not in {"bench", "animal", "simulation"}:
            raise ValueError("Unknown run mode")
        if self.modality not in {"force", "risk", "delay"}:
            raise ValueError("Unknown modality")
        if self.lick_source not in {"capacitive", "optical"} or self.optical_clock not in {"streamdio", "bpod"}:
            raise ValueError("Unknown lick sensor or clock")
        if min(self.free_trials, self.forced_trials) < 0 or self.free_trials + self.forced_trials <= 0:
            raise ValueError("Choose at least one trial; counts cannot be negative")
        if not math.isfinite(self.max_minutes) or self.max_minutes <= 0:
            raise ValueError("Maximum minutes must be a finite positive number")
        if not self.data_dir:
            raise ValueError("Choose a data directory")


def prepare_config(root: Path, run_dir: Path, options: LaunchOptions) -> Path:
    """Save the exact launch configuration without changing any shipped profile."""
    options.validate()
    source = root / "configs" / f"{options.profile}.yaml"
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    raw["subject_id"] = options.subject
    raw["n_trials"] = options.free_trials + options.forced_trials
    raw.setdefault("task", {}).update(
        modality=options.modality, n_free_trials=options.free_trials,
        n_forced_trials=options.forced_trials, profile_name="custom",
    )
    raw["licks"].update(source=options.lick_source, optical_clock=options.optical_clock)
    data = Path(options.data_dir).expanduser().resolve()
    raw["results_dir"] = str(data)
    raw["camera"]["save_dir"] = str(data / "videos")
    raw["monitor"]["enabled"] = True
    # The commissioning artifact belongs to the rig, not to this run directory.
    cert = Path(raw["animal_use"]["certification_path"])
    raw["animal_use"]["certification_path"] = str((source.parent / cert).resolve())
    raw["welfare"]["subjects_dir"] = str((root / "subjects").resolve())
    if options.mode != "animal":
        raw["welfare"]["append_session_water"] = False
    path = run_dir / "effective.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def command_steps(action: str, options: LaunchOptions) -> list[tuple[str, list[str]]]:
    if action == "check":
        if options.mode == "animal":
            raise ValueError("Select Bench (no animal) to run the output test")
        if options.mode == "simulation":
            raise ValueError("Select Bench (no animal) for a hardware check")
        return [
            ("outputs_camera", ["--hardware-self-test"]),
            ("screen_sync", ["--frame2ttl-diagnostic", "--frame2ttl-flips", "6"]),
            ("screen_barcodes", ["--frame2ttl-barcode-diagnostic", "--frame2ttl-barcode-repetitions", "1"]),
        ]
    if action == "inputs":
        return [("inputs", ["--preflight", "--preflight-duration", "20"])]
    if action != "session":
        raise ValueError("Unknown launch action")
    if options.mode == "simulation":
        return [("session", ["--simulate", "--simulate-trials", str(options.free_trials + options.forced_trials)])]
    if options.mode == "bench":
        return [("session", ["--allow-uncertified", "--confirm-no-animal"])]
    return [("session", ["--confirm-animal-use"])]


def diagnostic_result(name: str, session_path: str) -> dict[str, Any] | None:
    """A successful process exit/data export does not prove the input test passed."""
    expected = {
        "screen_sync": ("frame2ttl_diagnostic_end", "detected"),
        "screen_barcodes": ("frame2ttl_barcode_diagnostic_end", "all_match"),
    }
    if name not in expected:
        return None
    event_type, flag = expected[name]
    document = json.loads(Path(session_path).read_text(encoding="utf-8"))
    payload = next((event["payload"] for event in reversed(document.get("events", []))
                    if event.get("event_type") == event_type), {})
    return {"passed": payload.get(flag) is True, "event_type": event_type, "result": payload}


class AcquisitionJob:
    def __init__(self, root: Path, action: str, options: LaunchOptions) -> None:
        options.validate()
        self.steps = command_steps(action, options)
        self.root, self.options, self.action = root, options, action
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_dir = root / "data/runtime/operator/runs" / stamp
        self.run_dir.mkdir(parents=True)
        self.config = prepare_config(root, self.run_dir, options)
        self.stop_path = self.run_dir / "stop.request"
        self.log_path = self.run_dir / "console.log"
        self.report: dict[str, Any] = {
            "status": "ready", "action": action, "options": asdict(options), "steps": [],
            "started_wall_s": time.time(), "config": str(self.config),
        }
        self.status = "Ready"
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread is not None:
            raise RuntimeError("Job already started")
        self.thread = threading.Thread(target=self._run, name="evil-rig-job", daemon=True)
        self.thread.start()

    @property
    def running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def stop(self) -> None:
        self.stop_path.touch()
        self.status = "Stopping; saving data and switching outputs off…"

    def _save_report(self) -> None:
        path = self.run_dir / "result.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.report, indent=2), encoding="utf-8")
        temporary.replace(path)

    def _run(self) -> None:
        self.report["status"] = "running"
        self._save_report()
        try:
            with self.log_path.open("wb", buffering=0) as output:
                for name, arguments in self.steps:
                    if self.stop_path.exists():
                        break
                    self.status = f"Running: {name.replace('_', ' ')}"
                    result_path = self.run_dir / f"{name}.json"
                    env = os.environ.copy()
                    limit = 180.0 if self.action != "session" else self.options.max_minutes * 60
                    env.update(PYTHONUTF8="1", PYTHONUNBUFFERED="1",
                               HF_DM_STOP_REQUEST=str(self.stop_path),
                               HF_DM_RUN_TIMEOUT_S=str(limit), HF_DM_RESULT_PATH=str(result_path))
                    command = [sys.executable, "-m", "hf_dm.cli", "--config", str(self.config),
                               "--no-session-dialog", "--no-certification-dialog", *arguments]
                    output.write((f"\n--- {name} ---\n" + json.dumps(command) + "\n").encode())
                    started = time.monotonic()
                    timed_out = False
                    process = subprocess.Popen(command, cwd=self.root, env=env, stdout=output,
                                               stderr=subprocess.STDOUT,
                                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                    stop_started = None
                    while process.poll() is None:
                        if time.monotonic() - started > limit:
                            timed_out = True
                            self.stop()
                        if self.stop_path.exists():
                            stop_started = stop_started or time.monotonic()
                            if time.monotonic() - stop_started > 30:
                                # The acquisition's persistent cleanup latch stays
                                # armed if its graceful shutdown cannot finish.
                                import psutil
                                parent = psutil.Process(process.pid)
                                children = parent.children(recursive=True)
                                for child in children:
                                    try:
                                        child.terminate()
                                    except psutil.NoSuchProcess:
                                        pass
                                process.terminate()
                                process.wait(timeout=5)
                                raise TimeoutError("Acquisition failed to stop; inspect cleanup interlock and log")
                        time.sleep(0.1)
                    step: dict[str, Any] = {"name": name, "returncode": process.returncode,
                                            "duration_s": time.monotonic() - started}
                    self.report["steps"].append(step)
                    if result_path.exists():
                        step.update(json.loads(result_path.read_text(encoding="utf-8")))
                    if step.get("cleanup_error"):
                        raise RuntimeError(f"Hardware cleanup failed: {step['cleanup_error']}")
                    if step.get("session_path"):
                        from hf_dm.data.session_export import validate_session
                        self.status = "Verifying saved session data…"
                        validation = validate_session(step["session_path"])
                        step["data_validation"] = validation
                        if not validation["valid"]:
                            raise RuntimeError("Saved-data validation failed; see result.json")
                        check = diagnostic_result(name, step["session_path"])
                        if check is not None:
                            step["diagnostic"] = check
                            if not check["passed"] and not self.stop_path.exists():
                                raise RuntimeError(f"{name}: recorded sensor check failed; see result.json")
                    elif process.returncode == 0:
                        raise RuntimeError("Acquisition exited without a saved session result")
                    if timed_out:
                        raise TimeoutError("Run exceeded its time limit; inspect saved data and log")
                    expected_stop = self.stop_path.exists() and str(step.get("error", "")).startswith("OperatorStop:")
                    if process.returncode and not expected_stop:
                        raise RuntimeError(f"{name} failed (exit {process.returncode}); see console.log")
                    self._save_report()
            self.report["status"] = "stopped" if self.stop_path.exists() else "passed"
            self.status = "Stopped; see saved data" if self.stop_path.exists() else "Completed; saved data verified"
        except BaseException as exc:
            self.report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            self.status = str(exc)
        finally:
            self.report["finished_wall_s"] = time.time()
            self._save_report()


def main() -> None:
    parser = argparse.ArgumentParser(description="evil_rig operator launcher")
    parser.add_argument("--check-install", action="store_true")
    args = parser.parse_args()
    root = checkout_root()
    if args.check_install:
        from importlib.metadata import version
        import tkinter
        print(json.dumps({"root": str(root), "python": sys.executable,
                          "version": version("bpod-hf-dm"), "tk": tkinter.TkVersion}))
        return
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    window = tk.Tk()
    window.title("evil_rig")
    window.geometry("840x640")
    window.minsize(720, 550)
    frame = ttk.Frame(window, padding=16)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="evil_rig", font=("Segoe UI", 19, "bold")).grid(row=0, column=0, sticky="w")
    ttk.Label(frame, text="Choose a session. Checks are available when needed.").grid(row=1, column=0, columnspan=4, sticky="w", pady=(0, 12))
    state_path = root / "data/runtime/operator/settings.json"
    defaults = asdict(LaunchOptions(data_dir=str(root / "data/sessions")))
    try:
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        defaults.update({k: v for k, v in saved.items() if k in defaults})
    except (OSError, ValueError, AttributeError):
        pass
    # Animal mode always requires an explicit selection for this launch.
    defaults["mode"] = "bench"
    fields = [("subject", "Subject", None), ("mode", "Mode", ["bench", "animal", "simulation"]),
              ("profile", "Task profile", ["dm_50", "dm_100", "dm_200", "centering_50"]),
              ("modality", "Modality", ["force", "risk", "delay"]),
              ("free_trials", "Free trials", None), ("forced_trials", "Forced trials", None),
              ("lick_source", "Lick sensor", ["optical", "capacitive"]),
              ("optical_clock", "Optical clock", ["streamdio", "bpod"]),
              ("max_minutes", "Maximum minutes", None), ("data_dir", "Data directory", None)]
    variables = {}
    widgets = []
    for index, (key, label, choices) in enumerate(fields):
        row, col = 2 + index // 2, (index % 2) * 2
        ttk.Label(frame, text=label).grid(row=row, column=col, sticky="w", padx=(0, 8), pady=5)
        var = tk.StringVar(value=str(defaults[key]))
        variables[key] = var
        widget = (ttk.Combobox(frame, textvariable=var, values=choices, state="readonly")
                  if choices else ttk.Entry(frame, textvariable=var))
        widget.grid(row=row, column=col + 1, sticky="ew", padx=(0, 14), pady=5)
        widgets.append(widget)
    frame.columnconfigure(1, weight=1)
    frame.columnconfigure(3, weight=1)
    ttk.Label(frame, text="Bench = no animal. Rig check briefly activates pump, force and sound.").grid(row=7, column=0, columnspan=4, sticky="w", pady=6)
    controls = ttk.Frame(frame)
    controls.grid(row=8, column=0, columnspan=4, sticky="w", pady=8)
    status = tk.StringVar(value="Ready")
    ttk.Label(frame, textvariable=status, wraplength=760).grid(row=9, column=0, columnspan=4, sticky="w", pady=6)
    log = tk.Text(frame, height=13, font=("Consolas", 9), state="disabled")
    log.grid(row=10, column=0, columnspan=4, sticky="nsew")
    frame.rowconfigure(10, weight=1)
    job: AcquisitionJob | None = None
    closing = False

    def launch(action: str) -> None:
        nonlocal job
        if job is not None and job.running:
            return
        try:
            values = {key: var.get() for key, var in variables.items()}
            values["free_trials"] = int(values["free_trials"])
            values["forced_trials"] = int(values["forced_trials"])
            values["max_minutes"] = float(values["max_minutes"])
            options = LaunchOptions(**values)
            job = AcquisitionJob(root, action, options)
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps(asdict(options), indent=2), encoding="utf-8")
            job.start()
        except Exception as exc:
            messagebox.showerror("Cannot start", str(exc), parent=window)

    def stop() -> None:
        if job is not None and job.running:
            job.stop()

    def open_path(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(str(path))
        else:
            subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])

    for title, action in [("Start session", "session"), ("Check rig", "check"), ("Watch inputs (20 s)", "inputs")]:
        ttk.Button(controls, text=title, command=lambda a=action: launch(a)).pack(side="left", padx=(0, 7))
    ttk.Button(controls, text="Stop", command=stop).pack(side="left", padx=7)
    ttk.Button(controls, text="Logs", command=lambda: open_path(job.run_dir if job else state_path.parent)).pack(side="left", padx=7)
    ttk.Button(controls, text="Data…", command=lambda: variables["data_dir"].set(
        filedialog.askdirectory(initialdir=variables["data_dir"].get()) or variables["data_dir"].get())).pack(side="left")

    def tick() -> None:
        if job is not None:
            status.set(job.status)
            try:
                with job.log_path.open("rb") as handle:
                    handle.seek(max(0, job.log_path.stat().st_size - 8000))
                    content = handle.read().decode("utf-8", errors="replace")
                log.configure(state="normal")
                log.delete("1.0", "end")
                log.insert("end", content)
                log.see("end")
                log.configure(state="disabled")
            except OSError:
                pass
        if closing and (job is None or not job.running):
            window.destroy()
        else:
            window.after(300, tick)

    def close() -> None:
        nonlocal closing
        closing = True
        stop()

    window.protocol("WM_DELETE_WINDOW", close)
    tick()
    window.mainloop()


if __name__ == "__main__":
    main()
