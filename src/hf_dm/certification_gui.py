"""Tk remediation window shown when an animal-use session is blocked."""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

from hf_dm.certification import evaluate_animal_use_gate
from hf_dm.certification_workflow import (
    append_gui_audit,
    certification_command,
    certification_workspace,
    ensure_certification_path,
    load_operator_state,
    save_operator_state,
    write_json_artifact,
)
from hf_dm.config import SessionConfig
from hf_dm.durable_io import fsync_best_effort
from hf_dm.subject_log import append_weight_entry
from hf_dm.valve_calibration import run_guided_valve_calibration


@dataclass(frozen=True)
class CertificationDialogResult:
    continue_session: bool
    operator_confirmed: bool


def run_certification_dialog(
    config: SessionConfig,
    report: dict[str, Any],
) -> CertificationDialogResult:
    """Open the fail-closed operator workflow and return only after a decision."""

    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog, ttk

    ensure_certification_path(config)
    state_error: str | None = None
    try:
        saved = load_operator_state(config)
    except Exception as exc:
        saved = {}
        state_error = f"{type(exc).__name__}: {exc}"

    root = tk.Tk()
    root.title("HF-DM animal-use readiness")
    root.geometry("920x720")
    root.minsize(780, 620)
    result: dict[str, Any] = {
        "continue": False,
        "operator": "",
        "operator_confirmed": False,
    }
    running: dict[str, Any] = {}

    operator_var = tk.StringVar(value=str(saved.get("operator") or ""))
    calibration_var = tk.StringVar(
        value=str(saved.get("reward_calibration_path") or config.reward.calibration_path or "")
    )
    evidence_var = tk.StringVar(value=str(saved.get("physical_evidence_path") or ""))
    failsafe_var = tk.StringVar(
        value=str(
            saved.get("hardware_failsafe_revision")
            or config.animal_use.hardware_failsafe_revision
            or ""
        )
    )
    cap_var = tk.StringVar(
        value=""
        if config.welfare.max_cumulative_reward_ul is None
        else str(config.welfare.max_cumulative_reward_ul)
    )
    subjects_var = tk.StringVar(
        value=str(saved.get("subjects_dir") or config.welfare.subjects_dir)
    )
    run_confirmed_var = tk.BooleanVar(value=False)
    safety_vars = {
        "no_animal_present": tk.BooleanVar(value=False),
        "collection_vessel_installed": tk.BooleanVar(value=False),
        "force_mechanism_clear": tk.BooleanVar(value=False),
        "speaker_exposure_safe": tk.BooleanVar(value=False),
        "emergency_power_control_available": tk.BooleanVar(value=False),
    }
    status_var = tk.StringVar(value="Animal-use session blocked")

    header = ttk.Frame(root, padding=12)
    header.pack(fill="x")
    ttk.Label(header, text="Animal-use readiness", font=("TkDefaultFont", 15, "bold")).pack(
        anchor="w"
    )
    ttk.Label(
        header,
        text=(
            "No animal task output will be enabled until every production gate is PASS. "
            "Physical measurements and approvals are never inferred by this window."
        ),
        wraplength=860,
    ).pack(anchor="w", pady=(4, 0))
    ttk.Label(header, textvariable=status_var).pack(anchor="w", pady=(6, 0))

    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True, padx=12, pady=(0, 8))
    status_tab = ttk.Frame(notebook, padding=8)
    setup_tab = ttk.Frame(notebook, padding=8)
    actions_tab = ttk.Frame(notebook, padding=8)
    notebook.add(status_tab, text="Failed checks")
    notebook.add(setup_tab, text="Operator data")
    notebook.add(actions_tab, text="Calibration & certification")

    tree = ttk.Treeview(status_tab, columns=("status", "detail"), show="headings")
    tree.heading("status", text="Status")
    tree.heading("detail", text="Check and detail")
    tree.column("status", width=80, stretch=False)
    tree.column("detail", width=730)
    scroll = ttk.Scrollbar(status_tab, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=scroll.set)
    tree.pack(side="left", fill="both", expand=True)
    scroll.pack(side="right", fill="y")

    setup_tab.columnconfigure(1, weight=1)

    def field(row: int, label: str, variable: Any, browse: Any = None) -> None:
        ttk.Label(setup_tab, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(setup_tab, textvariable=variable).grid(
            row=row, column=1, sticky="ew", padx=5, pady=5
        )
        if browse is not None:
            ttk.Button(setup_tab, text="Browse…", command=browse).grid(
                row=row, column=2, padx=5, pady=5
            )

    def choose_calibration() -> None:
        value = filedialog.askopenfilename(
            parent=root, title="Select valve calibration", filetypes=(("JSON", "*.json"),)
        )
        if value:
            calibration_var.set(value)

    def choose_evidence() -> None:
        value = filedialog.askopenfilename(
            parent=root,
            title="Select signed physical evidence",
            filetypes=(("JSON", "*.json"),),
        )
        if value:
            evidence_var.set(value)

    def choose_subjects() -> None:
        value = filedialog.askdirectory(parent=root, title="Select subject records folder")
        if value:
            subjects_var.set(value)

    field(0, "Operator", operator_var)
    field(1, "Valve calibration JSON", calibration_var, choose_calibration)
    field(2, "Physical evidence JSON", evidence_var, choose_evidence)
    field(3, "Hardware fail-safe revision", failsafe_var)
    field(4, "Maximum cumulative reward (µL)", cap_var)
    field(5, "Subject records folder", subjects_var, choose_subjects)
    ttk.Label(
        setup_tab,
        text=(
            "These operational values are saved beside the ignored rig certificate, not in the "
            "tracked experiment YAML. Weight records are append-only JSONL files."
        ),
        wraplength=760,
    ).grid(row=6, column=0, columnspan=3, sticky="w", padx=5, pady=(12, 5))

    def values() -> dict[str, Any]:
        return {
            "operator": operator_var.get(),
            "reward_calibration_path": calibration_var.get(),
            "physical_evidence_path": evidence_var.get(),
            "hardware_failsafe_revision": failsafe_var.get(),
            "max_cumulative_reward_ul": cap_var.get(),
            "subjects_dir": subjects_var.get(),
        }

    def save_fields(*, notify: bool = True) -> bool:
        try:
            path = save_operator_state(config, values())
            append_gui_audit(
                config,
                "operator_state_saved",
                operator=operator_var.get(),
                detail={"path": str(path)},
            )
        except Exception as exc:
            append_gui_audit(config, "operator_state_save_failed", detail=str(exc))
            messagebox.showerror("Cannot save operator data", str(exc), parent=root)
            return False
        if notify:
            messagebox.showinfo("Saved", f"Operator data saved to\n{path}", parent=root)
        return True

    setup_buttons = ttk.Frame(setup_tab)
    setup_buttons.grid(row=7, column=0, columnspan=3, sticky="w", padx=5, pady=10)
    ttk.Button(setup_buttons, text="Save operator data", command=save_fields).pack(side="left")

    def log_weight() -> None:
        if not save_fields(notify=False):
            return
        weight = simpledialog.askfloat(
            "Subject weight",
            f"Weight for {config.subject_id} (g)",
            minvalue=0.001,
            parent=root,
        )
        if weight is None:
            return
        target = simpledialog.askfloat(
            "Water target",
            "Daily water target (mL)",
            minvalue=0.0,
            parent=root,
        )
        if target is None:
            return
        notes = simpledialog.askstring("Notes", "Optional notes", parent=root)
        try:
            entry = append_weight_entry(
                config.welfare.subjects_dir,
                config.subject_id,
                weight_g=weight,
                water_target_ml=target,
                entry_date=date.today().isoformat(),
                notes=notes,
            )
            append_gui_audit(
                config,
                "subject_weight_logged",
                operator=operator_var.get(),
                detail={"subject_id": config.subject_id, "entry": entry},
            )
            messagebox.showinfo("Weight logged", f"Saved for {config.subject_id}", parent=root)
        except Exception as exc:
            append_gui_audit(config, "subject_weight_log_failed", detail=str(exc))
            messagebox.showerror("Cannot log weight", str(exc), parent=root)

    ttk.Button(setup_buttons, text="Log weight / water target", command=log_weight).pack(
        side="left", padx=(8, 0)
    )

    ttk.Label(
        actions_tab,
        text="Valve calibration actuates the pump. It remains disabled until you confirm the rig is clear.",
        wraplength=820,
    ).pack(anchor="w", pady=(0, 8))

    def calibrate_valve() -> None:
        if not save_fields(notify=False):
            return
        warnings = (
            "No animal is present.",
            "A tared collection vessel is installed.",
            "Emergency power control is available.",
        )
        if not all(
            messagebox.askyesno("Valve calibration safety", warning, parent=root)
            for warning in warnings
        ):
            append_gui_audit(config, "valve_calibration_cancelled", detail="safety confirmation")
            return
        output = certification_workspace(config) / "calibrations" / (
            f"valve_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )

        def gui_input(prompt: str) -> str:
            if prompt.startswith("Tare"):
                if not messagebox.askokcancel("Prepare collection", prompt, parent=root):
                    raise RuntimeError("Valve calibration cancelled by operator")
                return ""
            answer = simpledialog.askstring("Valve mass measurement", prompt, parent=root)
            if answer is None:
                raise RuntimeError("Valve calibration cancelled by operator")
            return answer

        append_gui_audit(
            config,
            "valve_calibration_started",
            operator=operator_var.get(),
            detail={"output": str(output)},
        )
        try:
            calibrated = run_guided_valve_calibration(
                config,
                durations_s=(0.02, 0.04, 0.06, 0.08, 0.10),
                repetitions=10,
                output=output,
                input_fn=gui_input,
            )
            calibration_var.set(str(output))
            save_fields(notify=False)
            append_gui_audit(
                config,
                "valve_calibration_completed",
                operator=operator_var.get(),
                detail=calibrated,
            )
            messagebox.showinfo("Calibration complete", f"Saved to\n{output}", parent=root)
        except Exception as exc:
            append_gui_audit(config, "valve_calibration_failed", detail=str(exc))
            messagebox.showerror("Valve calibration failed", str(exc), parent=root)

    ttk.Button(actions_tab, text="Run guided valve calibration…", command=calibrate_valve).pack(
        anchor="w", pady=(0, 12)
    )

    def edit_evidence() -> None:
        window = tk.Toplevel(root)
        window.title("Signed physical evidence JSON")
        window.geometry("850x650")
        text_widget = tk.Text(window, wrap="none", undo=True)
        ybar = ttk.Scrollbar(window, orient="vertical", command=text_widget.yview)
        xbar = ttk.Scrollbar(window, orient="horizontal", command=text_widget.xview)
        text_widget.configure(yscrollcommand=ybar.set, xscrollcommand=xbar.set)
        text_widget.grid(row=0, column=0, sticky="nsew")
        ybar.grid(row=0, column=1, sticky="ns")
        xbar.grid(row=1, column=0, sticky="ew")
        window.rowconfigure(0, weight=1)
        window.columnconfigure(0, weight=1)
        path = Path(evidence_var.get()).expanduser() if evidence_var.get().strip() else None
        if path is not None and path.is_file():
            initial = path.read_text(encoding="utf-8")
        else:
            example = Path(__file__).resolve().parents[2] / "docs" / "physical_evidence.example.json"
            initial = example.read_text(encoding="utf-8") if example.is_file() else "{}\n"
        text_widget.insert("1.0", initial)

        def save_evidence() -> None:
            try:
                payload = json.loads(text_widget.get("1.0", "end"))
                if not isinstance(payload, dict):
                    raise ValueError("Evidence must be one JSON object")
                if path is None:
                    target = certification_workspace(config) / "physical_evidence.json"
                else:
                    target = path.resolve()
                target = write_json_artifact(target, payload)
                evidence_var.set(str(target))
                save_fields(notify=False)
                append_gui_audit(
                    config,
                    "physical_evidence_saved",
                    operator=operator_var.get(),
                    detail={"path": str(target)},
                )
                window.destroy()
            except Exception as exc:
                messagebox.showerror("Invalid evidence", str(exc), parent=window)

        ttk.Button(window, text="Save evidence", command=save_evidence).grid(
            row=2, column=0, sticky="e", padx=8, pady=8
        )

    ttk.Button(actions_tab, text="Enter / edit physical evidence JSON…", command=edit_evidence).pack(
        anchor="w", pady=(0, 12)
    )
    ttk.Label(
        actions_tab,
        text=(
            "Before certification, confirm each live-hardware safety condition. The certification "
            "suite opens Bpod, StreamDIO, camera, Frame2TTL and cap-touch diagnostics."
        ),
        wraplength=820,
    ).pack(anchor="w", pady=(0, 6))
    labels = {
        "no_animal_present": "No animal is present",
        "collection_vessel_installed": "Collection vessel is installed",
        "force_mechanism_clear": "Force mechanism is clear",
        "speaker_exposure_safe": "Speaker exposure area is safe",
        "emergency_power_control_available": "Emergency power control is available",
    }
    for name, label in labels.items():
        ttk.Checkbutton(actions_tab, text=label, variable=safety_vars[name]).pack(anchor="w")

    certify_button: Any

    def certification_finished() -> None:
        process = running.get("process")
        if process is None:
            return
        returncode = process.poll()
        if returncode is None:
            root.after(500, certification_finished)
            return
        handle = running.pop("log_handle")
        handle.flush()
        fsync_best_effort(handle, path=running.get("log_path"))
        handle.close()
        running.clear()
        certify_button.configure(state="normal")
        append_gui_audit(
            config,
            "rig_certification_completed" if returncode == 0 else "rig_certification_failed",
            operator=operator_var.get(),
            detail={"returncode": returncode, "log": str(log_path)},
        )
        refresh_report()
        if returncode == 0:
            messagebox.showinfo("Certification complete", "Rig certification passed.", parent=root)
        else:
            messagebox.showerror(
                "Certification did not pass",
                f"Review failed checks and the log:\n{log_path}",
                parent=root,
            )

    log_path = certification_workspace(config) / "logs" / "certification.log"

    def run_certification() -> None:
        nonlocal log_path
        if running:
            return
        if not save_fields(notify=False):
            return
        confirmations = {name: variable.get() for name, variable in safety_vars.items()}
        try:
            command = certification_command(
                config,
                physical_evidence=evidence_var.get().strip() or None,
                confirmations=confirmations,
            )
        except Exception as exc:
            messagebox.showerror("Cannot start certification", str(exc), parent=root)
            return
        log_path = certification_workspace(config) / "logs" / (
            f"certification_{time.strftime('%Y%m%d_%H%M%S')}.log"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("w", encoding="utf-8", newline="\n")
        append_gui_audit(
            config,
            "rig_certification_started",
            operator=operator_var.get(),
            detail={"command": command, "log": str(log_path)},
        )
        try:
            process = subprocess.Popen(
                command,
                cwd=str(Path(__file__).resolve().parents[2]),
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=dict(os.environ),
            )
        except Exception as exc:
            handle.close()
            append_gui_audit(config, "rig_certification_launch_failed", detail=str(exc))
            messagebox.showerror("Cannot launch certification", str(exc), parent=root)
            return
        running.update(process=process, log_handle=handle, log_path=log_path)
        certify_button.configure(state="disabled")
        status_var.set("Certification is running; hardware diagnostics may take several minutes...")
        root.after(500, certification_finished)

    certify_button = ttk.Button(
        actions_tab, text="Run full rig certification", command=run_certification
    )
    certify_button.pack(anchor="w", pady=(10, 0))

    footer = ttk.Frame(root, padding=(12, 4, 12, 12))
    footer.pack(fill="x")
    ttk.Checkbutton(
        footer,
        text="I confirm this animal-use run and completed the safety/welfare checklist",
        variable=run_confirmed_var,
    ).pack(side="left")
    continue_button = ttk.Button(footer, text="Start session", state="disabled")
    continue_button.pack(side="right")

    def cancel_dialog() -> None:
        result["operator"] = operator_var.get()
        result["operator_confirmed"] = run_confirmed_var.get()
        root.destroy()

    ttk.Button(footer, text="Cancel", command=cancel_dialog).pack(side="right", padx=(0, 8))
    ttk.Button(footer, text="Recheck", command=lambda: refresh_report()).pack(
        side="right", padx=(0, 8)
    )

    current_report = report

    def render(current: dict[str, Any]) -> None:
        for item in tree.get_children():
            tree.delete(item)
        for check in current.get("checks", []):
            if check.get("status") == "PASS":
                continue
            detail = json.dumps(check.get("detail"), sort_keys=True, default=str)
            tree.insert("", "end", values=(check.get("status"), f"{check.get('name')}: {detail}"))
        passed = current.get("status") == "PASS"
        status_var.set(
            "All release checks pass; confirm this run to enable Start session."
            if passed
            else f"Blocked: {sum(c.get('status') != 'PASS' for c in current.get('checks', []))} check(s) failed."
        )
        continue_button.configure(
            state="normal" if passed and run_confirmed_var.get() else "disabled"
        )

    def refresh_report() -> bool:
        nonlocal current_report
        if not save_fields(notify=False):
            return False
        current_report = evaluate_animal_use_gate(
            config,
            operator_confirmed=run_confirmed_var.get(),
        )
        append_gui_audit(
            config,
            "animal_use_gate_rechecked",
            operator=operator_var.get(),
            detail={"status": current_report.get("status"), "checks": current_report.get("checks")},
        )
        render(current_report)
        return True

    def confirmation_changed(*_args: Any) -> None:
        render(current_report)

    def continue_session() -> None:
        if not refresh_report():
            return
        if current_report.get("status") != "PASS":
            messagebox.showerror("Still blocked", "One or more release checks failed.", parent=root)
            return
        result["continue"] = True
        result["operator"] = operator_var.get()
        result["operator_confirmed"] = run_confirmed_var.get()
        append_gui_audit(
            config,
            "animal_session_authorized",
            operator=operator_var.get(),
            detail={"subject_id": config.subject_id},
        )
        root.destroy()

    continue_button.configure(command=continue_session)
    run_confirmed_var.trace_add("write", confirmation_changed)
    root.protocol("WM_DELETE_WINDOW", cancel_dialog)
    append_gui_audit(
        config,
        "certification_dialog_opened",
        operator=operator_var.get(),
        detail={"initial_report": report, "state_error": state_error},
    )
    render(report)
    if state_error:
        messagebox.showwarning(
            "Operator state could not be loaded",
            state_error,
            parent=root,
        )
    root.mainloop()
    process = running.get("process")
    if process is not None and process.poll() is None:
        # Closing the window does not kill a safety diagnostic mid-actuation.
        process.wait()
    handle = running.get("log_handle")
    if handle is not None and not handle.closed:
        handle.flush()
        fsync_best_effort(handle, path=running.get("log_path"))
        handle.close()
    if not result["continue"]:
        append_gui_audit(
            config,
            "certification_dialog_cancelled",
            operator=str(result["operator"]),
        )
    return CertificationDialogResult(
        continue_session=bool(result["continue"]),
        operator_confirmed=bool(result["continue"] and result["operator_confirmed"]),
    )
