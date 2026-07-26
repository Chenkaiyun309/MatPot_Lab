#!/usr/bin/env python3
"""Dependency-light web UI for the CHGNet calculator workflow."""

from __future__ import annotations

import csv
import json
import mimetypes
import os
import subprocess
import sys
import threading
import time
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
MODULES_DIR = PROJECT_DIR / "modules"
RUNS_DIR = PROJECT_DIR / "calculator_runs"
MODELS_DIR = PROJECT_DIR / "models"
TEMPLATE_DIR = CODE_DIR / "templates"
STATIC_DIR = CODE_DIR / "static"
LOGO_DIR = PROJECT_DIR / "logo"

DEFAULTS = {
    "generated_dir": "calculator_runs/generated_structures",
    "relaxed_dir": "calculator_runs/relaxed_results",
    "phase_dir": "calculator_runs/phase_stability_results",
    "mechanical_dir": "calculator_runs/mechanical_results",
    "stress_dir": "calculator_runs/stress_strain_results",
    "md_dir": "calculator_runs/md_results",
}

OUTPUT_KINDS = {
    "calculator_runs/generated_structures": "generate",
    "calculator_runs/relaxed_results": "relax",
    "calculator_runs/phase_stability_results": "phase",
    "calculator_runs/mechanical_results": "mechanical",
    "calculator_runs/stress_strain_results": "stress",
    "calculator_runs/md_results": "md",
}

RUNS_DIR.mkdir(exist_ok=True)


def to_project_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return path.resolve()


def display_path(path: str | Path) -> str:
    resolved = to_project_path(path)
    try:
        return str(resolved.relative_to(PROJECT_DIR))
    except ValueError:
        return str(resolved)


def output_kind(path: Path) -> str:
    relative = display_path(path)
    for prefix, kind in OUTPUT_KINDS.items():
        if relative == prefix or relative.startswith(f"{prefix}/"):
            return kind
    if relative.startswith("calculator_runs/downloads/"):
        name = path.name.lower()
        if "md" in name or "molecular" in name:
            return "md"
        if "mechanical" in name:
            return "mechanical"
        if "stressstrain" in name or "stress_strain" in name:
            return "stress"
        if "phase" in name:
            return "phase"
        if "relax" in name:
            return "relax"
        return "relax"
    if relative.startswith("calculator_runs/") and "generate" in path.name.lower():
        return "generate"
    return "other"


def safe_project_file(path_value: str) -> Path:
    resolved = to_project_path(path_value)
    root = PROJECT_DIR.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError("Path is outside this project.")
    return resolved


def parse_bool(value: Any) -> bool:
    return value is True or str(value).lower() in {"1", "true", "yes", "on"}


def append_flag(command: list[str], flag: str, value: Any) -> None:
    if value is None or value == "":
        return
    command.extend([flag, str(value)])


def append_switch(command: list[str], flag: str, enabled: Any) -> None:
    if parse_bool(enabled):
        command.append(flag)


def build_generation_command(data: dict[str, Any]) -> list[str]:
    """Build the generate CLI directly from form data — no temp input file needed."""
    python = sys.executable
    command = [python, str(MODULES_DIR / "generate_hea_structures.py")]
    elements = str(data.get("elements", "")).strip()
    num_elements = str(data.get("num_elements", "")).strip()
    if elements:
        append_flag(command, "--elements", elements)
        append_flag(command, "--ratios", str(data.get("ratios", "1 1 1 1 1")).strip())
    elif num_elements:
        append_flag(command, "--num-elements", num_elements)
        append_flag(command, "--candidate-elements", str(data.get("candidate_elements", "Ti V Cr Mn Fe Co Ni Zr Nb Mo Ta W")).strip())
        append_flag(command, "--ratios", str(data.get("ratios", "1 1 1 1 1")).strip())
    else:
        # Fall back to the built-in default composition.
        append_flag(command, "--elements", "Hf Nb Ta Ti Zr")
        append_flag(command, "--ratios", "1 1 1 1 1")
    append_flag(command, "--crystal-structure", data.get("crystal_structure", "bcc"))
    append_flag(command, "--lattice-constant", data.get("lattice_constant", 3.0))
    append_flag(command, "--hcp-c-over-a", data.get("hcp_c_over_a", 1.633))
    command.extend(["--supercell", *normalized_supercell(data.get("supercell"), "4 4 4")])
    append_flag(command, "--seed", data.get("seed", 42))
    append_flag(command, "--num-structures", data.get("num_structures", 1))
    append_flag(command, "--output-dir", data.get("output_dir", DEFAULTS["generated_dir"]))
    return command


def normalized_supercell(value: Any, fallback: str) -> list[str]:
    parts = str(value or fallback).replace(",", " ").split()
    if len(parts) != 3:
        parts = fallback.split()
    return parts[:3]


def build_command(kind: str, data: dict[str, Any]) -> tuple[list[str], str]:
    python = sys.executable
    if kind == "generate":
        return build_generation_command(data), DEFAULTS["generated_dir"]

    if kind == "relax":
        output_dir = data.get("output_dir", DEFAULTS["relaxed_dir"])
        command = [python, str(MODULES_DIR / "relax_structures.py"), "--input-dir", str(data.get("input_dir", DEFAULTS["generated_dir"])), "--output-dir", str(output_dir)]
        for flag, key, default in [("--input-format", "input_format", "auto"), ("--ediffg", "ediffg", -0.005), ("--nsw", "nsw", 200), ("--isif", "isif", 3), ("--loginterval", "loginterval", 1)]:
            append_flag(command, flag, data.get(key, default))
        append_switch(command, "--recursive", data.get("recursive"))
        append_switch(command, "--verbose", data.get("verbose"))
        append_flag(command, "--model", data.get("model", ""))
        return command, str(output_dir)

    if kind == "phase":
        output_dir = data.get("output_dir", DEFAULTS["phase_dir"])
        phases = ",".join(data.get("phases") or ["bcc", "fcc", "hcp"])
        command = [python, str(MODULES_DIR / "phase_stability.py"), "--mode", data.get("mode", "fixed"), "--output-dir", str(output_dir), "--phases", phases]
        append_flag(command, "--task-name", data.get("task_name", ""))
        append_flag(command, "--samples-per-phase", data.get("samples_per_phase", 2))
        append_flag(command, "--seed", data.get("seed", 42))
        command.extend(["--supercell", *normalized_supercell(data.get("supercell"), "3 3 3")])
        for flag, key, default in [
            ("--lattice-bcc", "lattice_bcc", 3.0),
            ("--lattice-fcc", "lattice_fcc", 3.6),
            ("--lattice-hcp", "lattice_hcp", 3.0),
            ("--hcp-c-over-a", "hcp_c_over_a", 1.633),
            ("--ediffg", "ediffg", -0.005),
            ("--nsw", "nsw", 200),
            ("--isif", "isif", 3),
            ("--loginterval", "loginterval", 1),
        ]:
            append_flag(command, flag, data.get(key, default))
        append_switch(command, "--no-relax", data.get("no_relax"))
        if data.get("mode", "fixed") == "binary":
            for flag, key, default in [("--element-a", "element_a", "Ti"), ("--element-b", "element_b", "Nb"), ("--x-start", "x_start", 0.1), ("--x-end", "x_end", 0.9), ("--num-points", "num_points", 9)]:
                append_flag(command, flag, data.get(key, default))
        else:
            append_flag(command, "--elements", data.get("elements", "Hf Nb Ta Ti Zr"))
            append_flag(command, "--ratios", data.get("ratios", "1 1 1 1 1"))
        append_flag(command, "--model", data.get("model", ""))
        return command, str(output_dir)

    if kind in {"mechanical", "stress"}:
        script_name = "mechanical_properties.py" if kind == "mechanical" else "stress_strain_curve.py"
        output_dir = data.get("output_dir", DEFAULTS["mechanical_dir" if kind == "mechanical" else "stress_dir"])
        command = [python, str(MODULES_DIR / script_name), "--input-dir", str(data.get("input_dir", DEFAULTS["relaxed_dir"])), "--output-dir", str(output_dir)]
        append_flag(command, "--input-format", data.get("input_format", "auto"))
        if kind == "mechanical":
            for flag, key, default in [("--strain-max", "strain_max", 0.006), ("--num-points", "num_points", 5)]:
                append_flag(command, flag, data.get(key, default))
        else:
            for flag, key, default in [("--direction", "direction", "xx"), ("--strain-start", "strain_start", -0.02), ("--strain-end", "strain_end", 0.02), ("--num-points", "num_points", 9)]:
                append_flag(command, flag, data.get(key, default))
        for flag, key, default in [("--fmax", "fmax", 0.03), ("--max-steps", "max_steps", 80)]:
            append_flag(command, flag, data.get(key, default))
        append_switch(command, "--recursive", data.get("recursive"))
        append_switch(command, "--relax-atoms", data.get("relax_atoms"))
        if not parse_bool(data.get("write_structures", True)):
            command.append("--no-write-structures")
        if not parse_bool(data.get("plot", True)):
            command.append("--no-plot")
        append_flag(command, "--model", data.get("model", ""))
        return command, str(output_dir)

    if kind == "md":
        output_dir = data.get("output_dir", DEFAULTS["md_dir"])
        command = [python, str(MODULES_DIR / "run_molecular_dynamics.py"), "--input-dir", str(data.get("input_dir", DEFAULTS["relaxed_dir"])), "--output-dir", str(output_dir)]
        for flag, key, default in [
            ("--input-format", "input_format", "auto"),
            ("--ensemble", "ensemble", "nvt"),
            ("--thermostat", "thermostat", "Nose-Hoover"),
            ("--temperature", "temperature", 1000),
            ("--timestep", "timestep", 2.0),
            ("--pressure", "pressure", 1.0),
            ("--steps", "steps", 1000),
            ("--loginterval", "loginterval", 10),
            ("--use-device", "use_device", ""),
        ]:
            append_flag(command, flag, data.get(key, default))
        append_switch(command, "--recursive", data.get("recursive"))
        append_flag(command, "--model", data.get("model", ""))
        return command, str(output_dir)

    raise ValueError(f"Unknown task kind: {kind}")


class TaskManager:
    def __init__(self) -> None:
        self.tasks: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()

    def start(self, kind: str, command: list[str], output_dir: str) -> str:
        task_id = f"{kind}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with self.lock:
            self.tasks[task_id] = {
                "id": task_id,
                "kind": kind,
                "status": "running",
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "finished_at": "",
                "logs": [f"$ {' '.join(command)}"],
                "returncode": None,
                "output_dir": display_path(output_dir),
                "process": None,
            }
        threading.Thread(target=self._run, args=(task_id, command), daemon=True).start()
        return task_id

    def _run(self, task_id: str, command: list[str]) -> None:
        try:
            process = subprocess.Popen(command, cwd=str(PROJECT_DIR), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            with self.lock:
                self.tasks[task_id]["process"] = process
            assert process.stdout is not None
            for line in process.stdout:
                self.add_log(task_id, line.rstrip())
            returncode = process.wait()
            with self.lock:
                task = self.tasks[task_id]
                if task["status"] != "stopped":
                    task["status"] = "completed" if returncode == 0 else "failed"
                task["returncode"] = returncode
                task["finished_at"] = datetime.now().isoformat(timespec="seconds")
                task["process"] = None
        except Exception as exc:
            self.add_log(task_id, f"[ERROR] {exc}")
            with self.lock:
                self.tasks[task_id]["status"] = "failed"
                self.tasks[task_id]["finished_at"] = datetime.now().isoformat(timespec="seconds")
                self.tasks[task_id]["process"] = None

    def add_log(self, task_id: str, message: str) -> None:
        with self.lock:
            if task_id in self.tasks:
                self.tasks[task_id]["logs"].append(message)

    def snapshot(self, task_id: str) -> dict[str, Any] | None:
        with self.lock:
            task = self.tasks.get(task_id)
            if not task:
                return None
            public = {key: value for key, value in task.items() if key != "process"}
            return json.loads(json.dumps(public))

    def latest(self) -> dict[str, Any] | None:
        with self.lock:
            if not self.tasks:
                return None
            task_id = list(self.tasks.keys())[-1]
        return self.snapshot(task_id)

    def stop(self, task_id: str) -> bool:
        with self.lock:
            task = self.tasks.get(task_id)
            process = task.get("process") if task else None
        if not process or process.poll() is not None:
            return False
        process.terminate()
        time.sleep(0.5)
        if process.poll() is None:
            process.kill()
        self.add_log(task_id, "[STOPPED] Task was terminated by user.")
        with self.lock:
            self.tasks[task_id]["status"] = "stopped"
            self.tasks[task_id]["finished_at"] = datetime.now().isoformat(timespec="seconds")
            self.tasks[task_id]["process"] = None
        return True


task_manager = TaskManager()


def render_index() -> bytes:
    base = (TEMPLATE_DIR / "base.html").read_text(encoding="utf-8")
    content = (TEMPLATE_DIR / "index.html").read_text(encoding="utf-8")
    content = content.replace('{% extends "base.html" %}', "").replace("{% block content %}", "").replace("{% endblock %}", "")
    for key, value in DEFAULTS.items():
        content = content.replace("{{ defaults." + key + " }}", value)
    html = base.replace("{% block content %}{% endblock %}", content)
    html = html.replace("{{ defaults | tojson | safe }}", json.dumps(DEFAULTS))
    version = str(int(time.time()))
    html = html.replace("{{ url_for('static', filename='css/style.css', v=static_version) }}", f"/static/css/style.css?v={version}")
    html = html.replace("{{ url_for('static', filename='js/main.js', v=static_version) }}", f"/static/js/main.js?v={version}")
    return html.encode("utf-8")


def list_files_payload() -> dict[str, Any]:
    files_payload: list[dict[str, Any]] = []
    for path in sorted(RUNS_DIR.rglob("*"), key=lambda item: item.stat().st_mtime if item.exists() else 0, reverse=True):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.suffix.lower() not in {".csv", ".txt", ".log", ".json", ".cif", ".vasp", ".poscar", ".png", ".xyz", ".traj", ".zip"} and "POSCAR" not in path.name:
            continue
        files_payload.append(
            {
                "path": display_path(path),
                "name": path.name,
                "suffix": path.suffix.lower() or "poscar",
                "size": path.stat().st_size,
                "mtime": datetime.fromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds"),
                "kind": output_kind(path),
            }
        )
        if len(files_payload) >= 80:
            break
    return {"files": files_payload, "defaults": DEFAULTS}


def list_models_payload() -> dict[str, Any]:
    """List selectable CHGNet models from models/registry.json + auto-discovery.

    Returns models in registry order, then any extra .pth.tar files found in
    models/ that are not listed. Each entry carries the value sent to --model.
    """
    models: list[dict[str, Any]] = []
    seen_files: set[str] = set()
    registry_path = MODELS_DIR / "registry.json"
    if registry_path.is_file():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except Exception:
            registry = []
        for entry in registry:
            if not isinstance(entry, dict) or not entry.get("id"):
                continue
            file_name = entry.get("file", "")
            if file_name:
                seen_files.add(file_name)
            models.append({
                "id": str(entry["id"]),
                "label": str(entry.get("label", entry["id"])),
                "file": file_name,
                "default": bool(entry.get("default", False)),
                "builtin": bool(entry.get("builtin", False)),
            })
    # Auto-discover any .pth.tar in models/ not already in the registry.
    if MODELS_DIR.is_dir():
        for path in sorted(MODELS_DIR.glob("*.pth.tar")):
            if path.name in seen_files:
                continue
            models.append({
                "id": path.name,
                "label": f"Custom: {path.name}",
                "file": path.name,
                "default": False,
                "builtin": False,
            })
    if not any(m.get("default") for m in models):
        for m in models:
            if m["id"] == "0.3.0":
                m["default"] = True
                break
    return {"models": models}


def preview_payload(path_value: str) -> tuple[int, dict[str, Any]]:
    try:
        path = safe_project_file(path_value)
    except ValueError as exc:
        return 400, {"error": str(exc)}
    if not path.exists() or not path.is_file():
        return 404, {"error": "File not found"}
    try:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            with path.open("r", encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                rows = [row for index, row in enumerate(reader) if index < 80]
                return 200, {"type": "csv", "columns": reader.fieldnames or [], "rows": rows}
        if suffix == ".json":
            return 200, {"type": "text", "content": json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2, ensure_ascii=False)}
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".svg"}:
            return 200, {"type": "image", "path": display_path(path)}
        if suffix in {".txt", ".log", ".cif", ".vasp", ".poscar", ".xyz"} or "POSCAR" in path.name:
            return 200, {"type": "text", "content": path.read_text(encoding="utf-8", errors="replace")[:200000]}
        return 200, {"type": "binary", "path": display_path(path)}
    except Exception as exc:
        return 500, {"error": f"Preview failed: {exc}"}


def select_directory_payload(data: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    if sys.platform != "darwin":
        return 400, {"error": "Directory picker is currently available on macOS only."}
    current = to_project_path(data.get("current", PROJECT_DIR))
    if current.is_file():
        current = current.parent
    if not current.exists():
        current = PROJECT_DIR
    escaped = str(current).replace("\\", "\\\\").replace('"', '\\"')
    script = f'set selectedFolder to choose folder with prompt "Choose directory" default location POSIX file "{escaped}"\nreturn POSIX path of selectedFolder'
    result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=120, check=False)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        if "User canceled" in stderr or result.returncode == 1:
            return 200, {"cancelled": True}
        return 500, {"error": stderr or "Directory selection failed"}
    return 200, {"path": display_path(result.stdout.strip().rstrip("/"))}


class CHGNetHandler(BaseHTTPRequestHandler):
    server_version = "CHGNetWeb/1.0"

    def send_bytes(self, body: bytes, content_type: str = "text/html; charset=utf-8", status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        self.send_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        return json.loads(raw.decode("utf-8") or "{}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/":
            self.send_bytes(render_index())
            return
        if path.startswith("/static/"):
            target = (STATIC_DIR / path.removeprefix("/static/")).resolve()
            if STATIC_DIR.resolve() not in target.parents or not target.is_file():
                self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            self.send_bytes(target.read_bytes(), content_type)
            return
        if path.startswith("/logo/"):
            target = (LOGO_DIR / path.removeprefix("/logo/")).resolve()
            if LOGO_DIR.resolve() not in target.parents or not target.is_file():
                self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(str(target))[0] or "image/svg+xml"
            self.send_bytes(target.read_bytes(), content_type)
            return
        if path == "/api/files":
            self.send_json(list_files_payload())
            return
        if path == "/api/models":
            self.send_json(list_models_payload())
            return
        if path == "/api/task/latest":
            self.send_json(task_manager.latest() or {})
            return
        if path.startswith("/api/task/"):
            task_id = path.removeprefix("/api/task/")
            task = task_manager.snapshot(task_id)
            self.send_json(task or {"error": "Task not found"}, 200 if task else 404)
            return
        if path == "/api/preview":
            query = parse_qs(parsed.query)
            status, payload = preview_payload(query.get("path", [""])[0])
            self.send_json(payload, status)
            return
        if path == "/api/raw":
            query = parse_qs(parsed.query)
            try:
                target = safe_project_file(query.get("path", [""])[0])
            except ValueError as exc:
                self.send_json({"error": str(exc)}, 400)
                return
            if not target.is_file():
                self.send_json({"error": "File not found"}, 404)
                return
            content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
            self.send_bytes(target.read_bytes(), content_type)
            return
        self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            data = self.read_json()
        except Exception as exc:
            self.send_json({"error": f"Invalid JSON: {exc}"}, 400)
            return
        if path.startswith("/api/run/"):
            kind = path.removeprefix("/api/run/")
            try:
                command, output_dir = build_command(kind, data)
                task_id = task_manager.start(kind, command, output_dir)
            except Exception as exc:
                self.send_json({"error": str(exc)}, 400)
                return
            self.send_json({"task_id": task_id})
            return
        if path.startswith("/api/task/") and path.endswith("/stop"):
            task_id = path.removeprefix("/api/task/").removesuffix("/stop")
            self.send_json({"stopped": task_manager.stop(task_id)})
            return
        if path == "/api/select_directory":
            status, payload = select_directory_payload(data)
            self.send_json(payload, status)
            return
        if path == "/api/open_file":
            try:
                target = safe_project_file(str(data.get("path", "")))
            except ValueError as exc:
                self.send_json({"error": str(exc)}, 400)
                return
            if not target.is_file():
                self.send_json({"error": "File not found"}, 404)
                return
            opener = ["open", str(target)] if sys.platform == "darwin" else ["xdg-open", str(target)]
            try:
                result = subprocess.run(opener, capture_output=True, text=True, timeout=8, check=False)
            except subprocess.TimeoutExpired:
                self.send_json({"error": "Opening the file timed out."}, 504)
                return
            except Exception as exc:
                self.send_json({"error": f"Failed to open file: {exc}"}, 500)
                return
            if result.returncode != 0:
                message = (result.stderr or result.stdout or "System open command failed.").strip()
                self.send_json({"error": message}, 500)
                return
            self.send_json({"opened": display_path(target), "absolute_path": str(target)})
            return
        self.send_json({"error": "Not found"}, HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:
        sys.stderr.write("[%s] %s\n" % (datetime.now().strftime("%H:%M:%S"), format % args))


def check_runtime() -> int:
    for script_name in ["generate_hea_structures.py", "relax_structures.py", "phase_stability.py", "mechanical_properties.py", "stress_strain_curve.py", "run_molecular_dynamics.py"]:
        if not (MODULES_DIR / script_name).is_file():
            print(f"Missing script: {script_name}")
            return 1
    print("Web app check passed.")
    print(f"Project: {PROJECT_DIR}")
    print("No Flask dependency is required.")
    return 0


def main() -> None:
    if "--check" in sys.argv:
        raise SystemExit(check_runtime())
    port = int(os.getenv("PORT", "5055"))
    ThreadingHTTPServer.allow_reuse_address = True
    try:
        server = ThreadingHTTPServer(("127.0.0.1", port), CHGNetHandler)
    except OSError as exc:
        if exc.errno == 48:  # Address already in use
            print(f"Port {port} is already in use. Find and stop the process holding it:")
            print(f"  lsof -ti:{port} | xargs kill")
            print(f"Or start on a different port: PORT={port + 1} python web_code/web_app.py")
            raise SystemExit(1)
        raise
    url = f"http://127.0.0.1:{port}"
    print(f"CHGNet web UI: {url}")
    print("Press Ctrl+C to stop.")
    if not parse_bool(os.getenv("NO_OPEN_BROWSER")):
        threading.Thread(target=lambda: (time.sleep(1.0), webbrowser.open(url)), daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
