# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Boon-How Chew

"""Run analysis code in isolation and capture what actually happened.

The code contract (what the Analysis Agent must write):

    def analyse(data_path: str, seed: int) -> dict:
        ...                      # read the CSV at data_path, return JSON-safe results

Two layers of protection:
  1. `check_code` rejects code that imports anything outside the statistics
     stack, touches files other than through pandas/numpy readers, or reaches
     for dunder attributes, eval/exec, and similar. This keeps analyses
     deterministic and reviewable.
  2. The Modal sandbox is the real boundary: no network, its own filesystem,
     a time limit, and it is destroyed after each run.

`LocalExecutor` exists for tests and offline development only; it is NOT a
security boundary.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

ALLOWED_IMPORTS = {
    "numpy", "pandas", "scipy", "statsmodels", "sklearn", "patsy",
    "math", "statistics", "json", "warnings", "itertools", "collections", "functools", "typing", "dataclasses",
}
BANNED_CALLS = {"eval", "exec", "compile", "__import__", "open", "input", "breakpoint", "globals", "locals",
                "vars", "getattr", "setattr", "delattr"}
BANNED_ATTRS = {"to_csv", "to_excel", "to_pickle", "to_parquet", "to_sql", "to_hdf", "to_feather", "savefig",
                "system", "popen", "remove", "unlink", "rmdir", "read_pickle", "read_sql", "load"}
WORK = "/work"
RUNNER = r'''
import hashlib, json, os, platform, random, sys, time, traceback, warnings
from importlib import metadata
work = os.environ.get("SAKSI_WORK", "/work")
seed = int(os.environ["SAKSI_SEED"])
data_path = os.path.join(work, "input", "data.csv")
out = os.path.join(work, "output")
os.makedirs(out, exist_ok=True)
random.seed(seed)
try:
    import numpy as np
    np.random.seed(seed)
except Exception:
    pass
def ver(p):
    try:
        return metadata.version(p)
    except Exception:
        return None
meta = {
    "data_sha256": hashlib.sha256(open(data_path, "rb").read()).hexdigest(),
    "code_sha256": hashlib.sha256(open(os.path.join(work, "analysis.py"), "rb").read()).hexdigest(),
    "seed": seed, "python": sys.version.split()[0], "platform": platform.platform(),
    "packages": {p: ver(p) for p in ("numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "patsy")},
    "container_image_id": os.environ.get("MODAL_IMAGE_ID"), "sandbox_id": os.environ.get("MODAL_SANDBOX_ID"),
}
t0 = time.time()
status, error, results, caught = "ok", None, None, []
try:
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ns = {"__name__": "saksi_analysis"}
        exec(compile(open(os.path.join(work, "analysis.py")).read(), "analysis.py", "exec"), ns)
        results = ns["analyse"](data_path, seed)
        caught = sorted({f"{x.category.__name__}: {str(x.message)[:200]}" for x in w})[:20]
    json.dumps(results)
except Exception:
    status, error = "error", traceback.format_exc()[-3000:]
    results = None
meta["seconds"] = round(time.time() - t0, 3)
meta["warnings"] = caught
json.dump({"status": status, "error": error, "results": results, "meta": meta},
          open(os.path.join(out, "result.json"), "w"), default=str)
'''


class CodeRejected(ValueError):
    pass


def check_code(code: str) -> None:
    """Raise CodeRejected with every problem found; return None if acceptable."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise CodeRejected(f"syntax error: {e}") from e
    problems: list[str] = []
    has_analyse = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"import of {a.name!r} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                problems.append(f"import from {node.module!r} is not allowed")
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id in BANNED_CALLS:
                problems.append(f"call to {f.id}() is not allowed")
            if isinstance(f, ast.Attribute) and f.attr in BANNED_ATTRS:
                problems.append(f"call to .{f.attr}() is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            problems.append(f"access to {node.attr} is not allowed")
        elif isinstance(node, ast.Name) and node.id.startswith("__") and node.id != "__name__":
            problems.append(f"use of {node.id} is not allowed")
        elif isinstance(node, ast.FunctionDef) and node.name == "analyse" and node in tree.body:
            args = [a.arg for a in node.args.args]
            has_analyse = args[:2] == ["data_path", "seed"]
    if not has_analyse:
        problems.append("code must define analyse(data_path, seed) at top level")
    if problems:
        raise CodeRejected("; ".join(sorted(set(problems))))


@dataclass
class ExecResult:
    backend: str
    status: str                      # ok | error | timeout | infrastructure_error
    results: Any
    error: str | None
    meta: dict[str, Any] = field(default_factory=dict)
    exit_code: int | None = None
    stdout_tail: str = ""
    stderr_tail: str = ""
    wall_seconds: float | None = None

    def to_payload(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("results")
        return d


class Executor(Protocol):
    def run(self, code: str, data: bytes, seed: int, timeout: int = 300) -> ExecResult: ...


def _parse(raw: str | None, backend: str, exit_code, stdout, stderr, wall) -> ExecResult:
    if not raw:
        return ExecResult(backend, "error", None, "analysis produced no result file", {}, exit_code,
                          stdout[-2000:], stderr[-2000:], wall)
    r = json.loads(raw)
    return ExecResult(backend, r["status"], r["results"], r["error"], r["meta"], exit_code,
                      stdout[-2000:], stderr[-2000:], wall)


class LocalExecutor:
    """Subprocess runner for tests and offline work. Not a security boundary."""

    backend = "local-subprocess"

    def run(self, code: str, data: bytes, seed: int, timeout: int = 300) -> ExecResult:
        with tempfile.TemporaryDirectory() as d:
            work = Path(d)
            (work / "input").mkdir()
            (work / "input" / "data.csv").write_bytes(data)
            (work / "analysis.py").write_text(code)
            (work / "runner.py").write_text(RUNNER)
            env = {"SAKSI_WORK": str(work), "SAKSI_SEED": str(seed), "PATH": os.environ.get("PATH", ""),
                   "PYTHONHASHSEED": str(seed)}
            t0 = time.monotonic()
            try:
                p = subprocess.run([sys.executable, "-I", str(work / "runner.py")], cwd=work, env=env,
                                   capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                return ExecResult(self.backend, "timeout", None, f"exceeded {timeout}s", {}, None, "", "",
                                  round(time.monotonic() - t0, 3))
            out = work / "output" / "result.json"
            return _parse(out.read_text() if out.exists() else None, self.backend, p.returncode,
                          p.stdout, p.stderr, round(time.monotonic() - t0, 3))


# Exact versions the analyses run with; recorded again from inside the sandbox.
SANDBOX_PACKAGES = ("numpy==2.5.4", "pandas==3.1.0", "scipy==1.18.1", "statsmodels==0.15.0",
                    "scikit-learn==1.9.1", "patsy==1.0.3")


class ModalExecutor:
    """Runs each analysis in a fresh Modal Sandbox with networking blocked."""

    backend = "modal-sandbox"

    def __init__(self, app_name: str = "saksi-sandboxes", cpu: float = 1.0, memory_mb: int = 2048):
        self.app_name, self.cpu, self.memory_mb = app_name, cpu, memory_mb
        self._image = None

    def _image_obj(self):
        import modal
        if self._image is None:
            self._image = modal.Image.debian_slim(python_version="3.12").pip_install(*SANDBOX_PACKAGES)
        return self._image

    def run(self, code: str, data: bytes, seed: int, timeout: int = 300) -> ExecResult:
        import modal
        t0 = time.monotonic()
        sb = None
        try:
            app = modal.App.lookup(self.app_name, create_if_missing=True)
            sb = modal.Sandbox.create(app=app, image=self._image_obj(), block_network=True,
                                      timeout=timeout + 60, cpu=self.cpu, memory=self.memory_mb)
            sb.filesystem.write_bytes(data, f"{WORK}/input/data.csv")
            sb.filesystem.write_text(code, f"{WORK}/analysis.py")
            sb.filesystem.write_text(RUNNER, f"{WORK}/runner.py")
            p = sb.exec("python", "-I", f"{WORK}/runner.py", timeout=timeout,
                        env={"SAKSI_WORK": WORK, "SAKSI_SEED": str(seed), "PYTHONHASHSEED": str(seed),
                             "MODAL_SANDBOX_ID": sb.object_id})
            p.wait()
            stdout, stderr = p.stdout.read(), p.stderr.read()
            try:
                raw = sb.filesystem.read_text(f"{WORK}/output/result.json")
            except Exception:
                raw = None
            res = _parse(raw, self.backend, p.returncode, stdout, stderr, round(time.monotonic() - t0, 3))
            res.meta["sandbox_object_id"] = sb.object_id
            res.meta["network"] = "blocked"
            return res
        except Exception as e:
            return ExecResult(self.backend, "infrastructure_error", None, f"{type(e).__name__}: {e}"[:1000], {},
                              None, "", "", round(time.monotonic() - t0, 3))
        finally:
            if sb is not None:
                try:
                    sb.terminate()
                except Exception:
                    pass
