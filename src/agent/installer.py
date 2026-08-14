# Copyright (c) 2026 Mike Koci-Eliseart Company. All rights reserved.
#
"""Environment setup: venv + pip install for the text (LLM) and image toolchains.

Called by the ``ppte install txt|img`` CLI. Works standalone (system Python) or
from within an existing venv (npm wrapper). If no Python exists on the system,
``install_python()`` attempts an OS-native install (apt/brew/winget) — the npm
wrapper does this before invoking Python at all.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

TXT_BASE_DEPS = [
    "pyyaml>=6.0",
    "requests>=2.31",
    "beautifulsoup4>=4.12",
    "lxml>=5.0",
    "duckduckgo_search>=5.0",
]
TXT_ML_DEPS = [
    "torch>=2.1",
    "transformers>=4.40",
    "trl>=0.8",
    "peft>=0.10",
    "datasets>=2.19",
    "accelerate>=0.29",
]
IMG_DEPS = [
    "diffusers>=0.28",
    "peft>=0.10",
    "safetensors>=0.4",
    "accelerate>=0.29",
    "torchvision>=0.16",
]


def _run(cmd: list[str], dry_run: bool = False) -> int:
    if dry_run:
        print("[install] (dry-run) " + " ".join(cmd))
        return 0
    print("[install] " + " ".join(cmd))
    return subprocess.call(cmd)


def in_venv() -> bool:
    return sys.prefix != sys.base_prefix


def ensure_venv(dry_run: bool = False) -> str:
    """Return the venv python; create one (./.venv) if running on system python."""
    if in_venv():
        return sys.executable
    venv_dir = os.environ.get("PPTE_VENV") or "venv"
    py = Path(venv_dir) / ("Scripts/python.exe" if platform.system() == "Windows" else "bin/python")
    if not py.exists():
        ok = _run([sys.executable, "-m", "venv", venv_dir], dry_run=dry_run) == 0
        if not ok and not dry_run:
            # Fallback for systems missing the python3.x-venv package (no ensurepip):
            # create a pip-less venv and bootstrap pip via get-pip.py.
            ok = _run([sys.executable, "-m", "venv", "--without-pip", venv_dir]) == 0
            if ok:
                import urllib.request

                get_pip = Path(venv_dir) / "get-pip.py"
                urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", get_pip)
                ok = _run([str(py), str(get_pip), "--quiet"]) == 0
        if not ok and not dry_run:
            raise RuntimeError(
                "could not create a Python venv; install the 'python3-venv' "
                "package (Debian/Ubuntu) and retry"
            )
    return str(py)


def install_python() -> bool:
    """Attempt to install Python 3 via the OS package manager. Best effort."""
    if shutil.which("python3") or shutil.which("python"):
        print("[install] python already available.")
        return True
    system = platform.system().lower()
    pm = shutil.which("apt-get") if system == "linux" else shutil.which("brew") if system == "darwin" else shutil.which("winget")
    if not pm:
        return False
    if system == "linux":
        return _run(["sudo", pm, "install", "-y", "python3", "python3-venv", "python3-pip"]) == 0
    if system == "darwin":
        return _run([pm, "install", "python@3.12"]) == 0
    if system == "windows":
        return _run([pm, "install", "-e", "--id", "Python.Python.3.12"]) == 0
    return False


def _pip_install(python: str, deps: list[str], dry_run: bool = False) -> int:
    return _run([python, "-m", "pip", "install", "--upgrade", "pip", "--quiet"], dry_run) or _run(
        [python, "-m", "pip", "install", "--quiet", *deps], dry_run
    )


def install_torch(dry_run: bool = False) -> int:
    py = ensure_venv(dry_run=dry_run)
    if not dry_run and py != sys.executable:
        print(f"[install] using venv python: {py}")
    if _pip_install(py, TXT_ML_DEPS, dry_run=dry_run) != 0:
        return 1
    print("[install] ML toolchain ready (torch, transformers, trl, ...).")
    return 0


def install_txt(dry_run: bool = False) -> int:
    py = ensure_venv(dry_run=dry_run)
    if not dry_run and py != sys.executable:
        print(f"[install] using venv python: {py}")
    if _pip_install(py, [*TXT_BASE_DEPS, *TXT_ML_DEPS], dry_run=dry_run) != 0:
        return 1
    print("[install] text/LLM toolchain ready.")
    return 0


def install_img(dry_run: bool = False) -> int:
    py = ensure_venv(dry_run=dry_run)
    if not dry_run and py != sys.executable:
        print(f"[install] using venv python: {py}")
    if _pip_install(py, IMG_DEPS, dry_run=dry_run) != 0:
        return 1
    print("[install] image toolchain ready (diffusers + LoRA).")
    return 0
