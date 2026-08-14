#!/usr/bin/env node
// Copyright (c) 2026 Mike Koci-Eliseart Company. All rights reserved.
//
// ppte - npm CLI wrapper.
//
// Bootstraps the venv on first run (if postinstall was skipped) and forwards
// all arguments to the Python CLI, preserving stdio and exit codes. User data
// (data/, checkpoints) resolves to the caller's working directory, not the
// package dir.
import { existsSync } from "node:fs";
import { spawnSync } from "node:child_process";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import os from "node:os";

const here = dirname(fileURLToPath(import.meta.url));
const pkgRoot = join(here, "..");
const pyRoot = join(pkgRoot, "python");
const venvDir = join(pkgRoot, ".venv");

const isWin = os.platform() === "win32";
const pythonBin = isWin ? "python.exe" : "bin/python3";
const venvPython = join(venvDir, pythonBin);

// The venv is only usable once the base deps (e.g. pyyaml) are installed.
// A venv that exists but has no deps (failed/interrupted install) is useless,
// so verify by importing yaml; if that fails, run the bootstrap (re)install.
function venvUsable() {
  if (!existsSync(venvPython)) return false;
  const probe = spawnSync(venvPython, ["-c", "import yaml"], { stdio: "ignore" });
  return probe.status === 0;
}

if (!venvUsable()) {
  console.error("[ppte] venv missing/incomplete - running bootstrap ...");
  const boot = spawnSync(
    process.execPath,
    [join(here, "..", "scripts", "bootstrap.mjs")],
    { stdio: "inherit", env: process.env }
  );
  if (boot.status !== 0) process.exit(boot.status ?? 1);
  if (!venvUsable()) {
    console.error(
      "[ppte] bootstrap did not complete. Check disk space and try again: `npm install -g @eliseart/ppte`"
    );
    process.exit(1);
  }
}

// Put the bundled python source on the import path and run as a module so
// `from src...` imports resolve regardless of the caller's cwd.
const env = { ...process.env, PYTHONPATH: pyRoot };
const r = spawnSync(venvPython, ["-m", "src.main", ...process.argv.slice(2)], {
  stdio: "inherit",
  env,
});
if (r.error) {
  console.error("[ppte] failed to run:", r.error.message);
  process.exit(1);
}
process.exit(r.status ?? 0);