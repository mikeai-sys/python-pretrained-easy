// Copyright (c) 2026 Mike Koci-Eliseart Company. All rights reserved.
//
// Best-effort install of Python 3.10+ using the OS package manager.
// Exits 0 on success (or if a usable python is already present), 1 otherwise.
import { spawnSync } from "node:child_process";
import os from "node:os";

function have(cmd) {
  try {
    spawnSync(cmd, ["--version"], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

function pythonPresent() {
  const c = os.platform() === "win32" ? "python" : "python3";
  if (have(c)) return true;
  if (os.platform() === "win32") {
    // `py` launcher.
    try {
      spawnSync("py", ["-3", "--version"], { stdio: "ignore" });
      return true;
    } catch {
      return false;
    }
  }
  return false;
}

function run(cmd, args) {
  console.log(`[ppte] ${cmd} ${args.join(" ")}`);
  const r = spawnSync(cmd, args, { stdio: "inherit" });
  return r.status === 0;
}

function installLinux() {
  if (have("apt-get"))
    return run("sudo", ["apt-get", "update", "-y"]) &&
      run("sudo", ["apt-get", "install", "-y", "python3", "python3-venv", "python3-pip"]);
  if (have("dnf"))
    return run("sudo", ["dnf", "install", "-y", "python3", "python3-pip"]);
  if (have("yum"))
    return run("sudo", ["yum", "install", "-y", "python3", "python3-pip"]);
  return false;
}

function installMac() {
  if (!have("brew")) return false;
  return run("brew", ["install", "python@3.12"]);
}

function installWindows() {
  if (have("winget"))
    return run("winget", ["install", "-e", "--id", "Python.Python.3.12", "--accept-package-agreements", "--accept-source-agreements"]);
  return false;
}

if (pythonPresent()) {
  console.log("[ppte] python already available.");
  process.exit(0);
}

console.log("[ppte] python not found; attempting install ...");
let ok = false;
switch (os.platform()) {
  case "linux":
    ok = installLinux();
    break;
  case "darwin":
    ok = installMac();
    break;
  case "win32":
    ok = installWindows();
    break;
  default:
    ok = false;
}

if (ok && pythonPresent()) {
  console.log("[ppte] python installed.");
  process.exit(0);
}
console.error(
  "[ppte] Could not auto-install Python. Install Python 3.10+ from https://www.python.org/downloads/ and re-run."
);
process.exit(1);