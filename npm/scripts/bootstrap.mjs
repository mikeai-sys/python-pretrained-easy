// Copyright (c) 2026 Mike Koci-Eliseart Company. All rights reserved.
//
// ppte postinstall bootstrap.
//
// Cross-platform setup for the Python CLI:
//   1. Find a Python 3.10+ interpreter (venv on Windows, else system python3).
//   2. Create an isolated venv inside the package.
//   3. Detect hardware: NVIDIA GPU (nvidia-smi) -> CUDA torch;
//      Apple Silicon/Intel Mac -> MPS/CPU torch; else CPU torch.
//   4. pip install the project + GPU/CPU-appropriate torch.
//
// Skips silently if the venv already exists and ppte is importable (fast
// reinstall / node_modules cache).
import { execFileSync, spawnSync } from "node:child_process";
import { existsSync, rmSync } from "node:fs";
import { createInterface } from "node:readline";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import os from "node:os";

function readLine() {
  return new Promise((res) => {
    const rl = createInterface({ input: process.stdin, output: process.stdout });
    rl.question("", (a) => {
      rl.close();
      res(a);
    });
  });
}

const here = dirname(fileURLToPath(import.meta.url));
const pkgRoot = resolve(here, "..");
const pyRoot = join(pkgRoot, "python");
const venvDir = join(pkgRoot, ".venv");

const pythonBin = os.platform() === "win32" ? "python.exe" : "bin/python3";
const pipBin = os.platform() === "win32" ? "Scripts/pip.exe" : "bin/pip";
const venvPython = join(venvDir, pythonBin);
const venvPip = join(venvDir, pipBin);

function run(cmd, args, opts = {}) {
  const r = spawnSync(cmd, args, { stdio: "inherit", ...opts });
  if (r.status !== 0) {
    console.error(`[ppte] command failed: ${cmd} ${args.join(" ")}`);
    process.exit(r.status ?? 1);
  }
}

function which(cmd) {
  try {
    execFileSync(cmd, ["--version"], { stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

function findPython() {
  if (process.env.PPTE_PYTHON && existsSync(process.env.PPTE_PYTHON)) {
    return process.env.PPTE_PYTHON;
  }
  const candidates = os.platform() === "win32"
    ? ["python", "py -3.12", "py -3.11", "py -3.10"]
    : ["python3.12", "python3.11", "python3.10", "python3"];
  for (const c of candidates) {
    if (which(c)) return c;
  }
  return null;
}

function hasNvidiaGpu() {
  if (os.platform() === "win32") {
    return which("nvidia-smi") || spawnSync("nvidia-smi", { stdio: "ignore" }).status === 0;
  }
  return which("nvidia-smi");
}

function torchIndex(platform, arch, gpu) {
  // CUDA wheels are only published for x64 on Linux/Windows.
  if (gpu && arch === "x64" && (platform === "linux" || platform === "win32")) {
    return "--index-url https://download.pytorch.org/whl/cu121";
  }
  // macOS + Apple Silicon: the default PyPI torch ships MPS support.
  return null; // plain `pip install torch`
}

const baseReqs = [
  "pyyaml>=6.0",
  "requests>=2.31",
  "beautifulsoup4>=4.12",
  "lxml>=5.0",
  "duckduckgo_search>=5.0",
];
const mlReqs = [
  "transformers>=4.40",
  "trl>=0.8",
  "peft>=0.10",
  "datasets>=2.19",
  "accelerate>=0.29",
];

console.log("[ppte] bootstrapping (first install only) ...");
let python = findPython();
if (!python) {
  console.log("[ppte] no python found; attempting to install ...");
  const { spawnSync } = await import("node:child_process");
  spawnSync(process.execPath, [join(here, "install-python.mjs")], { stdio: "inherit" });
  python = findPython();
}
if (!python) {
  console.error(
    "[ppte] No Python 3.10+ found. Install Python from https://www.python.org/downloads/ then re-run npm install."
  );
  process.exit(1);
}
console.log(`[ppte] using python: ${python}`);

// Create the venv. Uses `--without-pip` + get-pip bootstrap as a fallback so
// systems without the python3.x-venv package (missing ensurepip) still work.
// A failed `python -m venv` can leave a partial .venv behind, so this always
// wipes and rebuilds to guarantee a consistent state.
function createVenv(python) {
  let r = spawnSync(python, ["-m", "venv", venvDir], { stdio: "inherit" });
  if (r.status === 0) return true;

  console.log("[ppte] standard venv failed; retrying with --without-pip ...");
  rmSync(venvDir, { recursive: true, force: true });
  r = spawnSync(python, ["-m", "venv", "--without-pip", venvDir], { stdio: "inherit" });
  if (r.status !== 0) return false;

  // Bootstrap pip into the pip-less venv via get-pip.py.
  const getPip = join(venvDir, "get-pip.py");
  const dl = spawnSync(
    process.platform === "win32" ? "powershell" : "curl",
    process.platform === "win32"
      ? ["-Command", "Invoke-WebRequest -Uri https://bootstrap.pypa.io/get-pip.py -OutFile", getPip]
      : ["-fsSL", "https://bootstrap.pypa.io/get-pip.py", "-o", getPip],
    { stdio: "inherit" }
  );
  if (dl.status !== 0) return false;
  const boot = spawnSync(venvPython, [getPip, "--quiet"], { stdio: "inherit" });
  return boot.status === 0;
}

function venvUsable() {
  return existsSync(venvPython) && existsSync(venvPip);
}

if (!venvUsable()) {
  if (!createVenv(python)) {
    console.error(
      `[ppte] Could not create a virtualenv. Install ${python === "python3" ? "python3-venv" : `${python}-venv`} (Debian/Ubuntu) or ` +
        "`python3-virtualenv` (macOS) then re-run npm install."
    );
    process.exit(1);
  }
  console.log("[ppte] created venv");
}

// Determine hardware / torch index.
const platform = os.platform();
const arch = os.arch();
const gpu = hasNvidiaGpu();
console.log(
  `[ppte] platform=${platform} arch=${arch} nvidia_gpu=${gpu ? "yes" : "no"}`
);
const idx = torchIndex(platform, arch, gpu);
if (idx) console.log(`[ppte] torch index: ${idx}`);

// Check free disk space (GB).
function freeGb() {
  const r = spawnSync("python3", [
    "-c",
    "import os; s=os.statvfs(os.getcwd()); print(s.f_bavail*s.f_frsize//2**30)",
  ]);
  if (r.status !== 0) return null;
  const n = Number(String(r.stdout).trim());
  return Number.isFinite(n) ? n : null;
}

// Install plan:
//   base deps  - always (small, ~20 MB, needed by every command incl. --help)
//   ML deps    - only when >= 4 GB free (torch/transformers need ~3-4 GB);
//                otherwise deferred to `ppte install txt`
const free = freeGb();
console.log(`[ppte] free disk: ${free === null ? "unknown" : `~${free} GB`}`);

console.log("[ppte] installing base dependencies ...");
run(venvPip, ["install", "--upgrade", "pip", "--quiet"]);
run(venvPip, ["install", "--quiet", ...baseReqs]);

const installMl = free !== null && free >= 4;
let installedMl = installMl;
if (installMl) {
  console.log("[ppte] installing ML toolchain (torch, transformers, trl, ...) ...");
  const args = ["install", "--quiet", ...mlReqs];
  if (idx) args.push(idx);
  run(venvPip, args);
} else {
  // Ask the user directly when we're on a real terminal (not a silent npm
  // postinstall) so they can opt into the full install in one go.
  const why = free === null ? "(free space unknown)" : `(need ~4 GB, have ~${free} GB)`;
  const tty = process.stdin.isTTY && process.env.npm_config_loglevel !== "silent";
  let answer = "n";
  if (tty) {
    process.stdout.write(
      `[ppte] SKIPPED ML toolchain ${why}. It enables run/train/rag/img.\n` +
        `[ppte] Install it now? [y/N] `
    );
    const line = (await readLine()) ?? "";
    answer = line.trim().toLowerCase();
  } else {
    console.log(
      `[ppte] SKIPPED ML toolchain ${why}. ` +
        "Base commands work now; run `ppte install torch` when you have disk space."
    );
  }
  if (answer === "y" || answer === "yes") {
    console.log("[ppte] installing ML toolchain (torch, transformers, trl, ...) ...");
    const args = ["install", "--quiet", ...mlReqs];
    if (idx) args.push(idx);
    run(venvPip, args);
    installedMl = true;
  }
}

// After a fresh bootstrap, always tell the user about torch so no tool fails
// later with a confusing ModuleNotFoundError.
if (installedMl) {
  console.log(
    "\n[ppte] INSTALL COMPLETE. The ML toolchain (torch) is installed — all tools work.\n"
  );
} else {
  console.log(
    "\n[ppte] NOTE: to use ALL tools without failures (run, train, rag --summarize, img, ...)\n" +
      "       you need to install torch. Do:  ppte install torch\n" +
      "       (or `ppte install txt` for the full LLM stack, `ppte install img` for images).\n" +
      "       Base tools (--help, crawl --raw, render, links, config) work without it.\n"
  );
}

// Install the bundled project itself (registers the `ppte` console script).
run(venvPip, ["install", "--quiet", "--no-deps", "-e", pyRoot]);
console.log("[ppte] done. Run `npx ppte --help` to verify.");