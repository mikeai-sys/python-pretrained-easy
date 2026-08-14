// Copyright (c) 2026 Mike Koci-Eliseart Company. All rights reserved.
//
// Copies the Python project source into the npm package so the postinstall
// bootstrap can build a venv from it. Runs on `npm pack`/`npm publish` via the
// `prepare` script.
import { cpSync, mkdirSync, rmSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, ".."); // npm package root
const project = resolve(root, ".."); // Python project root

const files = ["src", "config.yaml", "pyproject.toml", "README.md"];
const dest = resolve(root, "python");
mkdirSync(dest, { recursive: true });
for (const f of files) {
  const src = resolve(project, f);
  rmSync(resolve(dest, f), { recursive: true, force: true });
  cpSync(src, resolve(dest, f), { recursive: true, filter: (p) => !p.includes("__pycache__") });
}
console.log("[ppte] staged python source into python/");