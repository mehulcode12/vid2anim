#!/usr/bin/env node

/**
 * vid2anim CLI launcher (npm/npx wrapper)
 * Bridges npx/npm commands directly to the vid2anim Python engine.
 */

const { spawnSync } = require("child_process");
const path = require("path");
const fs = require("fs");

const args = process.argv.slice(2);

function run(command, cmdArgs) {
  const res = spawnSync(command, cmdArgs, { stdio: "inherit", shell: true });
  process.exit(res.status ?? 0);
}

function checkExecutable(cmd) {
  try {
    const isWin = process.platform === "win32";
    const checker = isWin ? "where" : "which";
    const check = spawnSync(checker, [cmd], { stdio: "pipe", shell: true });
    return check.status === 0;
  } catch (e) {
    return false;
  }
}

// 1. Check if vid2anim executable is directly available in PATH
if (checkExecutable("vid2anim")) {
  run("vid2anim", args);
}

// 2. Check if local virtualenv has vid2anim
const localVenvBin = process.platform === "win32"
  ? path.join(__dirname, "..", ".venv", "Scripts", "vid2anim.exe")
  : path.join(__dirname, "..", ".venv", "bin", "vid2anim");

if (fs.existsSync(localVenvBin)) {
  run(`"${localVenvBin}"`, args);
}

// 3. Fallback to python3 / python module execution
const pythonCandidates = process.platform === "win32" ? ["python", "py"] : ["python3", "python"];
for (const py of pythonCandidates) {
  if (checkExecutable(py)) {
    const checkMod = spawnSync(py, ["-m", "vid2anim.cli", "--version"], { stdio: "pipe", shell: true });
    if (checkMod.status === 0) {
      run(py, ["-m", "vid2anim.cli", ...args]);
    }
  }
}

// 4. If neither was found, display helpful installation guidance
console.log("=================================================================");
console.log("             vid2anim - AI Motion Capture Studio");
console.log("=================================================================");
console.log("vid2anim Python backend is not detected in your current environment.\n");
console.log("To install and run vid2anim, please install the Python engine:");
console.log("  pip install vid2anim\n");
console.log("Or clone and run locally:");
console.log("  git clone https://github.com/mehulcode12/vid2anim.git");
console.log("  cd vid2anim && pip install -e .\n");
console.log("For full documentation, visit: https://github.com/mehulcode12/vid2anim");
console.log("=================================================================");
process.exit(1);
