import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const uiRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = resolve(uiRoot, "..");
const apiHealthUrl = "http://127.0.0.1:8765/api/health";

function pythonExecutable() {
  const candidates = [
    process.env.OMNIPIANO_PYTHON,
    process.env.VIRTUAL_ENV && join(process.env.VIRTUAL_ENV, "bin", "python"),
    process.env.CONDA_PREFIX && join(process.env.CONDA_PREFIX, "bin", "python"),
    join(repositoryRoot, ".venv", "bin", "python"),
    join(repositoryRoot, ".venv-marl", "bin", "python"),
  ].filter(Boolean);
  return candidates.find((candidate) => existsSync(candidate)) ?? "python3";
}

async function apiIsReady() {
  try {
    const response = await fetch(apiHealthUrl, { signal: AbortSignal.timeout(500) });
    const payload = await response.json();
    return response.ok && payload.service === "omnipiano-experiments";
  } catch {
    return false;
  }
}

async function waitForApi(processHandle) {
  for (let attempt = 0; attempt < 80; attempt += 1) {
    if (await apiIsReady()) return;
    if (processHandle.exitCode !== null) {
      throw new Error(`Local experiment API exited with code ${processHandle.exitCode}.`);
    }
    await new Promise((resolveWait) => setTimeout(resolveWait, 100));
  }
  throw new Error("Timed out waiting for the local experiment API on port 8765.");
}

let apiProcess = null;
if (!(await apiIsReady())) {
  apiProcess = spawn(
    pythonExecutable(),
    ["-m", "omnipiano.experiments", "serve", "--allowed-root", "runs"],
    { cwd: repositoryRoot, env: process.env, stdio: "inherit" },
  );
  await waitForApi(apiProcess);
}

const viteProcess = spawn(
  join(uiRoot, "node_modules", ".bin", "vite"),
  process.argv.slice(2),
  { cwd: uiRoot, env: process.env, stdio: "inherit" },
);

let shuttingDown = false;

function stop(signal = "SIGTERM") {
  shuttingDown = true;
  if (viteProcess.exitCode === null && viteProcess.signalCode === null) viteProcess.kill(signal);
  if (apiProcess?.exitCode === null && apiProcess?.signalCode === null) apiProcess.kill(signal);
}

process.on("SIGINT", () => stop("SIGINT"));
process.on("SIGTERM", () => stop("SIGTERM"));

viteProcess.on("exit", (code, signal) => {
  shuttingDown = true;
  if (apiProcess?.exitCode === null && apiProcess?.signalCode === null) apiProcess.kill("SIGTERM");
  process.exitCode = code ?? (signal ? 1 : 0);
});

apiProcess?.on("exit", (code) => {
  if (!shuttingDown && viteProcess.exitCode === null && viteProcess.signalCode === null) {
    console.error(`Local experiment API stopped unexpectedly (code ${code}).`);
    viteProcess.kill("SIGTERM");
  }
});
