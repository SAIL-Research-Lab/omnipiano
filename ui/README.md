# OmniPiano Experiment Composer

This local web UI edits the same schema-v3 experiment document consumed by
`omnipiano.experiments`. It does not contain a second contract or reimplement
the four trainers.

## Run locally

From the repository root:

```bash
cd ui
npm install
npm run dev
```

Open `http://127.0.0.1:5173/`. The development command starts both the
loopback experiment API and Vite. If the API is already running, it is reused.
Use `npm run dev:ui` only when managing the API separately.

The single scrollable page lets a user:

- select Standard, Robust, Safe, or Multi-Agent RL;
- select a registered task (or define an exact custom MARL topology);
- export the configured environment as a portable ZIP containing `task.json`,
  an importable Python adapter, a manifest, and a runnable smoke test;
- optionally reveal the reference-baseline workflow and full experiment JSON;
- choose only algorithms and parameters accepted by that track;
- configure seeds, budget, evaluation, video, compute, tracking, and output;
- inspect, import, copy, or download the canonical JSON;
- validate and preview the exact backend commands;
- prepare run manifests without launching;
- explicitly confirm launch, then inspect or stop local runs.

Advanced controls live on the panel they affect. The UI fetches algorithms
and environment choices from the local API, imports the canonical JSON schema
and examples from `omnipiano/experiments/`, and keeps no duplicate catalog.

## Verification

```bash
cd ui
npm test
npm run build
```

The API is intentionally local-only: it has no CORS permission for remote
origins, accepts no arbitrary command, writes only below the configured output
root, and never launches until the user presses the launch action and confirms.
