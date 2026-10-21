# OmniPiano experiment UI contract

This directory contains the isolated local experiment composer. It does not
import, register, or modify any OmniPiano environment or training entrypoint.

## Run the local UI

From the repository root:

```bash
cd ui
npm install
npm run dev
```

Open the local address printed by Vite (normally
`http://127.0.0.1:5173/`). A production build can be checked with:

```bash
npm run build
npm test
```

The current browser prototype provides:

- one scrollable configuration page with six anchored sections;
- Standard, Robust, Safe, and Multi-Agent track switching;
- separate in-memory state for every track;
- independent Advanced controls on the cards that need them;
- preset-driven one-to-five-hand layouts with compact keyboard previews;
- automatic MARL ownership rebalancing after hand/agent-count changes;
- MARL hand ownership, action/observation range, overlap, and teammate-view editors;
- composable Robust signal/physical channels with L/M/H presets;
- conditional Safety cost targets, thresholds, and paper/custom budgets;
- compact Standard benchmark-variant cards;
- track-specific algorithm choices drawn from the repository's current
  implementations/templates;
- algorithm-derived MARL critic mode, including the MAPPO own-critic ablation;
- grouped environment and reward controls in the task card;
- structural and cross-field validation with a live JSON preview;
- canonical JSON import and lossless JSON round trips;
- copy and download actions for valid JSON.

It intentionally has no training launcher, repository write endpoint, W&B
connection, authentication, database, or other backend service.

The configuration contract remains independent from the React application:

- `contract/experiment.schema.json`: the canonical, versioned JSON Schema.
- `contract/support.json`: backend-support metadata for the future exporter.
- `contract/examples/`: one complete example for each paper track. These are
  also the future UI's initial form states, so defaults are not duplicated in
  a second file.
- `contract/validate_contract.py`: schema, semantic, and JSON round-trip checks.

The contract deliberately separates the shared piano task from the selected
benchmark design:

```text
shared task + one design (standard / robust / safety / marl)
            + algorithm + protocol + evaluation + tracking
```

The files here describe what the UI can configure. They are not yet training
inputs. In particular, the existing MARL v2 compiler remains unchanged; a
later exporter will translate this canonical format into its current input
format.

The shared hand editor describes initial placement, not MARL movement limits.
MARL action key ranges configure the intended wrist limits; Standard's mobility
variant records per-hand mobile flags in the JSON. Neither of these UI fields
is sent to a runtime by this prototype. Algorithm-specific parameters in the
UI are starter request values, not a claim that every backend accepts them
without a future exporter.

## Contract stability

Contract version `2` is the current local UI contract. Frontend layout or
wording may change without a version bump; adding, removing, renaming, or
changing the meaning of an exported JSON field requires a new contract version
and migrated examples. The UI upgrades version-1 imports by dropping the
unused MARL SCHO label and normalizing its original two-agent names.
Backend exporters should consume the contract only
after their corresponding track is merged.

Validate all checked-in examples with:

```bash
.venv-marl/bin/python ui/contract/validate_contract.py
```

The validator only needs `jsonschema` (already present in the MARL development
environment) and never imports the OmniPiano Python package.
