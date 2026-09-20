#!/usr/bin/env python3
from pathlib import Path
import copy
import datetime as dt
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(
    "/root/autodl-tmp/omnipiano/configs/scho/"
    "scho_v1_20260919T140444Z/main/runs"
)
stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
suite = f"scho_native_v1_{stamp}"
output = ROOT / "configs/scho" / suite
runs = output / "main/runs"
runs.mkdir(parents=True, exist_ok=False)

originals = []
for path in sorted(SOURCE.glob("*.json")):
    cfg = json.loads(path.read_text())
    if cfg["experiment"]["algo"] == "ippo":
        originals.append((path, cfg))
if len(originals) != 12:
    raise RuntimeError("Expected exactly 12 original IPPO configurations")

identities = {(cfg["task"]["name"], cfg["experiment"]["seed"]) for _, cfg in originals}
if len(identities) != 12 or len({task for task, _ in identities}) != 4:
    raise RuntimeError("Expected four settings with three seeds each")
for task in {task for task, _ in identities}:
    if {seed for name, seed in identities if name == task} != {0, 1, 2}:
        raise RuntimeError("Seed coverage differs from 0,1,2")

manifest = []
index = 0
for source, original in originals:
    for algo in ("happo", "mat", "facmac", "masac", "ppo-monolithic"):
        cfg = copy.deepcopy(original)
        if cfg["protocol"]["total_steps"] != 10_000_000:
            raise RuntimeError("Source budget is not 10M")
        setting = cfg["task"]["name"].removeprefix("scho-v1-winterwind-4h-")
        seed = cfg["experiment"]["seed"]
        name = f"{suite}_{setting}_{algo}_s{seed}_10M"
        cfg["experiment"].update(
            algo=algo,
            run_dir=f"/autodl-fs/data/omnipiano_runs/{suite}/train/{name}",
        )
        cfg["compute"]["ray_num_cpus"] = None
        cfg["compute"]["num_workers"] = 3
        cfg["algorithm_overrides"] = {}
        cfg["native"] = {}
        cfg["wandb"].update(
            name=name, group=f"{suite}_{setting}_{algo}",
            tags=f"{suite},{setting},{algo},seed{seed},10M,native,validation-pending",
            notes=(cfg["wandb"].get("notes") or "")
                  + "; native backend; independent validation required",
        )
        path = runs / f"{index:02d}_{name}.json"
        path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n")
        manifest.append({
            "config": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "source": source.name,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "algorithm": algo, "seed": seed, "task": cfg["task"]["name"],
        })
        index += 1

(output / "manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
launch = ROOT / ".marl-launch"
launch.mkdir(exist_ok=True)
(launch / "current_native_path").write_text(str(output) + "\n")
print("Generated:", index)
print("Suite:", suite)
print("Configs:", runs)