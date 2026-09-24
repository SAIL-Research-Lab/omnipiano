# Safety Quick Start

Use a **separate Conda environment and checkout** for safety experiments. The
frozen OmniSafe 0.5.0 backend uses older Gymnasium/SB3/NumPy versions than a fresh
installation of the general benchmark. Do not install this profile into your
existing `pianist` environment. This guide targets **Linux and Python 3.10** and
the repository's **`safety` branch**, not an untested merge with newer `main`.

The Shadow Hand assets and default `TimGM6mb.sf2` soundfont are bundled. No
submodule initialization or upstream `install_deps.sh` is required.
See [the RTX 4070 Ti installation test](INSTALLATION_TEST.md) for measured
coverage, dependency exceptions and limitations.

## 1. Install system dependencies

```bash
sudo apt-get update
sudo apt-get install -y git build-essential fluidsynth libfluidsynth-dev \
  portaudio19-dev ffmpeg libegl1 libgl1
```

An NVIDIA driver is additionally required for GPU training; installing PyTorch
does not install or repair the system driver. CUDA-enabled PyTorch wheels include
their CUDA runtime, so a separate CUDA toolkit is not needed for these commands.
The GPU profile below uses PyTorch 2.13.0 with CUDA 13 on an RTX 4070 Ti. Older
GPUs/drivers may require a different PyTorch build; that is not covered by this
tested profile. CPU execution does not require an NVIDIA driver.

## 2. Create an isolated environment

```bash
conda create -n omnipiano-safety python=3.10 -y
conda activate omnipiano-safety
python -m pip install --upgrade pip setuptools wheel
```

Use this environment whenever running the commands below. Your general-RL
environment remains unchanged.

## 3. Clone and install the safety branch

```bash
git clone --branch safety --single-branch \
  https://github.com/SAIL-Research-Lab/omnipiano.git omnipiano-safety
cd omnipiano-safety

python -m pip install -e . -r omnipiano/safety/requirements.txt
python -m pip install --no-deps omnisafe==0.5.0 safety-gymnasium==0.4.1
```

The first command resolves the project and the safety runtime dependencies
together. The second intentionally installs only the two specified packages;
their runtime dependencies are supplied by the first command. Do not replace
these commands with an unconstrained `pip install omnisafe`.

### Important dependency exception

Safety-Gymnasium 0.4.1 declares `mujoco==2.3.0`, whereas OmniPiano requires MuJoCo
3.x. A separate Conda environment isolates this conflict from other experiments,
but **does not remove the incompatible upstream metadata**. This installation
keeps MuJoCo 3.12.0 for OmniPiano and bypasses dependency resolution only for the
two packages installed in the second command. It does not patch their source or
package metadata. Safety-Gymnasium's own task suite is not the target of this
profile; OmniSafe trains the registered OmniPiano CMDP instead.

```bash
python -m pip check
```

Expect this known nonzero result:

```text
safety-gymnasium 0.4.1 has requirement mujoco==2.3.0, but you have mujoco 3.12.0.
```

Other missing/conflicting dependencies are **not** covered by this exception.
This is a tested runtime compatibility profile, not a resolver-clean dependency
set. Key versions are pinned; it is not a complete lock of every transitive
dependency. Preserve `python -m pip freeze` with published experiment results.

SciPy is pinned to 1.11.4 to avoid an import-time interaction with the old main
registry's NumPy shim. A small compatibility guard in `safety/__init__.py`
restores native `numpy.array` under NumPy 1.x once safety is imported; this fixes
`subok`/broadcasting used by dm_env without editing the registry or changing the
algorithm, reward or cost definitions. NumPy 2.x behavior is left untouched.
PettingZoo, PyAudio and termcolor are included because this branch imports them
even though its older root package metadata does not list all of them.

## 4. Set headless execution options

```bash
export MUJOCO_GL=egl
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```

EGL avoids requiring a desktop display on the NVIDIA test server. Do not select
OSMesa for this pinned GPU stack: the installation test encountered a native
Triton library import crash with that backend. MuJoCo physics runs on the CPU;
`--device cuda:0` selects the GPU for the policy/value networks. These thread
settings are per process, not a restriction on running independent experiments.
For CPU-only use without image/video rendering, set `MUJOCO_GL=disable` and use
`--device cpu`; this skips OpenGL entirely. Restore `egl` before GPU rendering.

## 5. Verify the installation without PIG

```bash
python -c "import omnisafe, torch; print('OmniSafe:', omnisafe.__version__); print('PyTorch:', torch.__version__)"
python -c "import omnipiano; from robopianist import suite; env = suite.load('RoboPianist-debug-TwinkleTwinkleLittleStar-v0'); env.reset(); env.close(); print('Debug environment OK')"
python -m pytest omnipiano/safety/tests -q
```

For GPU execution, also check an actual CUDA operation:

```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); x = torch.ones((32, 32), device='cuda:0'); print((x @ x).mean().item()); torch.cuda.synchronize()"
```

The numeric result should be `32.0`. A driver/library mismatch from `nvidia-smi`
is a system issue, not a missing Python package. Report it to the machine owner;
do not reinstall drivers or reboot a shared server as part of this tutorial.

## 6. Preprocess the PIG dataset

Download `PianoFingeringDataset_v1.2.zip` from the
[PIG dataset website](https://beam.kisarazu.ac.jp/~saito/research/PianoFingeringDataset/)
after completing its registration requirements, then extract it. Point
`--dataset-dir` at the directory containing `FingeringFiles/` and `List.csv`.

```bash
python -c "from robopianist.cli import main; main()" preprocess \
  --dataset-dir /PATH/TO/PianoFingeringDataset_v1.2
python -c "from robopianist.cli import main; main()" --check-pig-exists
```

The check should print `PIG dataset is ready to use!`. This explicit CLI call
also works on the safety branch's older root `setup.py`, which does not install
the `robopianist` console command. No root packaging changes are needed.

If you already have the 150 preprocessed `.proto` files, copy them into this
checkout's `omnipiano/envs/robopianist/music/data/pig_single_finger/` instead of
preprocessing again, then run the same check. Do not publish the dataset in Git.

## 7. Verify safety tasks and one short training run

From the repository root, after PIG is ready:

```bash
# Reset and take zero/random actions in all registered safety tasks.
python -m omnipiano.safety.smoke --steps 2 --out safety_smoke.json

# First main-experiment cell: PPO, 2,000 steps, one evaluation episode/checkpoint.
python -m omnipiano.safety.run --group main --index 0 \
  --out quickstart_smoke --smoke-test --device cuda:0 --execute

# First PPOLag cell, using the same short protocol.
python -m omnipiano.safety.run --group main --index 3 \
  --out quickstart_smoke --smoke-test --device cuda:0 --execute
```

Use `--device cpu` if no usable GPU is available. The short training commands
save a model and replay it, exercising training, checkpoint loading and the
reward/cost/F1 evaluation path. They are installation checks, not benchmark
results. For the first environment, indices `0, 3, 6, 9, 12` select PPO, PPOLag,
OnCRPO, CPO and CUP respectively (all seed 1). `--index` is a run-cell index,
not an environment index. Use a new output directory after changing code or
dependencies; these commands do not resume interrupted optimizer state.

## 8. Start experiments

```bash
# Print the plan without launching training.
python -m omnipiano.safety.run --group main --device cuda:0 --out results_main

# Train one selected main cell for 5M steps.
python -m omnipiano.safety.run --group main --index 0 \
  --device cuda:0 --out results_main --execute
```

Use `--group hands` or `--group budget` for the two ablations. Omitting `--index`
with `--execute` runs the selected group's cells sequentially. The existing
branch's formal protocol evaluates **10 episodes per checkpoint**; `--smoke-test`
uses **1**. This installation change does not alter experiment defaults or import
the separate eval-1 handoff packages. See [README.md](README.md) for the matrices,
cost definitions, evaluation details and plotting commands.

For additional algorithms (FOCOPS, CPPOPID, SACLag, Saute/Simmer, etc.), use the
same installed environment and follow [Algorithm interface](ALGORITHMS.md).
Its `--algorithms ... --eval-episodes 1` interface supports custom comparisons
without changing the original five-algorithm experiment defaults.

## Optional: higher-quality soundfont

```bash
python -c "from robopianist.cli import main; main()" soundfont --download
```

This is only needed for higher-quality audio output, not training or evaluation.
