# Quick Start

This guide follows the installation and usage flow in the repository README.
It installs OmniPiano, prepares the Piano Fingering Dataset, and runs a first
Gymnasium environment.

## Create environment

OmniPiano targets Linux with Python 3.10 or newer. Windows users can follow the
same commands inside WSL2. Install the required native packages first:

```bash
sudo apt-get update
sudo apt-get install -y build-essential fluidsynth libfluidsynth-dev portaudio19-dev ffmpeg

```

Using Miniconda or Miniforge is recommended:

```bash
conda create -n pianist python=3.10 -y
conda activate pianist
```

## Clone and install OmniPiano

```bash
git clone https://github.com/SafeRL-Lab/omnipiano.git
cd omnipiano
python -m pip install -e .
```

OmniPiano bundles the Shadow Hand model files and the default
`TimGM6mb.sf2` soundfont. You do not need to initialize a Git submodule or run
the original RoboPianist `install_deps.sh` script.

## Prepare the PIG dataset

Benchmark tasks use pieces from the
[Piano Fingering Dataset (PIG)](https://beam.kisarazu.ac.jp/~saito/research/PianoFingeringDataset/).
Download `PianoFingeringDataset_v1.2.zip` from the dataset website, extract it,
and preprocess the fingering files:

```bash
robopianist preprocess --dataset-dir /PATH/TO/PianoFingeringDataset_v1.2
robopianist --check-pig-exists
```

## Verify the installation

The built-in debug piece provides a quick smoke test without the PIG dataset:

```bash
python -c "from omnipiano.envs.robopianist import suite; env = suite.load('RoboPianist-debug-TwinkleTwinkleLittleStar-v0'); print('OK')"
```

After preprocessing PIG, verify a benchmark task:

```bash
python -c "from omnipiano import make; env = make('OmniPiano-ClairDeLune-CollisionSafe-v0'); print('OK')"
```

## Run a first environment

OmniPiano single-agent tasks follow the standard Gymnasium API:

```python
from omnipiano import make

env = make("OmniPiano-ClairDeLune-CollisionSafe-v0")
observation, info = env.reset(seed=42)

terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    # Safety cost remains separate from the musical reward.
    step_cost = info["step_safety/cost_total"]

print("musical F1:", info["episode_task/f1"])
print("episode safety cost:", info["episode_safety/cost_total"])
env.close()
```

The random policy above is only an interface demonstration. Meaningful musical
performance requires training a policy with an RL algorithm.
