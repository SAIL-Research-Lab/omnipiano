"""OmniPiano: A Safety and Robustness Benchmark for Robot Piano Playing."""

from setuptools import setup, find_packages
import os

omni_pkgs = find_packages(include=["omnipiano", "omnipiano.*"])

rp_pkgs = find_packages(
    where="omnipiano/envs",
    include=["robopianist", "robopianist.*"],
)

pkg_dir = {}
for pkg in rp_pkgs:
    pkg_dir[pkg] = os.path.join("omnipiano", "envs", *pkg.split("."))

setup(
    name="omnipiano",
    version="0.1.0",
    description="A Safety and Robustness Benchmark for Robot Piano Playing",
    packages=omni_pkgs + rp_pkgs,
    package_dir=pkg_dir,
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "dm_control>=1.0.16",
        "dm_env_wrappers>=0.0.11",
        "mujoco>=3.1.1",
        "mujoco_utils>=0.0.6",
        "note_seq>=0.0.5",
        "pretty_midi>=0.2.10",
        "pyfluidsynth>=1.3.2",
        "scikit-learn",
        "shimmy[dm_control]",
        "gymnasium",
        "pettingzoo==1.24.3",
        "numpy",
        "stable-baselines3",
        "moviepy",
    ],
    extras_require={
        # Exact RLlib target used by the reproducible IPPO baseline. Keeping
        # this optional avoids forcing Ray onto single-agent benchmark users.
        # "marl": ["ray[rllib]==2.55.1"],
        "marl": [
            "ray[rllib]==2.55.1",
            "pettingzoo>=1.24",
            "wandb>=0.17",
        ],
    },
)
