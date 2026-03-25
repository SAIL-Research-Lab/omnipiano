"""SafeRoboPianist: A Safety and Robustness Benchmark for Robot Piano Playing."""

from setuptools import setup, find_packages
import os

safe_pkgs = find_packages(include=["safe_robopianist", "safe_robopianist.*"])

rp_pkgs = find_packages(
    where="safe_robopianist/envs",
    include=["robopianist", "robopianist.*"],
)

pkg_dir = {}
for pkg in rp_pkgs:
    pkg_dir[pkg] = os.path.join("safe_robopianist", "envs", *pkg.split("."))

setup(
    name="safe-robopianist",
    version="0.1.0",
    description="A Safety and Robustness Benchmark for Robot Piano Playing",
    packages=safe_pkgs + rp_pkgs,
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
        "numpy",
        "stable-baselines3",
        "moviepy",
    ],
)
