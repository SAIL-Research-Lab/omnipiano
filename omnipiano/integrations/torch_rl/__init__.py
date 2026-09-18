"""Optional TorchRL integration.

Install it with ``pip install -e .[torchrl]``.
"""

from .env.factory import make_env
from .model.policy_adapter import TorchRLPolicy

__all__ = ["TorchRLPolicy", "make_env"]
