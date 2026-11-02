"""Retrieval-anchored Bayesian autotuning for Triton matmul kernels."""

# Schema identifiers such as "heliostune.suite/1" and operator names such as
# "heliostune_fusion_v2::..." keep the project's former name on purpose, because they
# are written into saved results that are compared byte for byte.

from importlib.metadata import version

from quartet.configs import DEFAULT_CONFIGS, DEFAULT_WORKLOADS, KernelConfig, Workload
from quartet.schema import HardwareProfile, Measurement, read_jsonl, write_jsonl

__version__ = version("quartet")

__all__ = [
    "DEFAULT_CONFIGS",
    "DEFAULT_WORKLOADS",
    "HardwareProfile",
    "KernelConfig",
    "Measurement",
    "Workload",
    "__version__",
    "read_jsonl",
    "write_jsonl",
]
