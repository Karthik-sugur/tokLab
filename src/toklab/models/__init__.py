"""Optional torch-backed model load, gate, adapt, and fine-tune helpers.

Intrinsic tokenization never imports this package.
"""

from toklab.models.gate import GateResult, GateVerdict, ModelVariant, gate

__all__ = ["GateResult", "GateVerdict", "ModelVariant", "gate"]