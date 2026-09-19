"""Supporting numerics for mathkit.numerical_analysis's systems/ modules."""

from mathkit.numerical_analysis.utils.error_analysis import condition_number, estimate_convergence_order, lebesgue_constant

__all__ = [
    "estimate_convergence_order",
    "lebesgue_constant",
    "condition_number",
]
