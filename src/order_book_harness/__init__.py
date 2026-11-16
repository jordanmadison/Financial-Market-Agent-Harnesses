"""Self-improving order-book placement research harness."""

from .domain import *
from .harness import H0, compile_harness
from .implementation import (
    H0_IMPLEMENTATION,
    FunctionImplementationSpec,
    HarnessImplementationSpec,
    compile_implementation_harness,
    replace_function_implementation,
)

__all__ = [
    "H0", "compile_harness", "H0_IMPLEMENTATION", "FunctionImplementationSpec",
    "HarnessImplementationSpec", "compile_implementation_harness", "replace_function_implementation",
]

__version__ = "0.1.0"
