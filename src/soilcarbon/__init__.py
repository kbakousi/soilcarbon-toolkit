"""soilcarbon: soil organic carbon scenario analysis, sampling design and interpolation.

Each module is an independent implementation of the methods described in the
publications cited in its docstring.
"""

from . import interp, io, sampling, scenarios, units, validate

__all__ = ["interp", "io", "sampling", "scenarios", "units", "validate"]
__version__ = "0.1.1"
