"""Unit conversions for carbon accounting.

Carbon to carbon dioxide uses the ratio of molar masses, 44/12 (IPCC).
"""

from __future__ import annotations

__all__ = ["C_TO_CO2", "c_to_co2", "co2_to_c", "stock_t"]

#: Molar-mass ratio CO2 / C.
C_TO_CO2 = 44.0 / 12.0


def c_to_co2(t_c):
    """Tonnes of carbon to tonnes of CO2-equivalent."""
    return t_c * C_TO_CO2


def co2_to_c(t_co2):
    """Tonnes of CO2 to tonnes of carbon."""
    return t_co2 / C_TO_CO2


def stock_t(t_c_per_ha, area_ha):
    """Stock in t C from a density (t C/ha) and an area (ha)."""
    return t_c_per_ha * area_ha
