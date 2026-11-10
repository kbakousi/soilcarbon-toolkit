"""Read and harmonise soil-carbon scenario outputs.

RothC / GSOCseq workflows write one row per grid cell or point with the
initial stock, the business-as-usual (BAU) stock at the end of the horizon
and stocks under alternative management scenarios. Column names vary
between runs: shapefiles truncate field names to 10 characters, and
different scripts use different spellings. :func:`standardize_columns` maps
the known variants onto one canonical schema and refuses ambiguous input.

Canonical columns (all stocks in t C/ha):

=============  ==============================================
``soc_t0``     initial SOC stock
``soc_bau``    stock at the horizon under business as usual
``soc_low``    stock under the low-intensity scenario
``soc_med``    stock under the medium-intensity scenario
``soc_high``   stock under the high-intensity scenario
``unc_t0``     uncertainty of the initial stock
``unc_bau``    uncertainty of the BAU stock
``unc_ssm``    uncertainty of the scenario stocks
=============  ==============================================

References
----------
FAO (2022). Global soil organic carbon sequestration potential map (GSOCseq
    v1.1) - Technical manual. Rome. https://doi.org/10.4060/cb2642en
Coleman, K., Jenkinson, D.S. (1996). RothC-26.3 - A model for the turnover
    of carbon in soil. In: Evaluation of Soil Organic Matter Models,
    237-246. https://doi.org/10.1007/978-3-642-61094-3_17
"""

from __future__ import annotations

import pandas as pd

__all__ = ["CANONICAL", "DEFAULT_ALIASES", "standardize_columns", "SCENARIOS"]

CANONICAL = ["soc_t0", "soc_bau", "soc_low", "soc_med", "soc_high", "unc_t0", "unc_bau", "unc_ssm"]

#: Scenario columns compared against BAU.
SCENARIOS = ["soc_low", "soc_med", "soc_high"]

#: Known spellings per canonical column (case-insensitive). Includes
#: shapefile-truncated forms.
DEFAULT_ALIASES: dict[str, list[str]] = {
    "soc_t0": ["SOC_t0", "soc_t0"],
    "soc_bau": ["SOC_BAU_20", "SOC_BAU", "BAU"],
    "soc_low": ["Low_Scenario", "Low_Scenar", "Low_Scena0", "Lw_Sc"],
    "soc_med": ["Med_Scenario", "Med_Scenar", "Med_Scena0", "Md_Sc"],
    "soc_high": ["High_Scenario", "High_Scena", "High_Scen0", "Hg_Sc"],
    "unc_t0": ["UNC_t0"],
    "unc_bau": ["UNC_BAU"],
    "unc_ssm": ["UNC_SSM"],
}


def standardize_columns(df: pd.DataFrame, aliases: dict | None = None, required=("soc_t0", "soc_bau")) -> pd.DataFrame:
    """Return a copy with canonical column names.

    Raises ``ValueError`` if a canonical column matches more than one input
    column (e.g. both ``Med_Scenar`` and ``Med_Scena0``: two different
    outputs mixed in one table), or if a ``required`` column is missing.
    """
    aliases = DEFAULT_ALIASES if aliases is None else aliases
    lower = {c.lower(): c for c in df.columns}
    rename = {}
    for canon, names in aliases.items():
        hits = list(dict.fromkeys(lower[n.lower()] for n in names if n.lower() in lower))
        if len(hits) > 1:
            raise ValueError(f"{canon}: several matching columns {hits}; select one output before analysing")
        if hits:
            rename[hits[0]] = canon
    out = df.rename(columns=rename)
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"missing required columns {missing}")
    return out
