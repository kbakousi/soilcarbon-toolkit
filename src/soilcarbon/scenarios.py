"""Scenario analysis of modelled SOC stocks (RothC / GSOCseq outputs).

Definitions follow the GSOCseq technical manual (FAO 2022):

- **absolute change**: ``scenario - soc_t0`` (t C/ha over the horizon);
- **additional sequestration (delta)**: ``scenario - soc_bau``, the gain
  attributable to the practice change relative to business as usual;
- **rates**: change divided by the horizon length (t C/ha/yr).

Group summaries are **area-weighted** means (each row weighted by its area),
and totals multiply per-hectare values by area. Relative changes are NaN
where the reference stock is zero or negative, never divided by a small
number.

References
----------
FAO (2022). Global soil organic carbon sequestration potential map (GSOCseq
    v1.1) - Technical manual. Rome. https://doi.org/10.4060/cb2642en
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .io import SCENARIOS
from .units import C_TO_CO2

__all__ = ["changes", "plausible", "ordering_check", "area_weighted_summary", "totals"]


def _rel(num, den):
    num = np.asarray(num, dtype="float64")
    den = np.asarray(den, dtype="float64")
    out = np.full(np.broadcast(num, den).shape, np.nan)
    np.divide(100.0 * num, den, out=out, where=np.isfinite(den) & (den > 0))
    return out


def changes(df: pd.DataFrame, horizon_years: float, scenarios=None) -> pd.DataFrame:
    """Add change, delta and rate columns for BAU and each scenario.

    New columns per scenario ``s`` (and ``abs_change`` / ``rate`` for
    ``soc_bau``):

    - ``abs_change_<s>`` = s - soc_t0
    - ``rel_change_<s>`` = 100 * abs_change / soc_t0 (%)
    - ``delta_<s>`` = s - soc_bau (additional sequestration)
    - ``rel_delta_<s>`` = 100 * delta / soc_bau (%)
    - ``rate_<s>`` = abs_change / horizon_years
    - ``delta_rate_<s>`` = delta / horizon_years
    """
    if horizon_years <= 0:
        raise ValueError("horizon_years must be > 0")
    scenarios = [s for s in (scenarios or SCENARIOS) if s in df.columns]
    out = df.copy()
    for s in ["soc_bau", *scenarios]:
        out[f"abs_change_{s}"] = out[s] - out["soc_t0"]
        out[f"rel_change_{s}"] = _rel(out[f"abs_change_{s}"], out["soc_t0"])
        out[f"rate_{s}"] = out[f"abs_change_{s}"] / horizon_years
    for s in scenarios:
        out[f"delta_{s}"] = out[s] - out["soc_bau"]
        out[f"rel_delta_{s}"] = _rel(out[f"delta_{s}"], out["soc_bau"])
        out[f"delta_rate_{s}"] = out[f"delta_{s}"] / horizon_years
    return out


def plausible(df: pd.DataFrame, column: str, lower: float, upper: float) -> pd.Series:
    """Boolean mask of rows with ``lower <= column <= upper``.

    The range is a required, documented choice (e.g. the upper bound of the
    source SOC map for mineral soils); nothing is filtered silently.
    """
    v = df[column]
    return (v >= lower) & (v <= upper)


def ordering_check(df: pd.DataFrame, scenarios=None, tol: float = 1e-9, order=None) -> pd.DataFrame:
    """Flag rows where scenario stocks are not in the expected order.

    By default the order is BAU <= low <= med <= high, as for the GSOCseq
    sustainable-management scenarios, which all add carbon inputs. Designs
    with a scenario below business as usual (e.g. an IPCC low-input factor
    below 1) need their own order, e.g.
    ``order=["soc_low", "soc_bau", "soc_med", "soc_high"]``.

    Returns a DataFrame of booleans (one column per adjacent pair) plus
    ``any_violation``. Violations usually point to swapped labels or mixed
    outputs, which is worth checking before reporting.
    """
    if order is None:
        order = ["soc_bau", *[s for s in (scenarios or SCENARIOS) if s in df.columns]]
    missing = [c for c in order if c not in df.columns]
    if missing:
        raise KeyError(f"columns not found: {missing}")
    res = pd.DataFrame(index=df.index)
    for a, b in zip(order[:-1], order[1:], strict=True):
        res[f"{a}>{b}"] = df[a] > df[b] + tol
    res["any_violation"] = res.any(axis=1)
    return res


def area_weighted_summary(df: pd.DataFrame, value_cols, area_col: str, by: str | None = None) -> pd.DataFrame:
    """Area-weighted mean of ``value_cols`` overall or per group ``by``.

    Rows with a NaN value are excluded for that column only. Also returns
    the area (ha) contributing to each mean.
    """
    value_cols = list(value_cols)

    def _one(g):
        res = {}
        for c in value_cols:
            ok = g[c].notna() & g[area_col].notna()
            a = g.loc[ok, area_col]
            res[c] = float(np.average(g.loc[ok, c], weights=a)) if a.sum() > 0 else np.nan
        res["area_ha"] = float(g[area_col].sum())
        return pd.Series(res)

    if by is None:
        return _one(df).to_frame("all").T
    return df.groupby(by).apply(_one, include_groups=False)


def totals(df: pd.DataFrame, value_cols, area_col: str, by: str | None = None, co2: bool = False) -> pd.DataFrame:
    """Total (per-ha value x area) in t C, or t CO2 if ``co2`` (x 44/12)."""
    factor = C_TO_CO2 if co2 else 1.0
    tmp = df[list(value_cols)].mul(df[area_col], axis=0) * factor
    if by is None:
        return tmp.sum().to_frame("all").T
    tmp[by] = df[by]
    return tmp.groupby(by).sum()
