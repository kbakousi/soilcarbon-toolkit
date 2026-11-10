"""Agreement between modelled and measured values (e.g. modelled vs sampled SOC).

Error statistics are computed against the 1:1 line (prediction vs
observation), not against a regression fitted to the same data, which would
hide bias. Lin's concordance correlation coefficient measures agreement with
the 1:1 line; Pearson's r only measures linear association.

Compare modelled values with **point measurements** at the same locations.
Comparing with an interpolated surface mixes model error with interpolation
error.

References
----------
Lin, L.I.-K. (1989). A concordance correlation coefficient to evaluate
    reproducibility. Biometrics 45, 255-268. https://doi.org/10.2307/2532051
"""

from __future__ import annotations

import numpy as np

__all__ = ["agreement"]


def agreement(observed, predicted) -> dict:
    """Return n, bias (mean error), MAE, RMSE, R^2 (1:1), Pearson r, Lin's CCC,
    and the OLS slope/intercept of observed on predicted (for information).

    Pairs with a NaN on either side are dropped.
    """
    o = np.asarray(observed, dtype="float64")
    p = np.asarray(predicted, dtype="float64")
    ok = np.isfinite(o) & np.isfinite(p)
    o, p = o[ok], p[ok]
    n = o.size
    if n < 3:
        raise ValueError("need at least 3 valid pairs")
    e = p - o
    ss_res = np.sum((o - p) ** 2)
    ss_tot = np.sum((o - o.mean()) ** 2)
    r = float(np.corrcoef(o, p)[0, 1])
    so2, sp2 = o.var(), p.var()  # population variances, as in Lin (1989)
    sop = np.mean((o - o.mean()) * (p - p.mean()))
    ccc = 2 * sop / (so2 + sp2 + (o.mean() - p.mean()) ** 2)
    slope, intercept = np.polyfit(p, o, 1)
    return {
        "n": int(n),
        "bias": float(e.mean()),
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "r2_1to1": float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan,
        "pearson_r": r,
        "ccc": float(ccc),
        "ols_slope": float(slope),
        "ols_intercept": float(intercept),
    }
