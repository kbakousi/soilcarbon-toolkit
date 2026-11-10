"""Stratified field-sampling design for soil campaigns.

Allocation of a fixed sample size ``n`` across strata (e.g. pedoclimatic
clusters):

- **proportional**: n_h proportional to the stratum size N_h;
- **Neyman**: n_h proportional to N_h * S_h (stratum standard deviation),
  which minimises the variance of the stratified mean (Cochran 1977).

Fractional allocations are rounded with the largest-remainder (Hamilton)
method, so the total is exactly ``n`` and no stratum is shifted by more than
one sample from its exact share. Simply rounding and then correcting the
largest stratum does not guarantee this.

References
----------
Cochran, W.G. (1977). Sampling Techniques, 3rd ed. Wiley, New York.
    (Sections 5.5 and 5.9: proportional and optimum/Neyman allocation)
Balinski, M.L., Young, H.P. (2001). Fair Representation: Meeting the Ideal
    of One Man, One Vote, 2nd ed. Brookings Institution Press.
    (Hamilton / largest-remainder method)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["largest_remainder", "allocate", "draw_stratified", "split_stratified"]


def largest_remainder(shares, n: int, minimum: int = 0) -> np.ndarray:
    """Integer allocation of ``n`` proportional to ``shares``.

    Each stratum first receives ``minimum`` units (if possible); the rest is
    allocated by largest remainders. Ties are broken by stratum order.
    """
    w = np.asarray(shares, dtype="float64")
    if np.any(w < 0) or w.sum() <= 0:
        raise ValueError("shares must be non-negative with a positive sum")
    base = np.full(w.size, minimum, dtype=int)
    rest = n - base.sum()
    if rest < 0:
        raise ValueError("n is smaller than minimum * number of strata")
    exact = rest * w / w.sum()
    alloc = np.floor(exact).astype(int)
    left = rest - alloc.sum()
    order = np.argsort(-(exact - alloc), kind="stable")
    alloc[order[:left]] += 1
    return base + alloc


def allocate(sizes: pd.Series, n: int, method: str = "proportional", std: pd.Series | None = None,
             minimum: int = 0) -> pd.Series:
    """Allocate ``n`` samples to strata.

    Parameters
    ----------
    sizes : pandas.Series
        Stratum sizes N_h (index = stratum), e.g. number of cells or area.
    n : int
    method : {"proportional", "neyman"}
    std : pandas.Series, optional
        Stratum standard deviations S_h (required for Neyman), e.g. from a
        prior SOC map.
    minimum : int
        Minimum samples per stratum (e.g. 2, to estimate a stratum variance).

    The allocation never exceeds a stratum's size.
    """
    if method == "proportional":
        shares = sizes.astype(float)
    elif method == "neyman":
        if std is None:
            raise ValueError("Neyman allocation needs stratum standard deviations")
        shares = sizes.astype(float) * std.reindex(sizes.index).astype(float)
    else:
        raise ValueError("method must be 'proportional' or 'neyman'")
    alloc = pd.Series(largest_remainder(shares.to_numpy(), n, minimum), index=sizes.index, name="n_samples")
    if (alloc > sizes).any():
        raise ValueError("allocation exceeds stratum size; reduce n or the minimum")
    return alloc


def draw_stratified(frame: pd.DataFrame, stratum_col: str, allocation: pd.Series, seed: int) -> pd.DataFrame:
    """Simple random sample without replacement within each stratum."""
    parts = [
        frame[frame[stratum_col] == h].sample(n=int(k), random_state=seed)
        for h, k in allocation.items()
        if k > 0
    ]
    return pd.concat(parts)


def split_stratified(sample: pd.DataFrame, stratum_col: str, test_fraction: float, seed: int) -> pd.Series:
    """Label each sampled point ``"train"`` or ``"test"``, stratified.

    The number of test points per stratum is allocated by largest remainder
    from ``test_fraction`` of the total, so small strata are not dropped from
    either set by rounding. Strata with a single point stay in ``train``.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    counts = sample[stratum_col].value_counts().sort_index()
    eligible = counts[counts > 1]
    label = pd.Series("train", index=sample.index, name="set")
    if eligible.empty:
        return label
    n_test = int(round(test_fraction * len(sample)))
    k_test = pd.Series(largest_remainder(eligible.to_numpy(), min(n_test, int((eligible - 1).sum()))),
                       index=eligible.index)
    k_test = np.minimum(k_test, eligible - 1)
    for h, k in k_test.items():
        idx = sample.index[sample[stratum_col] == h].to_numpy()
        label.loc[rng.choice(idx, size=int(k), replace=False)] = "test"
    return label
