"""Spatial interpolation of soil point data (e.g. SOC stock, clay content).

- Inverse distance weighting (Shepard 1968).
- Ordinary kriging with a variogram fitted by weighted least squares
  (Matheron 1963; Cressie 1985).
- Leave-one-out cross-validation to compare them.
- A north-up raster grid of pixel centres with a matching affine transform.

Grid convention: row 0 is the northernmost row, and the transform's origin
is the outer top-left corner of the top-left pixel (not its centre). Arrays
from :func:`grid` can be written to GeoTIFF directly, with no flipping.

References
----------
Shepard, D. (1968). A two-dimensional interpolation function for
    irregularly-spaced data. Proceedings of the 1968 23rd ACM National
    Conference, 517-524. https://doi.org/10.1145/800186.810616
Matheron, G. (1963). Principles of geostatistics. Economic Geology 58,
    1246-1266. https://doi.org/10.2113/gsecongeo.58.8.1246
Cressie, N. (1985). Fitting variogram models by weighted least squares.
    Mathematical Geology 17, 563-586. https://doi.org/10.1007/BF01032109
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

__all__ = [
    "Grid",
    "grid",
    "idw",
    "empirical_variogram",
    "VARIOGRAM_MODELS",
    "fit_variogram",
    "ordinary_kriging",
    "loocv",
    "write_geotiff",
]


@dataclass
class Grid:
    """Pixel-centre coordinates of a north-up grid and its affine transform."""

    x: np.ndarray  # column centres, west -> east
    y: np.ndarray  # row centres, north -> south
    transform: tuple  # (a, b, c, d, e, f) as in rasterio.Affine(a, b, c, d, e, f)

    @property
    def shape(self):
        return (self.y.size, self.x.size)

    def points(self):
        gx, gy = np.meshgrid(self.x, self.y)
        return np.column_stack([gx.ravel(), gy.ravel()])


def grid(bounds, resolution: float) -> Grid:
    """Build a north-up grid covering ``bounds = (minx, miny, maxx, maxy)``."""
    minx, miny, maxx, maxy = bounds
    if resolution <= 0 or maxx <= minx or maxy <= miny:
        raise ValueError("invalid bounds or resolution")
    ncol = int(np.ceil((maxx - minx) / resolution))
    nrow = int(np.ceil((maxy - miny) / resolution))
    x = minx + resolution * (np.arange(ncol) + 0.5)
    y = maxy - resolution * (np.arange(nrow) + 0.5)
    return Grid(x, y, (resolution, 0.0, minx, 0.0, -resolution, maxy))


def idw(xy_obs, z, xy_pred, power: float = 2.0, k: int | None = None) -> np.ndarray:
    """Inverse distance weighting (Shepard 1968).

    Parameters
    ----------
    xy_obs : (n, 2) observation coordinates (projected, metres)
    z : (n,) values
    xy_pred : (m, 2) prediction coordinates
    power : float
        Distance exponent (commonly 2).
    k : int, optional
        Use only the ``k`` nearest observations; all observations if None.

    A prediction location that coincides with an observation takes that
    observation's value (the mean, if several share the location), instead of
    using a tiny-distance approximation.
    """
    xy_obs = np.asarray(xy_obs, dtype="float64")
    z = np.asarray(z, dtype="float64")
    xy_pred = np.asarray(xy_pred, dtype="float64")
    kk = len(z) if k is None else min(k, len(z))
    dist, idx = cKDTree(xy_obs).query(xy_pred, k=kk)
    dist, idx = dist.reshape(len(xy_pred), kk), idx.reshape(len(xy_pred), kk)
    exact = dist == 0
    with np.errstate(divide="ignore"):
        w = np.where(exact, 0.0, 1.0 / dist**power)
    pred = (w * z[idx]).sum(axis=1) / w.sum(axis=1)
    has_exact = exact.any(axis=1)
    if has_exact.any():
        ze = np.where(exact, z[idx], 0.0)
        pred[has_exact] = ze[has_exact].sum(axis=1) / exact[has_exact].sum(axis=1)
    return pred


def empirical_variogram(xy, z, n_lags: int, max_dist: float):
    """Matheron's estimator: gamma(h) = sum (z_i - z_j)^2 / (2 N(h)).

    Returns lag centres, semivariances and pair counts for ``n_lags`` equal
    bins up to ``max_dist`` (commonly about half the maximum pair distance).
    Bins without pairs are dropped.
    """
    xy = np.asarray(xy, dtype="float64")
    z = np.asarray(z, dtype="float64")
    i, j = np.triu_indices(len(z), k=1)
    d = np.hypot(*(xy[i] - xy[j]).T)
    sq = (z[i] - z[j]) ** 2
    edges = np.linspace(0, max_dist, n_lags + 1)
    b = np.digitize(d, edges) - 1
    ok = (b >= 0) & (b < n_lags)
    n = np.bincount(b[ok], minlength=n_lags)
    s = np.bincount(b[ok], weights=sq[ok], minlength=n_lags)
    dsum = np.bincount(b[ok], weights=d[ok], minlength=n_lags)
    keep = n > 0
    return dsum[keep] / n[keep], s[keep] / (2 * n[keep]), n[keep]


def _spherical(h, nugget, psill, rng):
    h = np.asarray(h, dtype="float64")
    r = np.minimum(h / rng, 1.0)
    g = nugget + psill * (1.5 * r - 0.5 * r**3)
    return np.where(h == 0, 0.0, g)


def _exponential(h, nugget, psill, rng):
    h = np.asarray(h, dtype="float64")
    g = nugget + psill * (1 - np.exp(-3.0 * h / rng))  # rng = practical range
    return np.where(h == 0, 0.0, g)


def _gaussian(h, nugget, psill, rng):
    h = np.asarray(h, dtype="float64")
    g = nugget + psill * (1 - np.exp(-3.0 * (h / rng) ** 2))
    return np.where(h == 0, 0.0, g)


#: Variogram models gamma(h; nugget, partial sill, range). gamma(0) = 0.
VARIOGRAM_MODELS = {"spherical": _spherical, "exponential": _exponential, "gaussian": _gaussian}


def fit_variogram(lags, gamma, counts, model: str = "spherical"):
    """Fit (nugget, partial sill, range) by Cressie's (1985) weighted least squares.

    Minimises sum N(h) * (gamma_hat(h) / gamma(h; theta) - 1)^2 with
    nugget >= 0, partial sill >= 0 and range > 0.
    """
    f = VARIOGRAM_MODELS[model]
    lags, gamma, counts = (np.asarray(a, dtype="float64") for a in (lags, gamma, counts))
    x0 = [max(gamma.min(), 1e-9) * 0.5, max(gamma.max() - gamma.min(), 1e-9), lags.max() / 2]

    def resid(theta):
        g = np.maximum(f(lags, *theta), 1e-12)
        return np.sqrt(counts) * (gamma / g - 1.0)

    res = least_squares(resid, x0, bounds=([0, 0, 1e-9], [np.inf, np.inf, np.inf]))
    return dict(zip(("nugget", "psill", "range"), map(float, res.x), strict=True)) | {"model": model}


def ordinary_kriging(xy_obs, z, xy_pred, variogram: dict, chunk: int = 20000):
    """Ordinary kriging predictions and kriging variances.

    Solves the ordinary kriging system once (global neighbourhood) with a
    Lagrange multiplier enforcing weights that sum to 1. Suited to the
    tens to a few thousand points typical of soil campaigns.

    Returns
    -------
    pred, var : numpy.ndarray of shape (m,)
    """
    xy_obs = np.asarray(xy_obs, dtype="float64")
    z = np.asarray(z, dtype="float64")
    xy_pred = np.asarray(xy_pred, dtype="float64")
    f = VARIOGRAM_MODELS[variogram["model"]]
    theta = (variogram["nugget"], variogram["psill"], variogram["range"])
    n = len(z)
    A = np.ones((n + 1, n + 1))
    A[:n, :n] = f(cdist(xy_obs, xy_obs), *theta)
    A[n, n] = 0.0
    lu = lu_factor(A)
    pred = np.empty(len(xy_pred))
    var = np.empty(len(xy_pred))
    for s in range(0, len(xy_pred), chunk):
        p = xy_pred[s : s + chunk]
        b = np.ones((n + 1, len(p)))
        b[:n] = f(cdist(xy_obs, p), *theta)
        lam = lu_solve(lu, b)
        pred[s : s + chunk] = lam[:n].T @ z
        var[s : s + chunk] = np.sum(lam * b, axis=0)  # sum(lambda_i gamma_i0) + mu
    return pred, np.maximum(var, 0.0)  # clip floating-point round-off


def loocv(xy_obs, z, method: str, **kwargs) -> np.ndarray:
    """Leave-one-out predictions at the observation points.

    ``method`` is ``"idw"`` (kwargs: ``power``, ``k``) or ``"kriging"``
    (kwargs: ``variogram``). Compare with :func:`soilcarbon.validate.agreement`.
    """
    xy_obs = np.asarray(xy_obs, dtype="float64")
    z = np.asarray(z, dtype="float64")
    out = np.empty(len(z))
    for i in range(len(z)):
        m = np.ones(len(z), dtype=bool)
        m[i] = False
        if method == "idw":
            out[i] = idw(xy_obs[m], z[m], xy_obs[i : i + 1], **kwargs)[0]
        elif method == "kriging":
            out[i] = ordinary_kriging(xy_obs[m], z[m], xy_obs[i : i + 1], **kwargs)[0][0]
        else:
            raise ValueError("method must be 'idw' or 'kriging'")
    return out


def write_geotiff(path, array, g: Grid, crs: str, nodata: float = np.nan):
    """Write a 2-D array on grid ``g`` to GeoTIFF (requires rasterio)."""
    import rasterio
    from rasterio.transform import Affine

    arr = np.asarray(array, dtype="float32").reshape(g.shape)
    with rasterio.open(path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1], count=1,
                       dtype="float32", crs=crs, transform=Affine(*g.transform), nodata=nodata) as dst:
        dst.write(arr, 1)
