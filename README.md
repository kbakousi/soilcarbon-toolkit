# soilcarbon-toolkit

[![Tests](https://github.com/kbakousi/soilcarbon-toolkit/actions/workflows/ci.yml/badge.svg)](https://github.com/kbakousi/soilcarbon-toolkit/actions/workflows/ci.yml) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24118766.svg)](https://doi.org/10.5281/zenodo.24118766)

Tools for soil organic carbon (SOC) work at regional scale:
- **sampling design** for field campaigns;
- **interpolation** of measured soil properties;
- **scenario analysis** of modelled SOC stocks (RothC / FAO GSOCseq outputs);
- **agreement statistics** between model and measurements.

The package is a refactoring of the analysis scripts of my MSc thesis (2025), which modelled SOC sequestration potential under alternative management scenarios in Eastern Macedonia & Thrace, Greece. Each module is an independent implementation of the methods described in the publications cited in its docstring.

## Workflow

```
stratified sampling  ->  field campaign  ->  interpolation (IDW / kriging)  ->  RothC / GSOCseq run  ->  scenario analysis
   soilcarbon.sampling                         soilcarbon.interp                   (external model)       soilcarbon.scenarios
                                                       \______ model vs measurement: soilcarbon.validate ______/
```

| Module | What it does | Method |
|---|---|---|
| `sampling` | Proportional or Neyman allocation over strata (e.g. pedoclimatic clusters), exact totals by largest remainder, stratified draw and train/test split | Cochran 1977; Balinski & Young 2001 |
| `interp` | North-up raster grid, IDW, empirical variogram, WLS variogram fit (spherical, exponential, Gaussian), ordinary kriging with kriging variance, leave-one-out CV | Shepard 1968; Matheron 1963; Cressie 1985 |
| `io` | One schema for scenario outputs (`soc_t0`, `soc_bau`, `soc_low`, `soc_med`, `soc_high`, uncertainties); detects mixed outputs | FAO 2022 (GSOCseq) |
| `scenarios` | Change vs initial stock, additional sequestration vs BAU, annual rates, scenario-ordering check, area-weighted summaries, totals in t C or t CO2 | FAO 2022; Coleman & Jenkinson 1996 |
| `validate` | Bias, MAE, RMSE and R² against the 1:1 line, Pearson r, Lin's CCC | Lin 1989 |
| `units` | C to CO2 (44/12), stock from density and area | IPCC convention |

The RothC model itself is not included. The package works on the outputs of the FAO GSOCseq workflow (FAO 2022).

## Example

```python
import pandas as pd
from soilcarbon import io, scenarios, sampling

df = io.standardize_columns(pd.read_csv("scenario_outputs.csv"))        # your RothC / GSOCseq outputs
df = df[scenarios.plausible(df, "soc_t0", lower=0, upper=250)]          # explicit, documented range
df = scenarios.changes(df, horizon_years=20)
print(scenarios.ordering_check(df)["any_violation"].sum(), "rows with BAU <= low <= med <= high violated")
summary = scenarios.area_weighted_summary(df, ["delta_soc_low", "delta_soc_med", "delta_soc_high"],
                                          area_col="area_ha", by="pedoclimatic_zone")
co2 = scenarios.totals(df, ["delta_soc_high"], area_col="area_ha", co2=True)   # t CO2 over 20 years

alloc = sampling.allocate(cells["cluster"].value_counts(), n=50, method="proportional", minimum=2)
```

## Tutorial

[`tutorials/01_soc_mapping_walkthrough.ipynb`](tutorials/01_soc_mapping_walkthrough.ipynb) works on a synthetic landscape whose true SOC is known. It covers:

- stratified sampling (proportional vs Neyman);
- why the land-use trend must be removed before fitting a variogram (regression kriging; Hengl et al. 2007);
- choosing between IDW and regression kriging by leave-one-out cross-validation;
- the map and its kriging uncertainty, checked against the truth;
- scenario deltas and totals in t CO₂.

It runs offline and is executed by the test suite.

## What changed compared with the thesis scripts

Refactoring the scripts into tested functions surfaced and fixed these issues (details in `CHANGELOG.md`):
1. **Upside-down rasters.** Interpolated rasters were written north–south flipped, and offset by half a pixel.
2. **RMSE against the wrong line.** RMSE was measured against a regression fitted to the same data instead of the 1:1 line.
3. **Inexact sample allocation.** Rounding could move a cluster by more than one sample from its exact share.
4. **Unweighted group means.** Summaries are now area-weighted.
5. **Division by zero.** Relative changes divided by zero or negative stocks; these are now NaN.
6. **Hidden filters.** A plausibility filter that was hard-coded, with different values in different scripts, is now an explicit parameter.
7. **Mixed outputs.** Scenario columns from two differently truncated shapefiles are now detected and rejected.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The tests use synthetic data only and check:
- the definitions and area weighting;
- that allocation is exact and fair;
- the raster orientation;
- that kriging is an exact interpolator and its weights sum to 1;
- the 1:1 error statistics.

## References

- Balinski, M.L., Young, H.P. (2001). *Fair Representation*, 2nd ed. Brookings Institution Press.
- Cochran, W.G. (1977). *Sampling Techniques*, 3rd ed. Wiley.
- Coleman, K., Jenkinson, D.S. (1996). RothC-26.3 – A model for the turnover of carbon in soil. In *Evaluation of Soil Organic Matter Models*, 237–246. https://doi.org/10.1007/978-3-642-61094-3_17
- Cressie, N. (1985). Fitting variogram models by weighted least squares. *Mathematical Geology* 17, 563–586. https://doi.org/10.1007/BF01032109
- FAO (2022). *Global soil organic carbon sequestration potential map (GSOCseq v1.1) – Technical manual*. Rome. https://doi.org/10.4060/cb2642en
- Hengl, T., Heuvelink, G.B.M., Rossiter, D.G. (2007). About regression-kriging: From equations to case studies. *Computers & Geosciences* 33, 1301–1315. https://doi.org/10.1016/j.cageo.2007.05.001
- Lin, L.I.-K. (1989). A concordance correlation coefficient to evaluate reproducibility. *Biometrics* 45, 255–268. https://doi.org/10.2307/2532051
- Matheron, G. (1963). Principles of geostatistics. *Economic Geology* 58, 1246–1266. https://doi.org/10.2113/gsecongeo.58.8.1246
- Shepard, D. (1968). A two-dimensional interpolation function for irregularly-spaced data. *Proc. 23rd ACM National Conference*, 517–524. https://doi.org/10.1145/800186.810616

## Licence and citation

Apache-2.0 (see `LICENSE` and `NOTICE`). To cite, use `CITATION.cff` ("Cite this repository" on GitHub).

Archived on Zenodo: [doi:10.5281/zenodo.24118766](https://doi.org/10.5281/zenodo.24118766) (all versions; each release also has its own DOI on that page).
