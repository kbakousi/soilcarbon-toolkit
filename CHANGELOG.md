# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versioning: [SemVer](https://semver.org/).

## [0.1.1] - 2026-11-10

### Changed
- Licence changed from all rights reserved to Apache-2.0 (`LICENSE`, `NOTICE`).

## [0.1.0] - 2026-11-10

Refactored from the analysis scripts of the author's MSc thesis (2025) into a tested package.

### Added
- `io`: one canonical schema for scenario outputs; detects mixed outputs (e.g. `Med_Scenar` and `Med_Scena0` in one table).
- `scenarios`: change, additional sequestration vs BAU, rates, ordering check (with an explicit order for designs
  where a scenario lies below BAU), area-weighted summaries, totals in t C or t CO2.
- `sampling`: proportional and Neyman allocation with largest-remainder rounding; stratified draw and train/test split.
- `interp`: north-up grid, IDW, empirical variogram, WLS variogram fit, ordinary kriging with variance, leave-one-out CV.
- `validate`: bias, MAE, RMSE and R^2 against the 1:1 line, Pearson r, Lin's CCC.
- Tutorial notebook `tutorials/01_soc_mapping_walkthrough.ipynb` (sampling, regression kriging, cross-validation,
  scenario totals), executed by the test suite.

### Fixed (compared with the thesis scripts)
- Interpolated rasters were written upside down (ascending y rows with a top-left origin) and offset by half a pixel; the grid now uses pixel centres and a north-up transform.
- RMSE was computed against a regression fitted to the same data; it is now computed against the 1:1 line.
- Sample allocation rounded per cluster and then corrected only the largest cluster; it now uses the largest-remainder method.
- Group means were unweighted; summaries are now area-weighted.
- Relative changes divided by zero or negative stocks; these are now NaN.
- The plausibility filter on the initial stock was hard-coded, with different values in different scripts; it is now an explicit parameter.
- Scenario labels came from two differently truncated shapefiles; mixed inputs now raise an error.
