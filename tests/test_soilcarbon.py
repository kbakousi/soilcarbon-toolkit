import numpy as np
import pandas as pd
import pytest

from soilcarbon import interp, io, sampling, scenarios, units, validate


def frame():
    return pd.DataFrame({
        "SOC_t0": [50.0, 60.0, 0.0],
        "SOC_BAU_20": [48.0, 61.0, 1.0],
        "Low_Scena0": [50.0, 62.0, 2.0],
        "Med_Scena0": [52.0, 63.0, 3.0],
        "High_Scen0": [55.0, 62.5, 4.0],
        "area_ha": [10.0, 30.0, 1.0],
        "zone": ["a", "b", "b"],
    })


def test_standardize_columns_and_ambiguity():
    df = io.standardize_columns(frame())
    assert {"soc_t0", "soc_bau", "soc_low", "soc_med", "soc_high"} <= set(df.columns)
    mixed = frame().assign(Med_Scenar=1.0)  # two different outputs mixed together
    with pytest.raises(ValueError, match="several matching"):
        io.standardize_columns(mixed)


def test_changes_definitions_and_zero_reference():
    df = scenarios.changes(io.standardize_columns(frame()), horizon_years=20)
    assert df.loc[0, "abs_change_soc_bau"] == -2.0
    assert df.loc[0, "delta_soc_high"] == pytest.approx(7.0)            # 55 - 48
    assert df.loc[0, "rel_change_soc_med"] == pytest.approx(4.0)        # 2/50 * 100
    assert df.loc[0, "delta_rate_soc_high"] == pytest.approx(7.0 / 20)
    assert np.isnan(df.loc[2, "rel_change_soc_low"])                    # t0 = 0 -> NaN, no division


def test_ordering_check_flags_swapped_labels():
    chk = scenarios.ordering_check(io.standardize_columns(frame()))
    assert chk["any_violation"].tolist() == [False, True, False]        # row 1: med 63 > high 62.5


def test_ordering_check_with_a_scenario_below_bau():
    df = pd.DataFrame({"soc_bau": [50.0, 50.0], "soc_low": [49.0, 51.0], "soc_med": [51.0, 52.0],
                       "soc_high": [58.0, 60.0]})
    assert scenarios.ordering_check(df)["any_violation"].tolist() == [True, False]  # default: BAU <= low
    chk = scenarios.ordering_check(df, order=["soc_low", "soc_bau", "soc_med", "soc_high"])
    assert chk["any_violation"].tolist() == [False, True]  # row 1: low 51 > BAU 50
    with pytest.raises(KeyError, match="soc_vhigh"):
        scenarios.ordering_check(df, order=["soc_bau", "soc_vhigh"])


def test_area_weighted_summary_and_totals():
    df = scenarios.changes(io.standardize_columns(frame()), horizon_years=20)
    s = scenarios.area_weighted_summary(df, ["delta_soc_high"], "area_ha", by="zone")
    assert s.loc["b", "delta_soc_high"] == pytest.approx((1.5 * 30 + 3 * 1) / 31)
    t = scenarios.totals(df, ["delta_soc_high"], "area_ha", co2=True)
    assert t.loc["all", "delta_soc_high"] == pytest.approx((7 * 10 + 1.5 * 30 + 3 * 1) * 44 / 12)
    assert units.c_to_co2(12) == pytest.approx(44)


def test_largest_remainder_is_exact_and_fair():
    shares = [0.334, 0.333, 0.333]
    a = sampling.largest_remainder(shares, 50)
    assert a.sum() == 50 and a.tolist() == [17, 17, 16]
    exact = 50 * np.array(shares) / sum(shares)
    assert np.all(np.abs(a - exact) < 1)
    # the "round, then fix the largest stratum" shortcut can move a stratum by more than one sample
    s = pd.Series({"a": 1, "b": 1, "c": 1, "d": 1, "e": 1, "f": 5})
    alloc = sampling.allocate(s, 9)
    assert alloc.sum() == 9 and np.all(np.abs(alloc - 9 * s / s.sum()) < 1)


def test_neyman_allocation_and_minimum():
    sizes = pd.Series({"a": 100, "b": 100})
    std = pd.Series({"a": 1.0, "b": 3.0})
    assert sampling.allocate(sizes, 40, "neyman", std).tolist() == [10, 30]
    # 15 each first; the remaining 10 split 1:3 = 2.5/7.5; the tie goes to the first stratum
    assert sampling.allocate(sizes, 40, "neyman", std, minimum=15).tolist() == [18, 22]
    with pytest.raises(ValueError):
        sampling.allocate(pd.Series({"a": 2, "b": 100}), 50, minimum=5)


def test_draw_and_split_stratified():
    pop = pd.DataFrame({"cluster": np.repeat([0, 1, 2], [60, 30, 10]), "v": np.arange(100)})
    alloc = sampling.allocate(pop["cluster"].value_counts().sort_index(), 20)
    smp = sampling.draw_stratified(pop, "cluster", alloc, seed=1)
    assert smp["cluster"].value_counts().sort_index().tolist() == alloc.tolist()
    lab = sampling.split_stratified(smp, "cluster", 0.2, seed=1)
    assert (lab == "test").sum() == 4
    assert set(smp.loc[lab == "train", "cluster"]) == {0, 1, 2}


def test_grid_is_north_up_with_corner_origin():
    g = interp.grid((0, 0, 100, 50), 10)
    assert g.shape == (5, 10)
    assert g.y[0] == 45 and g.y[-1] == 5          # row 0 is the northernmost row
    assert g.transform == (10, 0.0, 0, 0.0, -10, 50)
    # the centre of pixel (row 0, col 0) maps back through the transform
    a, b, c, d, e, f = g.transform
    assert (a * 0.5 + c, e * 0.5 + f) == (g.x[0], g.y[0])


def test_idw_exact_hit_and_north_south_orientation():
    xy = np.array([[0, 0], [0, 100.0]])
    z = np.array([1.0, 9.0])  # south = 1, north = 9
    assert interp.idw(xy, z, xy).tolist() == [1.0, 9.0]
    g = interp.grid((-50, -50, 50, 150), 50)
    surf = interp.idw(xy, z, g.points()).reshape(g.shape)
    assert surf[0].mean() > surf[-1].mean()       # northern rows hold the northern (high) values


def test_variogram_fit_and_kriging():
    rng = np.random.default_rng(5)
    xy = rng.uniform(0, 1000, (80, 2))
    z = 0.02 * xy[:, 0] + rng.normal(0, 1, 80)
    lags, gam, n = interp.empirical_variogram(xy, z, 10, 600)
    assert np.all(n > 0) and gam[-1] > gam[0]
    vg = interp.fit_variogram(lags, gam, n, "spherical")
    assert vg["nugget"] >= 0 and vg["psill"] >= 0 and vg["range"] > 0
    pred, var = interp.ordinary_kriging(xy, z, xy, vg)
    if vg["nugget"] < 1e-9:
        assert np.allclose(pred, z, atol=1e-6)    # exact interpolator without nugget
    assert np.all(var >= -1e-9)
    # weights sum to one: kriging a constant field returns the constant
    p_const, _ = interp.ordinary_kriging(xy, np.full(80, 7.0), np.array([[500.0, 500.0]]), vg)
    assert p_const[0] == pytest.approx(7.0)


def test_loocv_and_agreement():
    rng = np.random.default_rng(6)
    xy = rng.uniform(0, 1000, (40, 2))
    z = 0.01 * xy[:, 1] + rng.normal(0, 0.5, 40)
    pred = interp.loocv(xy, z, "idw", power=2)
    stats = validate.agreement(z, pred)
    assert stats["n"] == 40 and stats["rmse"] > 0
    perfect = validate.agreement([1, 2, 3, 4], [1, 2, 3, 4])
    assert perfect["rmse"] == 0 and perfect["ccc"] == pytest.approx(1) and perfect["r2_1to1"] == 1
    biased = validate.agreement([1, 2, 3, 4], [2, 3, 4, 5])
    assert biased["pearson_r"] == pytest.approx(1) and biased["ccc"] < 1 and biased["bias"] == 1
    assert biased["rmse"] == 1  # RMSE against the 1:1 line, not against a fitted regression
