"""GeoTIFF writing (skipped when rasterio is not installed)."""

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from soilcarbon import interp  # noqa: E402


def test_write_geotiff_is_north_up(tmp_path):
    g = interp.grid((1000, 2000, 1040, 2030), 10)  # 3 rows x 4 cols
    arr = np.arange(12, dtype=float).reshape(g.shape)  # row 0 = north
    path = tmp_path / "t.tif"
    interp.write_geotiff(path, arr, g, "EPSG:2100")
    with rasterio.open(path) as src:
        assert src.crs.to_epsg() == 2100 and src.res == (10.0, 10.0)
        assert src.bounds.top == 2030 and src.bounds.left == 1000
        # sampling the centre of the north-west pixel returns array[0, 0]
        assert next(src.sample([(g.x[0], g.y[0])]))[0] == 0.0
        assert next(src.sample([(g.x[-1], g.y[-1])]))[0] == 11.0
