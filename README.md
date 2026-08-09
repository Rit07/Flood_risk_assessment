# Flood Risk Assessment

A raster-based flood risk assessment pipeline that runs on real GeoTIFF imagery
and a shapefile Area-of-Interest (AOI). It maps flood inundation from a DEM,
extracts flood extent from before/after satellite scenes, and combines the
result with land cover into a multi-criteria risk map — all clipped to your AOI
and written back out as georeferenced GeoTIFFs you can open in QGIS or ArcGIS.

This module is the applications capstone of the wider remote-sensing pipeline:
the flood algorithms reuse the same ideas built from scratch in earlier modules
(spectral indices, Otsu thresholding, morphology, change detection, terrain
gradients). The only part that leans on external libraries is file I/O, which is
isolated in a single module (`io_geo.py`).

---

## What it does

| Stage | Script | Input | Output |
|-------|--------|-------|--------|
| 1. Inundation | `01_flood_inundation.py` | DEM GeoTIFF + AOI (+ optional river) | flood mask, water depth, slope |
| 2. Flood extent | `02_water_extraction.py` | before/after imagery GeoTIFFs + AOI | permanent / flooded / receded water masks + areas |
| 3. Risk mapping | `03_risk_scoring.py` | depth + slope + land-cover GeoTIFFs + AOI | continuous risk index + 5-class risk map |
| I/O layer | `io_geo.py` | — | reads/aligns/clips/writes all rasters & vectors |

The three stages can be run independently or chained: stage 1 writes the depth
and slope GeoTIFFs that stage 3 consumes.

---

## Inputs & assumptions

**Rasters** — supply as **GeoTIFF** (`.tif`). A DEM in metres for stage 1;
multi-band optical or single-band SAR imagery for stage 2; an integer
land-cover class raster for stage 3.

**AOI boundary** — supply as a **shapefile** (`.shp` with its sidecar
`.dbf/.shx/.prj`), GeoPackage, or GeoJSON containing the AOI polygon(s). The AOI
defines the working grid: its bounding box sets the extent, and everything
outside the polygon is masked to nodata.

**Coordinate systems and resolution are handled for you.** You do **not** need
your layers to already match. `io_geo` defines one reference grid from the AOI
and reprojects + resamples every raster onto it, so all layers line up
cell-for-cell. Pass a projected CRS via `target_crs` (e.g. `"EPSG:32644"`) so
that depth is in metres and areas in m²/km² — if your data is in lat/lon
degrees and you skip this, distances and areas will not be meaningful.

Resampling is chosen per layer: **bilinear** for continuous data (DEM,
reflectance) and **nearest** for the categorical land-cover raster so class
codes are never averaged into nonsense values.

---

## Installation

```bash
pip install numpy rasterio geopandas shapely
```

`rasterio` and `geopandas` bring GDAL/PROJ with them. On a clean system a conda
environment (`conda install -c conda-forge rasterio geopandas`) is often the
least painful route.

---

## Quick start

Run each script's built-in demo (synthetic data, no files needed) to confirm the
install works:

```bash
python 01_flood_inundation.py
python 02_water_extraction.py
python 03_risk_scoring.py
python io_geo.py
```

### Stage 1 — flood inundation from a DEM

```python
import sys; sys.path.insert(0, ".")   # so `import io_geo` resolves
import importlib.util

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

inund = load("inund", "01_flood_inundation.py")

result = inund.run_from_files(
    dem_path="data/dem.tif",
    aoi_path="data/aoi.shp",
    level=6.0,                    # water-surface elevation, metres
    resolution=30.0,             # output cell size, metres
    river_path="data/river.shp", # optional: seeds the connected flood
    target_crs="EPSG:32644",     # projected CRS (metres)
    out_dir="outputs",           # writes flood_mask/flood_depth/slope .tif
)
print(result["depth"].max())
```

If `river_path` is omitted, water is assumed to enter from the AOI edge. The
**connected** flood-fill only wets cells hydrologically linked to a source, so
isolated low pits are not wrongly flooded (unlike a naïve "bathtub" fill, which
is also available as `bathtub_inundation`).

### Stage 2 — flood extent from before/after imagery

```python
water = load("water", "02_water_extraction.py")

result = water.run_from_files(
    before_path="data/pre_flood.tif",
    after_path="data/post_flood.tif",
    aoi_path="data/aoi.shp",
    bands={"green": 2, "nir": 4},   # 1-based band indices in your GeoTIFF
    sensor="optical",               # or "sar" with {"backscatter": 1}
    resolution=30.0,
    target_crs="EPSG:32644",
    out_dir="outputs",
)
print(result["stats"]["flooded"]["area_km2"])
```

For sharper water detection over built-up areas use MNDWI by passing a SWIR band
instead of NIR: `bands={"green": 2, "swir": 5}`. For SAR data, water shows as
low backscatter and is thresholded automatically.

### Stage 3 — multi-criteria risk

```python
risk = load("risk", "03_risk_scoring.py")

result = risk.run_from_files(
    depth_path="outputs/flood_depth.tif",   # from stage 1
    slope_path="outputs/slope.tif",         # from stage 1
    landcover_path="data/landcover.tif",    # integer class codes
    aoi_path="data/aoi.shp",
    exposure_weights={0: 0.0, 1: 0.2, 2: 0.6, 3: 1.0},
    vuln_weights=    {0: 0.0, 1: 0.3, 2: 0.7, 3: 0.9},
    resolution=30.0,
    target_crs="EPSG:32644",
    out_dir="outputs",
)
print(result["summary"])   # area per risk class
```

The weight dictionary keys must match the integer class codes in your
land-cover raster (here: 0=water, 1=vegetation, 2=cropland, 3=urban — change to
match your data).

---

## How risk is computed

```
RISK  =  HAZARD  ×  EXPOSURE  ×  VULNERABILITY
```

- **Hazard** rises with flood depth and falls with slope (flat ground holds
  water, so it scores higher). Dry cells carry zero hazard.
- **Exposure** comes from land cover — how much of value sits in the wet
  footprint (urban high, open water zero).
- **Vulnerability** weights how badly each land-cover class is harmed by
  flooding.

The three factors are normalised to 0–1 and combined with a **weighted
geometric mean**, so a near-zero factor (e.g. nothing exposed) pulls risk toward
zero — which a plain weighted sum would not. The continuous index is then binned
into five classes: very low, low, moderate, high, very high.

---

## Changing the Area of Interest

The AOI is entirely file-driven: swap the `aoi_path` shapefile and the whole
pipeline re-clips, re-grids, and re-scores to the new region. No code edits are
needed to change study areas. To change resolution, adjust the `resolution`
argument (in CRS units) — every layer is resampled to match.

---

## Outputs

Each stage, when given `out_dir`, writes GeoTIFFs that carry the reference
grid's CRS and transform, so they overlay correctly on your source data in any
GIS:

- `flood_mask.tif`, `flood_depth.tif`, `slope.tif`
- `flooded.tif`, `permanent_water.tif`, `receded.tif`, `total_water.tif`
- `risk_index.tif`, `risk_class.tif`

Nodata is preserved so masked (outside-AOI) cells display as empty rather than
zero.

---

## Design notes

- **From-scratch algorithms, library-backed I/O.** The flood science (Otsu,
  morphology, flood-fill, gradients, change detection, risk math) is implemented
  by hand and depends only on NumPy. `rasterio`/`geopandas` are used *only* in
  `io_geo.py` for reading, warping, clipping, and writing geospatial formats —
  the parts where rolling your own would add risk without insight.
- **One grid, defined once.** All alignment flows from `grid_from_aoi`; every
  raster is warped to that single grid, which is what guarantees the layers are
  co-registered before any math runs.
- **Array functions stay pure.** The original array-based functions in each
  script are untouched and still usable directly if you already have aligned
  NumPy arrays; `run_from_files` is a thin file-driven wrapper around them.

---

## Limitations

- The inundation model is a level-based (planar / connected) approximation, not
  a hydrodynamic simulation — it answers "what floods at water level *h*", not
  flow velocity or timing.
- Optical water detection is limited by cloud cover; use the SAR path for
  cloudy flood events.
- Risk weights are user-supplied and scenario-dependent; calibrate them to your
  region and asset data rather than treating the defaults as authoritative.
