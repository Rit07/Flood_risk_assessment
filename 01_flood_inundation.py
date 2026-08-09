"""
01_flood_inundation.py
======================

Module 09 - Flood Risk Assessment · DEM-based inundation modelling.

Given a Digital Elevation Model (DEM), estimates which cells are flooded for a
given water level. Two complementary approaches are implemented from scratch:

    bathtub (planar)     — every cell below the water surface is wet. Simple,
                           fast, but floods hydrologically disconnected pits.
    connected (flood-fill) — only cells below the water surface that are
                           hydrologically connected to a water source (river
                           cell / map edge) are flooded. More realistic.

Also derives a terrain-following water depth grid and a simple slope layer
(reusing the gradient idea from module 04) used later for hazard scoring.

Public functions
----------------
    bathtub_inundation(dem, level, nodata=None)              -> (mask, depth)
    connected_inundation(dem, level, sources=None, nodata=None,
                          connectivity=8)                    -> (mask, depth)
    slope(dem, cellsize=1.0)                                 -> slope array (deg)
    water_depth(dem, mask, level)                            -> depth array

`dem` is a 2-D float array (metres). Returned masks are boolean (H, W);
depth arrays are float metres, 0 where dry.

Dependencies: numpy
"""

from __future__ import annotations

from collections import deque

import numpy as np


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #
def _as_dem(dem) -> np.ndarray:
    arr = np.asarray(dem, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("DEM must be a 2-D array of elevations.")
    return arr


def _nodata_mask(dem: np.ndarray, nodata) -> np.ndarray:
    if nodata is None:
        return np.zeros(dem.shape, dtype=bool)
    if np.isnan(nodata):
        return np.isnan(dem)
    return dem == nodata


# --------------------------------------------------------------------------- #
# Bathtub (planar) inundation
# --------------------------------------------------------------------------- #
def bathtub_inundation(dem, level: float, nodata=None):
    """
    Flood every valid cell whose elevation is below `level`.

    Returns (mask, depth). Ignores connectivity, so isolated low pits fill too.
    """
    dem = _as_dem(dem)
    invalid = _nodata_mask(dem, nodata)
    mask = (dem < level) & ~invalid
    depth = np.where(mask, level - dem, 0.0)
    return mask, depth


# --------------------------------------------------------------------------- #
# Hydrologically connected inundation (BFS flood fill)
# --------------------------------------------------------------------------- #
def _neighbours(connectivity: int):
    if connectivity == 4:
        return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    return [(-1, -1), (-1, 0), (-1, 1), (0, -1),
            (0, 1), (1, -1), (1, 0), (1, 1)]


def connected_inundation(dem, level: float, sources=None, nodata=None,
                         connectivity: int = 8):
    """
    Flood only cells below `level` that are connected to a water source.

    `sources` is an optional boolean array marking seed cells (e.g. a river
    channel). If None, all map-edge cells below `level` seed the flood — water
    is assumed to enter from the domain boundary.

    Returns (mask, depth).
    """
    dem = _as_dem(dem)
    h, w = dem.shape
    invalid = _nodata_mask(dem, nodata)
    below = (dem < level) & ~invalid

    mask = np.zeros((h, w), dtype=bool)
    q: deque = deque()

    if sources is not None:
        seeds = np.asarray(sources, dtype=bool)
        if seeds.shape != dem.shape:
            raise ValueError("sources must match DEM shape.")
        seeds = seeds & below
        ys, xs = np.nonzero(seeds)
        for y, x in zip(ys, xs):
            mask[y, x] = True
            q.append((y, x))
    else:
        # seed from every below-level cell on the border
        for x in range(w):
            for y in (0, h - 1):
                if below[y, x] and not mask[y, x]:
                    mask[y, x] = True
                    q.append((y, x))
        for y in range(h):
            for x in (0, w - 1):
                if below[y, x] and not mask[y, x]:
                    mask[y, x] = True
                    q.append((y, x))

    nbrs = _neighbours(connectivity)
    while q:
        y, x = q.popleft()
        for dy, dx in nbrs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and below[ny, nx] and not mask[ny, nx]:
                mask[ny, nx] = True
                q.append((ny, nx))

    depth = np.where(mask, level - dem, 0.0)
    return mask, depth


# --------------------------------------------------------------------------- #
# Water depth for an arbitrary mask
# --------------------------------------------------------------------------- #
def water_depth(dem, mask, level: float) -> np.ndarray:
    """Positive water depth (metres) where mask is True and below level."""
    dem = _as_dem(dem)
    mask = np.asarray(mask, dtype=bool)
    depth = np.where(mask, level - dem, 0.0)
    return np.clip(depth, 0.0, None)


# --------------------------------------------------------------------------- #
# Slope (first-derivative terrain gradient, degrees)
# --------------------------------------------------------------------------- #
def slope(dem, cellsize: float = 1.0) -> np.ndarray:
    """
    Slope in degrees via a 3x3 Horn gradient (same finite-difference idea as
    the Sobel operator in module 04, applied to elevation).
    """
    dem = _as_dem(dem)
    p = np.pad(dem, 1, mode="edge")
    # Horn's method weights
    dz_dx = ((p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) -
             (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])) / (8 * cellsize)
    dz_dy = ((p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) -
             (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])) / (8 * cellsize)
    grade = np.sqrt(dz_dx ** 2 + dz_dy ** 2)
    return np.degrees(np.arctan(grade))


# --------------------------------------------------------------------------- #
# File-based entry point (GeoTIFF DEM + AOI shapefile via io_geo)
# --------------------------------------------------------------------------- #
def run_from_files(dem_path, aoi_path, level, resolution=30.0,
                   river_path=None, target_crs=None, out_dir=None):
    """
    Full flood-inundation workflow from real files.

    dem_path   : GeoTIFF DEM (elevation in metres).
    aoi_path   : AOI boundary shapefile / GeoPackage / GeoJSON.
    level      : water-surface elevation in metres.
    resolution : output cell size in CRS units (default 30 m).
    river_path : optional vector layer seeding the connected flood. If omitted,
                 water enters from the AOI edge.
    target_crs : projected CRS to work in (e.g. 'EPSG:32644'). Strongly advised
                 if your data is in lat/lon so depth/area come out in metres.
    out_dir    : if given, writes flood_mask.tif / flood_depth.tif / slope.tif.

    Returns a dict with the arrays, the AOI mask, and the reference grid.
    """
    import io_geo as io

    grid, aoi = io.grid_from_aoi(aoi_path, resolution=resolution,
                                 target_crs=target_crs)
    dem_arr = io.read_aligned(dem_path, grid, resampling="bilinear")
    mask_aoi = io.aoi_mask(grid, aoi)

    sources = None
    if river_path is not None:
        sources = io.rasterize_vector(river_path, grid, target_crs=target_crs)
        sources = sources & mask_aoi

    # NaN outside AOI or in DEM nodata; treat those as invalid (never flooded).
    dem_work = np.where(mask_aoi, dem_arr, np.nan)
    wet, depth = connected_inundation(dem_work, level, sources=sources,
                                      nodata=np.nan)
    wet &= mask_aoi
    depth = np.where(wet, depth, 0.0)
    slp = slope(np.nan_to_num(dem_work, nan=float(np.nanmax(dem_work))),
                cellsize=resolution)

    result = {"grid": grid, "aoi_mask": mask_aoi, "dem": dem_arr,
              "flood_mask": wet, "depth": depth, "slope": slp}

    if out_dir is not None:
        import os
        os.makedirs(out_dir, exist_ok=True)
        io.write_geotiff(os.path.join(out_dir, "flood_mask.tif"),
                         wet.astype("int32"), grid, nodata=0, dtype="int32")
        io.write_geotiff(os.path.join(out_dir, "flood_depth.tif"),
                         depth, grid, nodata=-9999.0)
        io.write_geotiff(os.path.join(out_dir, "slope.tif"),
                         slp, grid, nodata=-9999.0)
    return result


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Synthetic valley: a channel down the middle plus an isolated pit.
    h, w = 40, 60
    yy, xx = np.mgrid[0:h, 0:w]
    dem = 4.0 + 0.15 * np.abs(xx - w / 2)           # V-shaped valley (floor ~4 m)
    dem += 0.05 * yy                                # gentle downstream slope
    dem[5:8, 45:48] = 2.0                           # isolated low pit (dry-land)

    level = 6.0
    bt_mask, bt_depth = bathtub_inundation(dem, level)
    cn_mask, cn_depth = connected_inundation(dem, level)

    print(f"bathtub  wet cells: {int(bt_mask.sum())}")
    print(f"connected wet cells: {int(cn_mask.sum())}")
    # The isolated pit should be wet under bathtub but dry under connected.
    assert bt_mask[6, 46] and not cn_mask[6, 46], "pit should be excluded when connected"
    assert cn_mask.sum() < bt_mask.sum()
    assert cn_depth.max() <= level

    s = slope(dem, cellsize=30.0)
    assert s.shape == dem.shape and s.min() >= 0
    print(f"slope range: {s.min():.2f}-{s.max():.2f} deg")

    print("\nFlood inundation modelling ran successfully.")
