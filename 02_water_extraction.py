"""
02_water_extraction.py
======================

Module 09 - Flood Risk Assessment · water extraction & flood-extent mapping.

Detects open water from imagery and maps flood extent by differencing a
"before" and "after" (flood) scene — the standard remote-sensing flood-mapping
workflow. Composes several earlier building blocks:

    * spectral indices (module 07)   -> NDWI / MNDWI water indices.
    * Otsu thresholding (module 06)  -> automatic water/land split.
    * morphology (module 05)         -> clean speckle, fill gaps.
    * change detection (module 08)   -> new water = flooded area.

Two index paths are supported so this works for optical multiband data and for
single-band SAR backscatter (where low backscatter = smooth water).

Public functions
----------------
    ndwi(green, nir)                              -> index array [-1, 1]
    mndwi(green, swir)                            -> index array [-1, 1]
    otsu_threshold(values)                        -> float threshold
    water_mask_optical(green, nir=None, swir=None,
                       clean=True)                -> boolean mask
    water_mask_sar(backscatter_db, clean=True)    -> boolean mask
    flood_extent(before_mask, after_mask)         -> dict of masks
    flood_stats(flood_masks, cell_area_m2=1.0)    -> dict of areas

Dependencies: numpy
"""

from __future__ import annotations

from collections import deque

import numpy as np


# --------------------------------------------------------------------------- #
# Band helpers
# --------------------------------------------------------------------------- #
def _band(x) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("Expected a 2-D band array.")
    return arr


def _norm_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = a + b
    out = np.zeros_like(a, dtype=np.float64)
    nz = denom != 0
    out[nz] = (a[nz] - b[nz]) / denom[nz]
    return np.clip(out, -1.0, 1.0)


def ndwi(green, nir) -> np.ndarray:
    """Normalised Difference Water Index (McFeeters): (Green - NIR)/(Green + NIR)."""
    return _norm_diff(_band(green), _band(nir))


def mndwi(green, swir) -> np.ndarray:
    """Modified NDWI (Xu): (Green - SWIR)/(Green + SWIR). Better vs. built-up."""
    return _norm_diff(_band(green), _band(swir))


# --------------------------------------------------------------------------- #
# Otsu threshold (from scratch)
# --------------------------------------------------------------------------- #
def otsu_threshold(values) -> float:
    """
    Otsu's method on a flat array of finite values. Returns the threshold that
    maximises between-class variance.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0
    lo, hi = v.min(), v.max()
    if lo == hi:
        return float(lo)
    hist, edges = np.histogram(v, bins=256, range=(lo, hi))
    centers = (edges[:-1] + edges[1:]) / 2
    total = hist.sum()
    w0 = np.cumsum(hist)
    w1 = total - w0
    valid = (w0 > 0) & (w1 > 0)
    cumsum = np.cumsum(hist * centers)
    grand = cumsum[-1]
    mu0 = np.zeros_like(centers)
    mu1 = np.zeros_like(centers)
    mu0[valid] = cumsum[valid] / w0[valid]
    mu1[valid] = (grand - cumsum[valid]) / w1[valid]
    between = np.zeros_like(centers)
    between[valid] = w0[valid] * w1[valid] * (mu0[valid] - mu1[valid]) ** 2
    return float(centers[int(np.argmax(between))])


# --------------------------------------------------------------------------- #
# Minimal morphology (open + close) for mask cleanup
# --------------------------------------------------------------------------- #
def _morph(binary: np.ndarray, op: str, iterations: int = 1) -> np.ndarray:
    h, w = binary.shape
    out = binary.copy()
    for _ in range(iterations):
        p = np.pad(out, 1, constant_values=(op == "erode"))
        acc = None
        for di in range(3):
            for dj in range(3):
                s = p[di:di + h, dj:dj + w]
                acc = s if acc is None else (
                    np.logical_and(acc, s) if op == "erode"
                    else np.logical_or(acc, s))
        out = acc
    return out


def _clean(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Opening then closing: drop speckle, then fill small holes."""
    opened = _morph(_morph(mask, "erode", iterations), "dilate", iterations)
    closed = _morph(_morph(opened, "dilate", iterations), "erode", iterations)
    return closed


# --------------------------------------------------------------------------- #
# Water masks
# --------------------------------------------------------------------------- #
def water_mask_optical(green, nir=None, swir=None, clean: bool = True) -> np.ndarray:
    """
    Water mask from optical bands. Uses MNDWI if SWIR is supplied (more robust),
    otherwise NDWI from green/NIR. Threshold chosen automatically with Otsu but
    forced to be non-negative (water indices are positive over water).
    """
    if swir is not None:
        index = mndwi(green, swir)
    elif nir is not None:
        index = ndwi(green, nir)
    else:
        raise ValueError("Provide either nir (for NDWI) or swir (for MNDWI).")
    t = max(otsu_threshold(index), 0.0)
    mask = index > t
    return _clean(mask) if clean else mask


def water_mask_sar(backscatter_db, clean: bool = True) -> np.ndarray:
    """
    Water mask from SAR backscatter (dB). Smooth water surfaces reflect energy
    away from the sensor -> low backscatter, so water is *below* the threshold.
    """
    b = _band(backscatter_db)
    t = otsu_threshold(b)
    mask = b < t
    return _clean(mask) if clean else mask


# --------------------------------------------------------------------------- #
# Flood extent from before/after water masks
# --------------------------------------------------------------------------- #
def flood_extent(before_mask, after_mask) -> dict:
    """
    Compare permanent (before) water to flood-time (after) water.

    Returns dict of boolean masks:
        permanent_water — wet before and after
        flooded         — dry before, wet after (new inundation)
        receded         — wet before, dry after
        total_water     — wet after
    """
    b = np.asarray(before_mask, dtype=bool)
    a = np.asarray(after_mask, dtype=bool)
    if b.shape != a.shape:
        raise ValueError("before/after masks must have the same shape.")
    return {
        "permanent_water": b & a,
        "flooded": ~b & a,
        "receded": b & ~a,
        "total_water": a,
    }


def flood_stats(flood_masks: dict, cell_area_m2: float = 1.0) -> dict:
    """Convert each mask's pixel count to area (m^2 and km^2)."""
    stats = {}
    for name, m in flood_masks.items():
        px = int(np.asarray(m, dtype=bool).sum())
        area = px * cell_area_m2
        stats[name] = {"pixels": px, "area_m2": area, "area_km2": area / 1e6}
    return stats


# --------------------------------------------------------------------------- #
# File-based entry point (GeoTIFF imagery + AOI shapefile via io_geo)
# --------------------------------------------------------------------------- #
def run_from_files(before_path, after_path, aoi_path, bands,
                   resolution=30.0, target_crs=None, sensor="optical",
                   out_dir=None):
    """
    Flood-extent mapping from before/after GeoTIFF scenes.

    before_path, after_path : co-registered GeoTIFFs (pre- and post-flood).
    aoi_path   : AOI boundary vector file.
    bands      : dict mapping the names this function needs to 1-based band
                 indices in the GeoTIFFs, e.g. {'green': 2, 'nir': 4} for
                 optical NDWI, {'green': 2, 'swir': 5} for MNDWI, or
                 {'backscatter': 1} for SAR.
    sensor     : 'optical' or 'sar'.
    resolution : output cell size in CRS units.
    target_crs : projected CRS to work in (recommended).
    out_dir    : if given, writes flooded.tif / total_water.tif etc.

    Returns dict with per-class masks, area stats, AOI mask, and grid.
    """
    import io_geo as io

    grid, aoi = io.grid_from_aoi(aoi_path, resolution=resolution,
                                 target_crs=target_crs)
    mask_aoi = io.aoi_mask(grid, aoi)

    def _mask(path):
        layers = io.read_multiband(path, grid, bands=bands,
                                   resampling="bilinear")
        if sensor == "sar":
            m = water_mask_sar(layers["backscatter"])
        else:
            m = water_mask_optical(layers.get("green"),
                                   nir=layers.get("nir"),
                                   swir=layers.get("swir"))
        return m & mask_aoi

    before = _mask(before_path)
    after = _mask(after_path)
    masks = flood_extent(before, after)
    stats = flood_stats(masks, cell_area_m2=grid.cell_area())

    if out_dir is not None:
        import os
        os.makedirs(out_dir, exist_ok=True)
        for name, m in masks.items():
            io.write_geotiff(os.path.join(out_dir, f"{name}.tif"),
                             m.astype("int32"), grid, nodata=0, dtype="int32")

    return {"grid": grid, "aoi_mask": mask_aoi, "masks": masks, "stats": stats}


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(3)
    h, w = 50, 50

    # "Before": a river channel (columns 23-26) is permanent water.
    green = np.full((h, w), 80.0)
    nir = np.full((h, w), 180.0)
    river = np.zeros((h, w), dtype=bool)
    river[:, 23:27] = True
    green[river] = 150.0                             # water: high green
    nir[river] = 40.0                                # water: low NIR
    green += rng.normal(0, 4, green.shape)
    nir += rng.normal(0, 4, nir.shape)
    before = water_mask_optical(green, nir=nir)

    # "After": river plus a flooded plain (rows 30-45) spreads out.
    green_a = green.copy()
    nir_a = nir.copy()
    flood_zone = np.zeros((h, w), dtype=bool)
    flood_zone[30:46, 10:40] = True
    green_a[flood_zone] = 150.0
    nir_a[flood_zone] = 40.0
    after = water_mask_optical(green_a, nir=nir_a)

    masks = flood_extent(before, after)
    stats = flood_stats(masks, cell_area_m2=30 * 30)

    print("before water px:", int(before.sum()))
    print("after  water px:", int(after.sum()))
    for name, s in stats.items():
        print(f"  {name:16s}: {s['pixels']:5d} px  {s['area_km2']:.3f} km2")

    assert stats["flooded"]["pixels"] > 0
    assert masks["permanent_water"].sum() > 0
    assert (masks["flooded"] & masks["permanent_water"]).sum() == 0

    # SAR path sanity check: low-backscatter blob = water.
    sar = rng.normal(-8, 1.5, (h, w))               # land ~ -8 dB
    sar[20:35, 20:35] = rng.normal(-18, 1.0, (15, 15))  # water ~ -18 dB
    sar_mask = water_mask_sar(sar)
    assert sar_mask[27, 27] and not sar_mask[2, 2]
    print("SAR water px:", int(sar_mask.sum()))

    print("\nWater extraction & flood mapping ran successfully.")
