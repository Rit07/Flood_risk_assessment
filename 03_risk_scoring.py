"""
03_risk_scoring.py
==================

Module 09 - Flood Risk Assessment · multi-criteria risk mapping.

Combines the physical flood picture with what is exposed and how vulnerable it
is, following the standard framing:

        RISK  =  HAZARD  x  EXPOSURE  x  VULNERABILITY

Hazard comes from flood depth and how fast water is likely to accumulate
(low slope + low elevation = worse). Exposure comes from land cover
(module 08 classification) — people and assets in the wet footprint.
Vulnerability weights each land-cover class by how badly flooding hurts it.

Everything is normalised to 0-1 and combined with a weighted geometric mean so
that a near-zero factor (e.g. no exposure) drives risk down, unlike a plain
weighted sum.

Public functions
----------------
    normalize(arr, lo=None, hi=None, invert=False)      -> array in [0,1]
    hazard_score(depth, slope, w_depth=0.7, w_slope=0.3) -> array [0,1]
    exposure_score(landcover, weights)                   -> array [0,1]
    vulnerability_score(landcover, weights)              -> array [0,1]
    risk_index(hazard, exposure, vulnerability,
               weights=(0.5,0.25,0.25))                  -> array [0,1]
    classify_risk(risk, bins=(0.2,0.4,0.6,0.8))          -> int class map 0-4
    risk_summary(risk_classes, cell_area_m2=1.0)         -> dict

Dependencies: numpy
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
def normalize(arr, lo=None, hi=None, invert: bool = False) -> np.ndarray:
    """Min-max scale to [0, 1]. `invert=True` flips (high input -> low score)."""
    a = np.asarray(arr, dtype=np.float64)
    lo = float(np.nanmin(a)) if lo is None else float(lo)
    hi = float(np.nanmax(a)) if hi is None else float(hi)
    if hi == lo:
        out = np.zeros_like(a)
    else:
        out = (a - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    return 1.0 - out if invert else out


# --------------------------------------------------------------------------- #
# Hazard
# --------------------------------------------------------------------------- #
def hazard_score(depth, slope, w_depth: float = 0.7,
                 w_slope: float = 0.3) -> np.ndarray:
    """
    Combine water depth (deeper = worse) with slope (flatter = water pools and
    lingers, so worse). Slope is inverted before combining.
    """
    depth = np.asarray(depth, dtype=np.float64)
    slope = np.asarray(slope, dtype=np.float64)
    if depth.shape != slope.shape:
        raise ValueError("depth and slope must share shape.")
    d = normalize(depth, lo=0.0)
    s = normalize(slope, invert=True)               # flat -> high hazard
    total = w_depth + w_slope
    haz = (w_depth * d + w_slope * s) / total
    haz[depth <= 0] = 0.0                            # dry cells carry no hazard
    return np.clip(haz, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Exposure & vulnerability from land cover
# --------------------------------------------------------------------------- #
def _map_weights(landcover, weights: dict) -> np.ndarray:
    lc = np.asarray(landcover)
    out = np.zeros(lc.shape, dtype=np.float64)
    for cls, val in weights.items():
        out[lc == cls] = float(val)
    return out


def exposure_score(landcover, weights: dict) -> np.ndarray:
    """
    Per-class exposure weight (e.g. urban=1.0, cropland=0.6, water=0.0),
    normalised to [0,1] by the max supplied weight.
    """
    raw = _map_weights(landcover, weights)
    hi = max(weights.values()) if weights else 1.0
    return normalize(raw, lo=0.0, hi=hi if hi > 0 else 1.0)


def vulnerability_score(landcover, weights: dict) -> np.ndarray:
    """Per-class vulnerability weight, normalised to [0,1]."""
    raw = _map_weights(landcover, weights)
    hi = max(weights.values()) if weights else 1.0
    return normalize(raw, lo=0.0, hi=hi if hi > 0 else 1.0)


# --------------------------------------------------------------------------- #
# Combined risk
# --------------------------------------------------------------------------- #
def risk_index(hazard, exposure, vulnerability,
               weights=(0.5, 0.25, 0.25)) -> np.ndarray:
    """
    Weighted geometric mean of the three factors, each in [0,1]. A geometric
    mean makes risk collapse toward 0 when any factor is ~0 (e.g. nothing
    exposed), which a weighted sum would not.
    """
    h = np.clip(np.asarray(hazard, dtype=np.float64), 0.0, 1.0)
    e = np.clip(np.asarray(exposure, dtype=np.float64), 0.0, 1.0)
    v = np.clip(np.asarray(vulnerability, dtype=np.float64), 0.0, 1.0)
    if not (h.shape == e.shape == v.shape):
        raise ValueError("hazard, exposure, vulnerability must share shape.")
    wh, we, wv = weights
    tot = wh + we + wv
    eps = 1e-9
    log_r = (wh * np.log(h + eps) + we * np.log(e + eps) +
             wv * np.log(v + eps)) / tot
    return np.clip(np.exp(log_r), 0.0, 1.0)


def classify_risk(risk, bins=(0.2, 0.4, 0.6, 0.8)) -> np.ndarray:
    """Bin the continuous risk index into 0-4 (very low -> very high)."""
    r = np.asarray(risk, dtype=np.float64)
    return np.digitize(r, np.asarray(bins, dtype=np.float64)).astype(np.int32)


def risk_summary(risk_classes, cell_area_m2: float = 1.0) -> dict:
    """Area per risk class (0-4)."""
    labels = ["very_low", "low", "moderate", "high", "very_high"]
    rc = np.asarray(risk_classes, dtype=np.int32)
    summary = {}
    for k, name in enumerate(labels):
        px = int((rc == k).sum())
        area = px * cell_area_m2
        summary[name] = {"pixels": px, "area_km2": area / 1e6}
    return summary


# --------------------------------------------------------------------------- #
# File-based entry point (depth/slope/land-cover GeoTIFFs + AOI via io_geo)
# --------------------------------------------------------------------------- #
def run_from_files(depth_path, slope_path, landcover_path, aoi_path,
                   exposure_weights, vuln_weights, resolution=30.0,
                   target_crs=None, out_dir=None):
    """
    Multi-criteria risk mapping from real files.

    depth_path     : GeoTIFF of flood depth (e.g. from 01's flood_depth.tif).
    slope_path     : GeoTIFF of slope in degrees (e.g. 01's slope.tif).
    landcover_path : GeoTIFF of integer land-cover class codes. Read with
                     NEAREST resampling so class codes are never averaged.
    aoi_path       : AOI boundary vector file.
    exposure_weights, vuln_weights : {class_code: weight} dicts. Keys must
                     match the codes in the land-cover raster.
    resolution     : output cell size in CRS units.
    target_crs     : projected CRS to work in (recommended).
    out_dir        : if given, writes risk_index.tif and risk_class.tif.

    Returns dict with risk arrays, class map, area summary, and grid.
    """
    import io_geo as io

    grid, aoi = io.grid_from_aoi(aoi_path, resolution=resolution,
                                 target_crs=target_crs)
    mask_aoi = io.aoi_mask(grid, aoi)

    depth = io.read_aligned(depth_path, grid, resampling="bilinear")
    slp = io.read_aligned(slope_path, grid, resampling="bilinear")
    landcover = io.read_aligned(landcover_path, grid, resampling="nearest")

    depth = np.nan_to_num(depth, nan=0.0)
    slp = np.nan_to_num(slp, nan=0.0)
    landcover = np.nan_to_num(landcover, nan=-1).astype(np.int32)

    haz = hazard_score(depth, slp)
    exp = exposure_score(landcover, exposure_weights)
    vul = vulnerability_score(landcover, vuln_weights)
    risk = risk_index(haz, exp, vul)
    risk[~mask_aoi] = 0.0
    classes = classify_risk(risk)
    classes[~mask_aoi] = 0
    summary = risk_summary(classes, cell_area_m2=grid.cell_area())

    if out_dir is not None:
        import os
        os.makedirs(out_dir, exist_ok=True)
        io.write_geotiff(os.path.join(out_dir, "risk_index.tif"),
                         risk, grid, nodata=-9999.0)
        io.write_geotiff(os.path.join(out_dir, "risk_class.tif"),
                         classes.astype("int32"), grid, nodata=0,
                         dtype="int32")

    return {"grid": grid, "aoi_mask": mask_aoi, "hazard": haz,
            "exposure": exp, "vulnerability": vul, "risk": risk,
            "risk_class": classes, "summary": summary}


# --------------------------------------------------------------------------- #
# Demo
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    rng = np.random.default_rng(7)
    h, w = 40, 40

    # Flood depth: deepest in a low corner, tapering out.
    yy, xx = np.mgrid[0:h, 0:w]
    depth = np.clip(3.0 - 0.08 * (xx + yy), 0.0, None)
    slope = np.abs(rng.normal(2.0, 1.0, (h, w)))    # mostly gentle terrain

    # Land cover: 0=water, 1=vegetation, 2=cropland, 3=urban.
    landcover = np.full((h, w), 1, dtype=np.int32)
    landcover[:, :5] = 0                             # river strip
    landcover[10:25, 10:25] = 2                      # cropland block
    landcover[6:16, 6:16] = 3                         # town (in the flooded corner)

    exposure_w = {0: 0.0, 1: 0.2, 2: 0.6, 3: 1.0}
    vuln_w = {0: 0.0, 1: 0.3, 2: 0.7, 3: 0.9}

    haz = hazard_score(depth, slope)
    exp = exposure_score(landcover, exposure_w)
    vul = vulnerability_score(landcover, vuln_w)
    risk = risk_index(haz, exp, vul)
    classes = classify_risk(risk)
    summary = risk_summary(classes, cell_area_m2=30 * 30)

    assert haz.shape == risk.shape == (h, w)
    assert risk.min() >= 0 and risk.max() <= 1
    # Water cells have zero exposure -> ~zero risk despite being wet.
    assert risk[5, 2] < 0.05, "open water should score near-zero risk"
    # At comparable hazard, urban must out-risk vegetation (higher exp+vuln).
    urban = landcover == 3
    veg = landcover == 1
    band = (depth > 0.1) & (depth < 1.5)            # comparable-depth cells
    town_risk = risk[urban & band].mean()
    veg_risk = risk[veg & band].mean()
    assert town_risk > veg_risk

    print("risk index range:", round(float(risk.min()), 3),
          "-", round(float(risk.max()), 3))
    for name, s in summary.items():
        print(f"  {name:10s}: {s['pixels']:4d} px  {s['area_km2']:.3f} km2")

    print("\nFlood risk scoring ran successfully.")
