"""
io_geo.py
=========

Module 09 - Flood Risk Assessment · geospatial I/O layer.

Bridges real-world GIS files and the array-based flood algorithms in this
module. Everything downstream (inundation, water extraction, risk scoring)
works on plain NumPy arrays that all share one grid; this file is what turns a
pile of GeoTIFFs and an AOI shapefile into that aligned stack.

Responsibilities
----------------
    * read GeoTIFF rasters (DEM, imagery bands) into arrays.
    * read an AOI boundary shapefile (or GeoPackage / GeoJSON).
    * define ONE common grid (CRS + resolution + bounds) from the AOI.
    * reproject / resample every raster onto that grid so all layers line up
      cell-for-cell.
    * clip everything to the AOI polygon and expose the AOI as a boolean mask.
    * write result arrays back out as GeoTIFFs that carry the same
      georeferencing, so they open correctly in QGIS / ArcGIS.

Design note
-----------
The flood *algorithms* in this repo are implemented from scratch. This file is
deliberately the exception: parsing GeoTIFF/shapefile formats and doing correct
CRS transforms is exactly what rasterio / geopandas exist for, and hand-rolling
them would add risk without teaching anything. Keeping all I/O in one module
also means the science code never imports a GIS library.

Dependencies: numpy, rasterio, geopandas, shapely
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import calculate_default_transform, reproject
from rasterio.features import geometry_mask
import geopandas as gpd


# --------------------------------------------------------------------------- #
# Grid definition
# --------------------------------------------------------------------------- #
@dataclass
class Grid:
    """The single reference grid every layer is aligned to.

    crs        : coordinate reference system (rasterio/pyproj CRS).
    transform  : affine transform mapping (row, col) -> (x, y).
    width      : number of columns.
    height     : number of rows.
    resolution : (x_res, y_res) cell size in CRS units (metres if projected).
    """
    crs: object
    transform: object
    width: int
    height: int
    resolution: tuple

    @property
    def shape(self) -> tuple:
        return (self.height, self.width)

    def cell_area(self) -> float:
        """Cell area in squared CRS units (m^2 for a projected CRS)."""
        xr, yr = self.resolution
        return abs(xr) * abs(yr)


def grid_from_aoi(aoi_path: str, resolution: float,
                  target_crs=None) -> tuple:
    """
    Build the reference grid from an AOI boundary file.

    Parameters
    ----------
    aoi_path   : path to a vector file (.shp / .gpkg / .geojson) holding the
                 area-of-interest polygon(s).
    resolution : desired cell size in target-CRS units (e.g. 30 for 30 m).
    target_crs : CRS to work in. If None, the AOI's own CRS is used. For flood
                 work you almost always want a *projected* CRS in metres, so
                 pass e.g. an EPSG code if the AOI is in lat/lon degrees.

    Returns
    -------
    (grid, aoi_gdf) where aoi_gdf is the AOI reprojected to the grid CRS.
    """
    aoi = gpd.read_file(aoi_path)
    if aoi.empty:
        raise ValueError(f"AOI file '{aoi_path}' contains no geometry.")
    if aoi.crs is None:
        raise ValueError(
            "AOI has no CRS. Assign one before use (aoi.set_crs(EPSG))."
        )

    if target_crs is not None:
        aoi = aoi.to_crs(target_crs)
    crs = aoi.crs

    minx, miny, maxx, maxy = aoi.total_bounds
    if resolution <= 0:
        raise ValueError("resolution must be positive.")

    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))
    if width < 1 or height < 1:
        raise ValueError(
            "AOI is smaller than one cell at this resolution; "
            "reduce `resolution`."
        )

    # Top-left origin; y resolution is negative (north-up).
    transform = from_origin(minx, maxy, resolution, resolution)
    grid = Grid(crs=crs, transform=transform, width=width, height=height,
                resolution=(resolution, resolution))
    return grid, aoi


# --------------------------------------------------------------------------- #
# Raster reading + alignment
# --------------------------------------------------------------------------- #
def read_aligned(raster_path: str, grid: Grid, band: int = 1,
                 resampling: str = "bilinear",
                 fill_value: float = np.nan) -> np.ndarray:
    """
    Read one band of a GeoTIFF and warp it onto `grid`.

    Reprojection + resampling happen together, so the returned array has exactly
    grid.shape and lines up cell-for-cell with every other layer read against
    the same grid.

    resampling : 'bilinear' or 'cubic' for continuous data (DEM, reflectance);
                 use 'nearest' for categorical rasters (e.g. a land-cover map)
                 so class codes are never averaged.
    """
    method = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "cubic": Resampling.cubic,
        "average": Resampling.average,
    }.get(resampling)
    if method is None:
        raise ValueError(f"Unknown resampling '{resampling}'.")

    dest = np.full(grid.shape, fill_value, dtype=np.float64)
    with rasterio.open(raster_path) as src:
        src_nodata = src.nodata
        reproject(
            source=rasterio.band(src, band),
            destination=dest,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=grid.transform,
            dst_crs=grid.crs,
            resampling=method,
            src_nodata=src_nodata,
            dst_nodata=fill_value,
        )
    return dest


def read_multiband(raster_path: str, grid: Grid, bands=None,
                   resampling: str = "bilinear") -> dict:
    """
    Read several bands of a GeoTIFF, each aligned to `grid`.

    `bands` is a dict mapping a friendly name to a 1-based band index, e.g.
    {'green': 2, 'nir': 4, 'swir': 5}. Returns {name: array}.
    """
    if bands is None:
        raise ValueError("Provide a {name: band_index} mapping.")
    return {name: read_aligned(raster_path, grid, band=idx,
                               resampling=resampling)
            for name, idx in bands.items()}


# --------------------------------------------------------------------------- #
# AOI mask + clipping
# --------------------------------------------------------------------------- #
def aoi_mask(grid: Grid, aoi_gdf) -> np.ndarray:
    """
    Boolean array (grid.shape) that is True *inside* the AOI polygon(s).

    Rasterises the AOI geometry onto the reference grid.
    """
    geoms = list(aoi_gdf.geometry)
    inside = geometry_mask(
        geometries=geoms,
        out_shape=grid.shape,
        transform=grid.transform,
        invert=True,           # True where a geometry covers the cell
    )
    return inside


def clip_to_aoi(array: np.ndarray, mask: np.ndarray,
                fill_value: float = np.nan) -> np.ndarray:
    """Set every cell outside the AOI mask to `fill_value`."""
    array = np.asarray(array, dtype=np.float64)
    if array.shape != mask.shape:
        raise ValueError("array and mask must share shape.")
    out = array.copy()
    out[~mask] = fill_value
    return out


# --------------------------------------------------------------------------- #
# Vector -> raster (e.g. river/source lines, land-cover polygons)
# --------------------------------------------------------------------------- #
def rasterize_vector(vector_path: str, grid: Grid, attribute: str = None,
                     target_crs=None, all_touched: bool = True) -> np.ndarray:
    """
    Burn a vector layer onto the reference grid.

    If `attribute` is None -> boolean mask (True where a feature falls), handy
    for a river/source layer to seed connected inundation.
    If `attribute` is given -> integer/float raster of that column's value,
    handy for turning land-cover polygons into a class-code raster.

    all_touched=True marks every cell the geometry touches (good for thin
    river lines that would otherwise fall between cell centres).
    """
    from rasterio.features import rasterize

    gdf = gpd.read_file(vector_path)
    if gdf.crs is None:
        raise ValueError("Vector layer has no CRS.")
    gdf = gdf.to_crs(target_crs if target_crs is not None else grid.crs)

    if attribute is None:
        shapes = ((geom, 1) for geom in gdf.geometry)
        dtype = "uint8"
        fill = 0
    else:
        if attribute not in gdf.columns:
            raise ValueError(f"'{attribute}' not in layer columns.")
        shapes = ((geom, val) for geom, val in zip(gdf.geometry, gdf[attribute]))
        dtype = "float64"
        fill = 0

    burned = rasterize(
        shapes=shapes,
        out_shape=grid.shape,
        transform=grid.transform,
        fill=fill,
        all_touched=all_touched,
        dtype=dtype,
    )
    return burned.astype(bool) if attribute is None else burned.astype(np.float64)


# --------------------------------------------------------------------------- #
# Writing results back out as georeferenced GeoTIFF
# --------------------------------------------------------------------------- #
def write_geotiff(path: str, array: np.ndarray, grid: Grid,
                  nodata: float = np.nan, dtype: str = None) -> str:
    """
    Save an array as a GeoTIFF carrying the grid's CRS + transform so it opens
    aligned in any GIS. Returns the path written.
    """
    array = np.asarray(array)
    if array.shape != grid.shape:
        raise ValueError("array shape does not match grid.")
    if dtype is None:
        dtype = "float32" if np.issubdtype(array.dtype, np.floating) else "int32"

    profile = {
        "driver": "GTiff",
        "height": grid.height,
        "width": grid.width,
        "count": 1,
        "dtype": dtype,
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": nodata,
        "compress": "deflate",
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype), 1)
    return path


# --------------------------------------------------------------------------- #
# Demo — builds tiny synthetic GeoTIFF + shapefile, runs the full I/O path
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import os
    import tempfile
    from shapely.geometry import box, LineString

    tmp = tempfile.mkdtemp()
    # --- make a synthetic DEM GeoTIFF in a projected CRS (EPSG:32644, metres)
    H, W, res = 60, 80, 30.0
    minx, maxy = 500000.0, 3000000.0
    transform = from_origin(minx, maxy, res, res)
    yy, xx = np.mgrid[0:H, 0:W]
    dem = (4.0 + 0.15 * np.abs(xx - W / 2) + 0.05 * yy).astype("float32")

    dem_path = os.path.join(tmp, "dem.tif")
    with rasterio.open(
        dem_path, "w", driver="GTiff", height=H, width=W, count=1,
        dtype="float32", crs="EPSG:32644", transform=transform, nodata=-9999,
    ) as dst:
        dst.write(dem, 1)

    # --- make an AOI shapefile covering the central portion of the DEM
    aoi_geom = box(minx + 10 * res, maxy - 50 * res,
                   minx + 70 * res, maxy - 10 * res)
    aoi_gdf_src = gpd.GeoDataFrame({"id": [1]}, geometry=[aoi_geom],
                                   crs="EPSG:32644")
    aoi_path = os.path.join(tmp, "aoi.shp")
    aoi_gdf_src.to_file(aoi_path)

    # --- make a river source line down the valley centre
    river = LineString([(minx + 40 * res, maxy - 5 * res),
                        (minx + 40 * res, maxy - 55 * res)])
    river_path = os.path.join(tmp, "river.shp")
    gpd.GeoDataFrame({"id": [1]}, geometry=[river],
                     crs="EPSG:32644").to_file(river_path)

    # --- run the I/O path
    grid, aoi = grid_from_aoi(aoi_path, resolution=res)
    print(f"grid: {grid.width}x{grid.height} @ {res} m, CRS {grid.crs.to_epsg()}")
    print(f"cell area: {grid.cell_area():.0f} m^2")

    dem_arr = read_aligned(dem_path, grid, resampling="bilinear")
    mask = aoi_mask(grid, aoi)
    dem_clipped = clip_to_aoi(dem_arr, mask)
    river_seed = rasterize_vector(river_path, grid)

    assert dem_arr.shape == grid.shape == mask.shape
    inside = np.isfinite(dem_clipped)
    assert inside.sum() > 0 and inside.sum() <= mask.sum()
    assert river_seed.dtype == bool and river_seed.sum() > 0
    print(f"AOI cells: {int(mask.sum())}, valid DEM cells: {int(inside.sum())}")
    print(f"river seed cells: {int(river_seed.sum())}")

    out = write_geotiff(os.path.join(tmp, "dem_clipped.tif"),
                        dem_clipped, grid, nodata=-9999.0)
    with rasterio.open(out) as chk:
        assert chk.crs.to_epsg() == 32644 and chk.width == grid.width
    print("wrote georeferenced output:", os.path.basename(out))

    print("\nGeospatial I/O layer ran successfully.")
