import warnings
import matplotlib.gridspec as gridspec
import rioxarray  # For the extension to load
import numpy as np
import rioxarray as rxr
import rasterio
import xarray as xr
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import fiona
from tqdm import tqdm
from shapely.geometry import box, Point, shape
from xhistogram.xarray import histogram
from glob import glob
from tools import (
    load_ana_data, format_lat_lon_ticks, plot_scale_bar, scaloa, 
    count_vertices, lonlim, latlim, process_satellite_image, 
    standardize_lat_lon, rasterize_geodataframe
)
from matplotlib.colors import ListedColormap, Normalize
import hvplot.xarray
import hvplot.pandas
import holoviews as hv
import cmasher as cmr
import gsw

# Enable KML support for fiona
fiona.drvsupport.supported_drivers['libkml'] = 'rw'
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw'

# Load ANA data
fnames = glob("../data/external/hydroweb/5-*")
ana = []
station = []

for fname in tqdm(fnames):
    ana.append(load_ana_data(fname))  # Custom function to load ANA data
    station.append(fname.split("-")[-4])  # Extract station name

ana = xr.concat(ana, "station").assign_coords(station=station)

# Extract longitude and latitude from ANA dataset
longitude = ana.longitude.values
latitude = ana.latitude.values
df = pd.DataFrame({"longitude": longitude, "latitude": latitude})

# Define search grid dimensions
dx = dy = 30  # Spatial resolution in meters
dlon = dx / (111.2e3 * np.cos(latitude.mean() * np.pi / 180))  # Adjust for longitude
dlat = dy / 111.2e3  # Latitude adjustment

# Load SWOT data files
fnames = sorted(glob("../data/external/swot/*"))
variables = [
    "height", "water_frac", "classification", "geoid", 
    "illumination_time", "pixel_area", "pole_tide", 
    "load_tide_fes", "solid_earth_tide", "height_cor_xover", 
    "model_dry_tropo_cor", "model_wet_tropo_cor", "iono_cor_gim_ka"
]

# Initialize dictionaries for datasets and heights
ds = {k: [] for k in ana.station.values}
heights = {k: [] for k in ana.station.values}

# Process SWOT data for each station
for fname in tqdm(fnames):
    swot = xr.open_dataset(fname, group="pixel_cloud")[variables]
    swot.load()
    npass = int(fname.split("_")[-6])  # Extract pass number

    for lon, lat, station in zip(ana.longitude.values, ana.latitude.values, ana.station.values):
        # Filter SWOT data within bounds
        where = (
            (swot.longitude > lon - dlon) & (swot.longitude < lon + dlon) &
            (swot.latitude > lat - dlat) & (swot.latitude < lat + dlat) &
            (swot.classification > 2) & (swot.classification < 6) &
            (swot.water_frac > 0.1) & (~np.isnan(swot.height))
        ).values
        ind = np.argwhere(where).ravel()
        dsi = swot.sel(points=ind)

        # Apply corrections to height
        correction = (
            dsi.pole_tide + dsi.load_tide_fes + 
            dsi.solid_earth_tide + dsi.geoid
        )
        dsi["water_level"] = dsi.height - correction

        if dsi.points.size > 0:
            dsa = dsi.median("points").assign_coords(
                time=dsi.illumination_time.mean().dt.round("h"), station=station
            ).expand_dims(["time", "station"])
            ds[station].append(dsa)

        if dsi.points.size > 1:
            level = xr.merge([
                dsi.illumination_time.mean().dt.round("min").rename("illumination_time"),
                dsi["water_level"].median(),
            ])
            level = level.set_coords("time").assign_coords(station=station).expand_dims("station")
            level["pass"] = npass
            heights[station].append(level)

# Combine heights and apply correction
ana_swot = xr.concat([xr.concat(heights[k], "time").drop_duplicates("time") for k in ana.station.values], "station")
ana_swot = ana_swot.isel(time=np.argsort(ana_swot.time.values))
ana_swot = ana_swot.dropna("time", how="all")
ana_swot["time"] = ana_swot["time"] - np.timedelta64(3, "h")
ana_swot["time"].attrs["timezone"] = "UTC-3"

correction = (ana.height.interp(time=ana_swot.time) - ana_swot.water_level).median()
ana_swot["water_level"] += correction
ana_swot.attrs["correction (m)"] = correction.values
ana_swot["water_level"].attrs = {"units": "m", "long_name": "height above geoid"}

# Save processed data
ana_swot.to_netcdf("../data/processed/swot_ana.nc")
ana.to_netcdf("../data/processed/ana.nc")

# Combine SWOT datasets for plotting
ds = xr.concat([xr.concat(ds[k], "time").drop_duplicates("time") for k in ana.station.values], "station")
correction = (ds.pole_tide + ds.load_tide_fes + ds.solid_earth_tide + ds.geoid)

# Plot corrected height and in-situ data
variables = ["height", "geoid", "pole_tide", "load_tide_fes", "solid_earth_tide"]
fig, ax = plt.subplots(2, 1, figsize=(8, 6))

for a, k in zip(ax, ana.station.values):
    (ds.height - correction).sel(station=k).dropna("time").plot(ax=a, label="corrected height", marker="o", markersize=4)
    for var in variables:
        ds.sel(station=k)[var].dropna("time").plot(ax=a, label=var, marker="o", markersize=4, ls="--")
    ana.sel(station=k).height.plot(ax=a, color="k", label="in situ data")
    a.grid(True, linestyle="--", alpha=0.5)
    a.set(title=k)
ax[0].legend(bbox_to_anchor=(1.05, 0.5), loc='upper left', borderaxespad=0.)
ax[0].set(xlabel="", xticklabels=[])
fig.savefig("../img/correction.png", dpi=300)

# Process Landsat imagery
fnames = ["../data/external/landsat/before.nc", "../data/external/landsat/after.nc"]
landsat = [xr.open_dataset(fname).load() for fname in fnames]
before_img = process_satellite_image(landsat[0], gamma=1.3).sel(longitude=slice(*lonlim), latitude=slice(*latlim))
after_img = process_satellite_image(landsat[1], gamma=1.3).sel(longitude=slice(*lonlim), latitude=slice(*latlim))

# Create masks and extract flooded areas
for idx, img in enumerate([landsat[0]["SCL"], landsat[1]["SCL"]]):
    mask = (img == 6).astype('uint8').T
    transform = rasterio.transform.from_origin(img.lon[0], img.lat[0], img.lon[1] - img.lon[0], img.lat[0] - img.lat[1])
    shapes = list(rasterio.features.shapes(mask, transform=transform))
    valid_shapes = [(shape(geom), val) for geom, val in shapes if val == 1]
    if idx == 0:
        gdf = gpd.GeoDataFrame(geometry=[shape[0] for shape in valid_shapes], crs=img.rio.crs)
    else:
        scl_after = gpd.GeoDataFrame(geometry=[shape[0] for shape in valid_shapes], crs=img.rio.crs)

gdf["count"] = gdf.geometry.apply(count_vertices)
gdf = gdf[gdf["count"] > 400]

flooded_areas = gpd.read_file("../data/external/ufrgs/inundacao_em_6_de_maio_de_2024.kml")[["geometry"]].to_crs(4326)
flooded_areas["land"] = True
gdf["land"] = False
flooded_areas = pd.concat([flooded_areas, gdf[["geometry", "land"]]]).reset_index(drop=True)

# Clip and rasterize before/after flood data
bbox = box(lonlim[0], latlim[0], lonlim[1], latlim[1])
before_gdf = gpd.GeoDataFrame(geometry=[flooded_areas[flooded_areas.land == False].unary_union], crs=flooded_areas.crs).clip(bbox)
after_gdf = gpd.GeoDataFrame(geometry=[flooded_areas.unary_union], crs=flooded_areas.crs).clip(bbox)
before_raster = rasterize_geodataframe(before_gdf, lonlim, latlim, resolution=0.001)
after_raster = rasterize_geodataframe(after_gdf, lonlim, latlim, resolution=0.001)

# Save before and after raster and GDF to NetCDF
before_raster.to_netcdf("../data/processed/before.nc")
after_raster.to_netcdf("../data/processed/after.nc")
before_gdf.to_file("../data/processed/before.shp")
after_gdf.to_file("../data/processed/after.shp")
