import xarray as xr
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tools import lonlim, latlim, process_satellite_image, standardize_lat_lon, rasterize_geodataframe
from tools import scaloa
import rasterio
from shapely.geometry import box, Point, shape
import geopandas as gpd
import fiona
import pandas as pd
from xhistogram.xarray import histogram


fiona.drvsupport.supported_drivers['libkml'] = 'rw' # enable KML support which is disabled by default
fiona.drvsupport.supported_drivers['LIBKML'] = 'rw' # enable KML support which is disabled by default


fnames = [
    "../data/external/landsat/before.nc",
    "../data/external/landsat/after.nc"
]
landsat = []
for fname in fnames:
    landsat.append(xr.open_dataset(fname).load())


before_img = process_satellite_image(landsat[0], gamma=1.3)
after_img = process_satellite_image(landsat[1], gamma=1.3)

before_img = before_img.sel(longitude=slice(*lonlim), latitude=slice(*latlim))
after_img = after_img.sel(longitude=slice(*lonlim), latitude=slice(*latlim))


da = landsat[0]["SCL"]

mask = (da==6).astype('uint8').T
mask = mask.rolling(lon=11, lat=11, center=True, min_periods=1).max()


# Get resolution from the coordinate differences
xres = (da.lon[1] - da.lon[0]).item()  # Convert to native Python float
yres = (da.lat[0] - da.lat[1]).item()

# Create the transform
transform = rasterio.transform.from_origin(
    da.lon[0], da.lat[0], xres, yres
)

# Convert the mask to shapes using rasterio.features.shapes
shapes = list(rasterio.features.shapes(mask, transform=transform))

# Filter out empty geometries (if any)
valid_shapes = [(shape(geom), val) for geom, val in shapes if val == 1]  # val == 1 means the mask is True

# Create a GeoDataFrame with the shapes and assign a CRS (if your raster has one)
gdf = gpd.GeoDataFrame(geometry=[shape[0] for shape in valid_shapes], crs=da.rio.crs)


mask = (landsat[1]["SCL"]==6).astype('uint8').T

# Convert the mask to shapes using rasterio.features.shapes
shapes = list(rasterio.features.shapes(mask, transform=transform))

# Filter out empty geometries (if any)
valid_shapes = [(shape(geom), val) for geom, val in shapes if val == 1]  # val == 1 means the mask is True

# Create a GeoDataFrame with the shapes and assign a CRS (if your raster has one)
scl_after = gpd.GeoDataFrame(geometry=[shape[0] for shape in valid_shapes], crs=da.rio.crs)


def count_vertices(geom):
    if geom.geom_type == 'Polygon':
        return len(geom.exterior.coords)
    elif geom.geom_type == 'MultiPolygon':
        return sum(len(poly.exterior.coords) for poly in geom)
    else:
        return None  # Handle non-polygon geometries if present

gdf["count"] = gdf.geometry.apply(count_vertices)
gdf = gdf[gdf["count"]>400]


flooded_areas = gpd.read_file("../data/external/ufrgs/inundacao_em_6_de_maio_de_2024.kml")[["geometry"]]
crs = flooded_areas.crs
flooded_areas["land"] = True
gdf["land"] = False
flooded_areas = pd.concat([flooded_areas.to_crs(4326), gdf[["geometry", "land"]].to_crs(4326)]).reset_index(drop=True)



bbox = box(lonlim[0], latlim[0], lonlim[1], latlim[1])
before_gdf = gpd.GeoDataFrame(geometry=[flooded_areas[flooded_areas.land==False].unary_union])
after_gdf = gpd.GeoDataFrame(geometry=[flooded_areas.unary_union])

before_gdf.crs=crs
after_gdf.crs=crs

before_gdf = gpd.clip(before_gdf, bbox)
after_gdf = gpd.clip(after_gdf, bbox)

before_raster = rasterize_geodataframe(before_gdf, lonlim, latlim, resolution=0.001)
after_raster = rasterize_geodataframe(after_gdf, lonlim, latlim, resolution=0.001)


ana_swot = xr.open_dataset("../data/processed/swot_ana.nc")


variables = [
    "height", "water_frac", "classification", "geoid", 
    "illumination_time", "pixel_area", 
    "pole_tide", "load_tide_fes", "solid_earth_tide", 
    "height_cor_xover", "model_dry_tropo_cor", 
    "model_wet_tropo_cor", "iono_cor_gim_ka"
]


fnames = glob(f"../data/external/swot/*")
fnames.sort()
fnames = np.array(fnames)

swot = []

for i, (fname, raster) in enumerate(zip(fnames[[9, 11]], (before_raster, after_raster))):
    swoti = standardize_lat_lon(xr.open_dataset(fname, group="pixel_cloud")[variables])
    swoti.load()

    water = raster.interp(longitude=swoti.longitude, latitude=swoti.latitude, method="nearest")
    
    where = (
        (swoti.classification>2)&(swoti.classification<6)&
        (swoti.water_frac>0.1)&(water==1)
    )
    
    ind = np.argwhere(where.values).ravel()
    
    swoti = swoti.isel(points=ind)

    correction = ana_swot.attrs["correction (m)"] - (
            swoti.pole_tide +
            swoti.load_tide_fes +
            swoti.solid_earth_tide +
            swoti.geoid
        ) 

    water_level = (swoti.height+correction).rename("water_level")

    swoti = swoti.where(water_level>0, drop=True)

    water_level = water_level.where(water_level>0, drop=True)

    swoti = xr.merge([swoti, water_level])
    
    swot.append(swoti)



dx = 0.002
bins = [
    np.arange(*swot[1].latitude.quantile([0, 1]).values, dx),
    np.arange(*swot[1].longitude.quantile([0, 1]).values, dx),
]

variables = ["water_level"]
swot_gridded = []
labels = ["before", "after"]
for variable in variables:
    data = []
    for ds, label in zip(swot, labels):
        # ds = ds.dropna("points")
    
        H = histogram(ds.latitude, ds.longitude, bins=bins, weights=ds[variable])/histogram(ds.latitude, ds.longitude, bins=bins)
        data.append(H.rename(variable).rename({dim: dim.split("_")[0] for dim in H.dims}))
    swot_gridded.append(xr.concat(data, "time").assign_coords(time=labels))
swot_gridded = xr.merge(swot_gridded)



xc, yc = np.meshgrid(swot_gridded.longitude, swot_gridded.latitude)
mask = after_raster.interp(longitude=swot_gridded.longitude, latitude=swot_gridded.latitude, method="nearest")
print(xc.size)



swoti = swot[1]

n = xc.size
ind = np.random.randint(0,swoti.water_level.size-1,n)
t = swoti.water_level.values[ind]
x, y = swoti.longitude.values[ind], swoti.latitude.values[ind]
tp, ep = scaloa(xc.ravel(), yc.ravel(), x, y, t, corrlenx=0.1, corrleny=0.1, err=0.01)
level = (xr.ones_like(swot_gridded.water_level[1])*tp.reshape(xc.shape)).where(mask==1)


level.to_netcdf("../data/processed/swot_oa.nc")