import earthaccess
from earthaccess import Auth, Store
import xarray as xr
import numpy as np
import os
from glob import glob
from datetime import datetime
import pyproj
from tqdm import tqdm
import ee
import geemap
import geopandas as gpd
from shapely.geometry import Polygon, LineString, Point
import gdown

ee.Initialize()

from tools import download_and_extract_data, load_ana_data
from tools import lonlim, latlim


wgs84 = pyproj.CRS("EPSG:4326") # Geographic coordinates (WGS84)
utm22s = pyproj.CRS("EPSG:32722")  # UTM zone 22S
transformer = pyproj.Transformer.from_crs(wgs84, utm22s, always_xy=True)
xlim, ylim = transformer.transform(lonlim, latlim)

lonmin, lonmax = lonlim
latmin, latmax = latlim


urls = [
    "https://urbanismodrive.procempa.com.br/geopmpa/SPM/PUBLICO/PDDUA_ATUAL/SHP/Bairros_LC12112_16.zip",
    "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2022/UFs/RS/RS_Municipios_2022.zip",
]
for url in urls:
    download_and_extract_data(url, target_dir="../data/external/shapefiles") 


urls = [
    "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SIM/DO24OPEN+(2).csv",
    "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/SIM/DO23OPEN.csv",
    "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/Leitos_SUS/Leitos_2024.csv",
    "https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/Leitos_SUS/Leitos_2023.csv",
]

for url in urls:
    download_and_extract_data(url, target_dir="../data/external/opendatasus")


fnames = [
    "../data/external/hydroweb/5-87242000-TERMINAL CATSUL GUAÍBA-2024-07-09.xls",
    "../data/external/hydroweb/5-87450004-CAIS MAUÁ C6-2024-05-23.xlsx",
]

ana = []
station = []
for fname in fnames:
    ana.append(load_ana_data(fname))
    station.append(fname.split("-")[-4])
ana = xr.concat(ana, "station").assign_coords(station=station)

gdf = gpd.read_file("../data/external/shapefiles/swot_swath/swot_science_orbit_sept2015-v2_swath.shp")

points = [Point(lon, lat) for lon, lat in zip(ana.longitude, ana.latitude)]  # One inside, one outside, one in the other square

# Function to check if a point is contained
def contains_any_point(geometry, points):
    return any(geometry.contains(point) for point in points)

gdf = gdf[gdf['geometry'].apply(contains_any_point, args=(points,))]


# Initialize and attempt login
auth = Auth()
auth.login(strategy="netrc")

# Define datasets of interest with their identifiers
datasets = {
    "swot": {
        "id": {
            "doi": "10.5067/SWOT-PIXC-2.0",
        },
    },
}

# Define the time range for data acquisition
time_range = ("2024-01-01", "2024-07-10")


key="swot"
print(f"{key}\n")

path = f"../data/external/{key}/"
# Create a directory for the current dataset if it doesn't exist

if not os.path.exists(path):
    os.mkdir(path)

selected_results = []
for passing in gdf.ID_PASS:
    # Search for data matching the dataset identifier and specified criteria
    results = earthaccess.search_data(
        **datasets[key]["id"],
        cloud_hosted = True,
        temporal = time_range,
        granule_name= f"*_{passing}_*"
    )

    for item in tqdm(results):
        spatial_domain = item["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]
        if "BoundingRectangles" in list(spatial_domain["Geometry"]):
            for bounds in spatial_domain["Geometry"]["BoundingRectangles"]:
                lons = [bounds["WestBoundingCoordinate"], bounds["EastBoundingCoordinate"]]
                lats = [bounds["SouthBoundingCoordinate"], bounds["NorthBoundingCoordinate"]]
                
                for lon, lat in zip(ana.longitude, ana.latitude):
                    isin = (
                        (lon > lons[0]) & (lon < lons[1]) &
                        (lat > lats[0]) & (lat < lats[1])
                    )
                    if isin:
                        selected_results.append(item)


# Initialize a store object for data retrieval
store = Store(auth)

# Download data files based on the search results and store them in the specified path
files = store.get(selected_results, path)
        
        

        


# landsat
path = f"../data/external/landsat/"
if not os.path.exists(path):
    os.mkdir(path)

region = ee.Geometry.Rectangle(lonmin, latmin, lonmax, latmax)

filterDate = ("2024-03-01", "2024-04-01")
CLOUDY_PIXEL_PERCENTAGE = 5

collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterDate(*filterDate)
    .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", CLOUDY_PIXEL_PERCENTAGE))
)
image = collection.median()

before = geemap.ee_to_xarray(image, crs="EPSG:4326", scale=0.0001, geometry=region).squeeze()

before = before[["B4", "B3", "B2", "SCL"]].load()

before.attrs["filterDate"] = filterDate
before.attrs["CLOUDY_PIXEL_PERCENTAGE"] = CLOUDY_PIXEL_PERCENTAGE
before.attrs["aggregation"] = "median"

before.to_netcdf(os.path.join(path,"before.nc"))



collection = (
    ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
    .filterDate("2024-05-06", "2024-05-07")
)
image = collection.median()

after = geemap.ee_to_xarray(image, crs="EPSG:4326", scale=0.0001, geometry=region).squeeze()

after = after[["B4", "B3", "B2", "SCL"]].load()

after.attrs["filterDate"] = filterDate
after.attrs["CLOUDY_PIXEL_PERCENTAGE"] = CLOUDY_PIXEL_PERCENTAGE
after.attrs["aggregation"] = "median"

after.to_netcdf(os.path.join(path,"after.nc"))

# srtm
image = ee.Image('USGS/SRTMGL1_003').select('elevation')
elevation = geemap.ee_to_xarray(image, crs="EPSG:4326", scale=0.0001, geometry=region).squeeze()
elevation.to_netcdf(os.path.join(path,"elevation.nc"))


landcover = dataset = ee.ImageCollection("ESA/WorldCover/v100").first()
landcover = geemap.ee_to_xarray(landcover, crs="EPSG:4326", scale=0.0001, geometry=region).squeeze()
landcover.to_netcdf(os.path.join(path,"landcover.nc"))


#
folder_id = "13rbfTgkm2BDppESboGL02YJBgT_nwfsV"
folder_url = f'https://drive.google.com/drive/folders/{folder_id}'

path = f"../data/external/gpm_merra2"
if not os.path.exists(path):
    os.mkdir(path)

# Download the folder using gdown
gdown.download_folder(url=folder_url, output=path, quiet=False, use_cookies=False)


datasets_all = {
    'NASA/GSFC/MERRA/slv/2': ["PS", "T10M", "TQV", "U10M", "V10M", "H500", "U500", "V500"],
    'NASA/GPM_L3/IMERG_V06': ["precipitationCal"],
}


lonlim_sa = [-(88+39/60), -(30+40/60)]
latlim_sa = [-(41+23/60), 12+58/60]
bbox = ee.Geometry.BBox(lonlim_sa[0], latlim_sa[0], lonlim_sa[1], latlim_sa[1])

for dataset_id in datasets_all:
    variables = datasets_all[dataset_id]
    dataset_name = dataset_id.replace("/","_")
    dataset = ee.ImageCollection(dataset_id).select(variables).filter(ee.Filter.date("2024-04-01","2024-06-01"))
    ds = geemap.ee_to_xarray(dataset.filterBounds(bbox)).squeeze().transpose("lat","lon","time").sel(lon=slice(*lonlim_sa), lat=slice(*latlim_sa)).load()
    ds.to_netcdf(f"{path}/{dataset_name}_2024.nc")


fnames = glob(f"{path}/*.tif")
for fname in fnames:

    ds_raster = xr.open_dataset(fname)
    
    if type(ds_raster.band_data.long_name)==str:
        ds = (
            ds_raster.squeeze()
            .rename({"band_data":ds_raster.band_data.long_name, "x": "lon", "y": "lat"})
        )
    elif np.all(["_p" in var for var in ds_raster.band_data.long_name]):
        variables, percentiles = zip(*[var.split("_p") for var in ds_raster.band_data.long_name])
        variables = np.array(variables)
        percentiles = np.array([int(p) for p in percentiles])
        
        ds_list = []
        for variable in np.unique(variables):
            # Optimized Indexing with Boolean Masking
            dsi = ds_raster.isel(band=variables == variable)
        
            # Conciser Renaming and Coordinate Assignment
            dsi = dsi.rename({"band": "percentile", "band_data": variable, "x": "lon", "y": "lat"})
            dsi = dsi.assign_coords(percentile=percentiles[variables == variable])
            dsi[variable].attrs["long_name"] = variable
        
            ds_list.append(dsi)
        
        ds = xr.merge(ds_list)
    else:
        variables = np.array(ds_raster.band_data.long_name)
        ds_list = []
        for variable in np.unique(variables):
            # Optimized Indexing with Boolean Masking
            dsi = ds_raster.isel(band=variables == variable).drop_vars("band").squeeze()
        
            # Conciser Renaming and Coordinate Assignment
            dsi = dsi.rename({"band_data": variable, "x": "lon", "y": "lat"})
            dsi[variable].attrs["long_name"] = variable
        
            ds_list.append(dsi)
        
        ds = xr.merge(ds_list)        

    basename = fname.split("/")[-1].split(".")[0]
    ds.to_netcdf(f"{path}/{basename}.nc")


