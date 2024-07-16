import numpy as np
import os
import zipfile
import requests
import xarray as xr
import pandas as pd
from datetime import datetime, timedelta
import geopandas as gpd
import rioxarray as rxr
import rasterio
from rasterio.features import rasterize
from bs4 import BeautifulSoup

lonlim = np.array([-51.67, -51.1])
latlim = np.array([-30.3, -29.83])


def set_hatch_color(cs, color):
    for collection in cs.collections:
        collection.set_edgecolor(color)
        collection.set_linewidth(0.) 
        

def get_band_info(dataset_id):
    """Fetches band information from the Earth Engine Data Catalog.

    Args:
        dataset_id: The Earth Engine dataset ID (e.g., 'NASA/GSFC/MERRA/slv/2').

    Returns:
        A pandas.Dataframe with the bands information.
    """
    
    # Construct the Data Catalog URL
    catalog_url = f'https://developers.google.com/earth-engine/datasets/catalog/{dataset_id.replace("/","_")}#bands'
    
    # Fetch the HTML content of the page
    response = requests.get(catalog_url)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find the table containing band information
    band_table = soup.find("table", {"class": "eecat"})

    # Parse table to a pandas.DataFrame
    band_info = pd.read_html(band_table.prettify(), index_col="Name")[0]

    return band_info

def standardize_lat_lon(ds: xr.Dataset) -> xr.Dataset:
    """Standardizes latitude and longitude dimensions and attributes in an xarray Dataset.

    This function renames "lon" and "lat" dimensions to "longitude" and "latitude" respectively,
    if they are not already named as such. It also sets the `long_name` and `units` attributes 
    for both dimensions to improve clarity and interoperability.

    Args:
        ds: The xarray Dataset to standardize.

    Returns:
        The standardized xarray Dataset with consistent latitude/longitude naming and attributes.
    """

    # Rename dimensions if necessary (avoids overwriting existing attributes)
    rename_dict = {}
    if "lon" in ds.dims and "longitude" not in ds.dims:
        rename_dict["lon"] = "longitude"
    if "lat" in ds.dims and "latitude" not in ds.dims:
        rename_dict["lat"] = "latitude"

    ds = ds.rename(rename_dict)  # Apply renaming only if needed

    # Set standard attributes for longitude and latitude
    ds["longitude"].attrs = dict(long_name="longitude", units="°")
    ds["latitude"].attrs = dict(long_name="latitude", units="°")

    return ds

def process_satellite_image(image: xr.DataArray, gamma: float = 0.6) -> xr.DataArray:
    """Processes a satellite image from raw RGB bands to a visually enhanced format.

    This function performs the following steps:

    1. Combines the specified RGB bands from the input image.
    2. Clips outlier values based on specified percentiles.
    3. Scales pixel values to the 0-1 range.
    4. Applies gamma correction for improved visual contrast (optional).
    5. Transposes the data for easier plotting and visualization.

    Args:
        image: An xarray DataArray containing the satellite image data. It is expected
            to have dimensions for bands, latitude, and longitude.
        gamma: The gamma correction factor. A value of 1.0 results in no correction. 
            Higher values increase contrast in darker regions. Defaults to 0.6.

    Returns:
        The processed satellite image as an xarray DataArray, with bands, latitude, and
        longitude dimensions, ready for visualization or further analysis.
    """

    # Extract RGB bands and rename to match further processing steps
    rgb = xr.concat([image[b] for b in ["B4", "B3", "B2"]], "bands").rename("image").assign_coords(bands=[0, 1, 2])

    # Clip outliers using 2nd and 98th percentiles
    vmin, vmax = rgb.quantile([0.02, 0.98])
    rgb_clipped = rgb.clip(vmin, vmax)

    # Linearly scale the clipped values to the 0-1 range
    rgb_scaled = (rgb_clipped - vmin) / (vmax - vmin)

    # Optional gamma correction for enhanced contrast (if gamma != 1.0)
    if gamma != 1.0:
        rgb_gamma_corrected = rgb_scaled ** (1 / gamma)
    else:
        rgb_gamma_corrected = rgb_scaled

    # Determine the correct dimension names for transposition
    latitude_dim = "latitude" if "latitude" in image.dims else "lat"
    longitude_dim = "longitude" if "longitude" in image.dims else "lon"

    # Transpose using the dynamically determined dimension names
    rgb_gamma_corrected = rgb_gamma_corrected.transpose(latitude_dim, longitude_dim, "bands")

    # Standardize longitude and latitude coordinates
    rgb_gamma_corrected = standardize_lat_lon(rgb_gamma_corrected)
    
    return rgb_gamma_corrected


def rasterize_geodataframe(gdf, lonlim, latlim, resolution=0.001):
    """
    Rasterizes a GeoDataFrame within specified longitude and latitude limits.

    Args:
        gdf (GeoDataFrame): The GeoDataFrame containing the geometries to rasterize.
        lonlim (tuple): A tuple of (min_longitude, max_longitude) defining the raster extent.
        latlim (tuple): A tuple of (min_latitude, max_latitude) defining the raster extent.
        resolution (float, optional): The desired resolution of the output raster (in degrees). Defaults to 0.001.

    Returns:
        xarray.DataArray: A rasterized representation of the GeoDataFrame.
    """

    # Ensure that the GeoDataFrame's CRS is set
    if gdf.crs is None:
        raise ValueError("GeoDataFrame must have a defined CRS")

    # Calculate raster dimensions based on limits and resolution
    nrows = int(np.ceil((latlim[1] - latlim[0]) / resolution))
    ncols = int(np.ceil((lonlim[1] - lonlim[0]) / resolution))

    # Create affine transformation for the raster
    transform = rasterio.transform.from_origin(lonlim[0], latlim[1], resolution, resolution)

    # Rasterize the GeoDataFrame's geometries
    shapes = ((geom, 1) for geom in gdf.geometry)  # Prepare shapes for rasterization
    raster = rasterio.features.rasterize(
        shapes,
        out_shape=(nrows, ncols),
        transform=transform,
        fill=0,  # Fill with 0 outside of geometries
        dtype='uint8',
    )

    # Create xarray DataArray with raster data and coordinates
    da_raster = xr.DataArray(
        raster,
        coords={'longitude': np.linspace(lonlim[0], lonlim[1], ncols), 
                'latitude': np.linspace(latlim[1], latlim[0], nrows)},
        dims=['latitude', 'longitude']
    )

    return da_raster
    


def load_ana_data(fname):
    info = {
        "Chuva Horária (mm)":{"name": "rain", "attrs": {"long_name": "Hourly rain", "units": "mm"}},
        "Nível adotado (cm)":{"name": "height", "attrs": {"long_name": "Height", "units": "m"}},
        "Bateria (V)":{"name": "battery", "attrs": {"long_name": "Battery", "units": "V"}},
        "Temp. Interna (ºC)":{"name": "temperature", "attrs": {"long_name": "Temperature", "units": "°C"}},
        "Vazão (m³/s)":{"name": "flow_rate", "attrs": {"long_name": "Flow rate", "units": "m³/s"}},
    }

    try:
        df = pd.read_html(fname)[0].drop([0, 1, 2, 3, 4]).reset_index(drop=True)
    except:
        df = pd.read_excel(fname, dtype="str").drop([0, 1, 2]).reset_index(drop=True)
    df = df.T.set_index(0).T.set_index("Data/Hora")
    dtime = []
    for i in df.index:
        if "/" in i:
            fmt = "%d/%m/%Y %H:%M:%S"
        else:
            fmt = "%Y-%d-%m %H:%M:%S"
        dtime.append(datetime.strptime(i, fmt))
    df.index = dtime
    df.columns.name = ""
    df.index.name = "time"
    df = df.iloc[:, ~df.isna().all().values]
    df = df.astype("float")
    df = df.iloc[np.argsort(df.index)]
    ds = df.to_xarray()
    
    for key in list(ds):
        ds[key].attrs = info[key]["attrs"]
        ds = ds.rename({key: info[key]["name"]})

    ds["height"] = ds["height"]*1e-2
    
    code = int(fname.split("-")[1])
    
    stations = pd.read_excel("../data/external/hydroweb/stations.xlsx")
    stations["Longitude"], stations["Latitude"] = stations["lon"], stations["lat"]
    stations = stations.drop(["lon", "lat"], axis=1)
    stations = gpd.GeoDataFrame(stations, geometry=gpd.points_from_xy(stations.Longitude, stations.Latitude))
    
    longitude = stations[stations["Codigo"]==code].Longitude.values.tolist()[0]
    latitude = stations[stations["Codigo"]==code].Latitude.values.tolist()[0]
    
    ds = ds.assign_coords(longitude=longitude, latitude=latitude)
    return ds
    

def download_and_extract_data(url, target_dir="../data/external/"):
    """Downloads a shapefile from a URL, if zip, extracts it to a specified directory, and deletes the zip file."""

    # Create the target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)

    # Get the filename from the URL (e.g., "Bairros_LC12112_16.zip")
    filename = url.split("/")[-1]
    filepath = os.path.join(target_dir, filename)

    # Download the file
    with requests.get(url, stream=True) as r:
        r.raise_for_status()  # Check for HTTP errors
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    if filename.split(".")[-1]=="zip":
        # Extract the contents
        with zipfile.ZipFile(filepath, "r") as zip_ref:
            zip_ref.extractall(shapefile_dir)
    
        # Delete the zip file
        os.remove(filepath)