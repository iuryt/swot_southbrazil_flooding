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


def parse_station_data(filename):
    """Reads a file containing station data and returns a dictionary for the first station."""

    station_data = {}

    with open(filename, 'r') as file:
        for line in file:
            if line.strip():  # Ignore empty lines
                key, value = line.split(':')
                key = key.strip().lower()
                value = value.strip()

                if key == 'codigo estacao':
                    value = int(value)
                elif key in ['latitude', 'longitude', 'altitude']:
                    value = float(value)
                elif key in ['data inicial', 'data final']:
                    value = datetime.strptime(value, '%Y-%m-%d')

                station_data[key] = value
            else:
                # Empty line found, stop reading
                break  

    return station_data



def scaloa(xc, yc, x, y, t=[], corrlenx=None,corrleny=None, err=None, zc=None):
    """
    (Adapted from Filipe Fernandes function)
    Scalar objective analysis. Interpolates t(x, y) into tp(xc, yc)
    Assumes spatial correlation function to be isotropic and Gaussian in the
    form of: C = (1 - err) * np.exp(-d**2 / corrlen**2) where:
    d : Radial distance from the observations.
    Parameters
    ----------
    corrlen : float
    Correlation length.
    err : float
    Random error variance (epsilon in the papers).
    Return
    ------
    tp : array
    Gridded observations.
    ep : array
    Normalized mean error.
    Examples
    --------
    See https://ocefpaf.github.io/python4oceanographers/blog/2014/10/27/OI/
    Notes
    -----
    The funcion `scaloa` assumes that the user knows `err` and `corrlen` or
    that these parameters where chosen arbitrary. The usual guess are the
    first baroclinic Rossby radius for `corrlen` and 0.1 e 0.2 to the sampling
    error.
    """
    corrlen = corrleny
    xc = xc*( corrleny*1./corrlenx)
    x = x*(corrleny*1./corrlenx)

    n = len(x)
    x, y = np.reshape(x, (1, n)), np.reshape(y, (1, n))
    # Squared distance matrix between the observations.
    d2 = ((np.tile(x, (n, 1)).T - np.tile(x, (n, 1))) ** 2 +
    (np.tile(y, (n, 1)).T - np.tile(y, (n, 1))) ** 2)
    nv = len(xc)
    xc, yc = np.reshape(xc, (1, nv)), np.reshape(yc, (1, nv))
    # Squared distance between the observations and the grid points.
    dc2 = ((np.tile(xc, (n, 1)).T - np.tile(x, (nv, 1))) ** 2 +
    (np.tile(yc, (n, 1)).T - np.tile(y, (nv, 1))) ** 2)
    # Correlation matrix between stations (A) and cross correlation (stations
    # and grid points (C))
    A = (1 - err) * np.exp(-d2 / corrlen ** 2)
    C = (1 - err) * np.exp(-dc2 / corrlen ** 2)
    if 0: # NOTE: If the parameter zc is used (`scaloa2.m`)
        A = (1 - d2 / zc ** 2) * np.exp(-d2 / corrlen ** 2)
        C = (1 - dc2 / zc ** 2) * np.exp(-dc2 / corrlen ** 2)
    # Add the diagonal matrix associated with the sampling error. We use the
    # diagonal because the error is assumed to be random. This means it just
    # correlates with itself at the same place.
    A = A + err * np.eye(len(A))
    # Gauss-Markov to get the weights that minimize the variance (OI).
    tp = None
    ep = 1 - np.sum(C.T * np.linalg.solve(A, C.T), axis=0) / (1 - err)
    if any(t)==True: ##### was t!=None:
        t = np.reshape(t, (n, 1))
        tp = np.dot(C, np.linalg.solve(A, t))
        #if 0: # NOTE: `scaloa2.m`
        #  mD = (np.sum(np.linalg.solve(A, t)) /
        #  np.sum(np.sum(np.linalg.inv(A))))
        #  t = t - mD
        #  tp = (C * (np.linalg.solve(A, t)))
        #  tp = tp + mD * np.ones(tp.shape)
        return tp, ep

    if any(t)==False: ##### was t==None:
        print("Computing just the interpolation errors.")
        #Normalized mean error. Taking the squared root you can get the
        #interpolation error in percentage.
        return ep
        
def plot_scale_bar(ax, length, x0, y0, linewidth=2, orientation=None):
    """
    Plot a scale bar on the map.

    Inputs:
    ax = the axes to draw the scalebar on
    length = length of the scalebar in km
    x0 = the map x location of the scale bar (in projected coordinates)
    y0 = the map y location of the scale bar (in projected coordinates)

    Keywords:
    linewidth: thickness of the scale bar (default is 3)
    orientation: vertical or horizontal (default is vertical)

    """

    km_per_deg_lat = 111.195 # 1 deg lat = 111.195 km
    km_per_deg_lon = km_per_deg_lat*np.cos(y0*np.pi/180) # 1 deg lon = 111.195*cos(lat) km

    bar_length_deg_lat=length/km_per_deg_lat
    bar_length_deg_lon=length/km_per_deg_lon

    if orientation == 'horizontal':
        x=[x0-bar_length_deg_lon/2, x0+bar_length_deg_lon/2]
        y=[y0, y0]
    else:
        x=[x0, x0]
        y=[y0-bar_length_deg_lat/2, y0+bar_length_deg_lat/2]
    
    ax.plot(x, y, markersize = linewidth*2, marker = "|", color = "0.1", linewidth = linewidth, solid_capstyle='butt')
    ax.text(x0, y0, f"{length} km\n\n", va = "center", fontsize = 8, ha = "center")



def format_lat_lon_ticks(ax, decimals=1):
    """
    Formats the x (longitude) and y (latitude) tick labels on a Matplotlib Axes object 
    to include degree symbols and N/S or E/W indicators, with special handling for the equator.
    
    Args:
        ax: The Matplotlib Axes object to format.
        decimals (int): Number of decimal places to display in the tick labels. Defaults to 1.
    """

    format_string = f".{decimals}f"
    
    # Format xtick labels (longitude)
    xticks = ax.get_xticks()
    new_xticklabels = []
    for tick in xticks:
        if tick < 0:
            new_xticklabels.append(f"{abs(tick):{format_string}}$^\circ$W")  # West
        elif tick > 0:
            new_xticklabels.append(f"{tick:{format_string}}$^\circ$E")  # East
        else:
            new_xticklabels.append("0$^\circ$")  # Prime Meridian

    ax.set_xticklabels(new_xticklabels)

    # Format ytick labels (latitude)
    yticks = ax.get_yticks()
    new_yticklabels = []
    for tick in yticks:
        if tick < 0:
            new_yticklabels.append(f"{abs(tick):{format_string}}$^\circ$S")  # South
        elif tick > 0:
            new_yticklabels.append(f"{tick:{format_string}}$^\circ$N")  # North
        else:
            new_yticklabels.append("Eq")  # Equator

    ax.set_yticklabels(new_yticklabels)
    
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