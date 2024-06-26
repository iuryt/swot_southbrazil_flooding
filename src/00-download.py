import earthaccess
from earthaccess import Auth, Store
import xarray as xr
import numpy as np
import os
from glob import glob
from datetime import datetime
import pyproj
from tqdm import tqdm

from tools import download_and_extract_shapefile
from tools import lonlim, latlim


wgs84 = pyproj.CRS("EPSG:4326") # Geographic coordinates (WGS84)
utm22s = pyproj.CRS("EPSG:32722")  # UTM zone 22S
transformer = pyproj.Transformer.from_crs(wgs84, utm22s, always_xy=True)
xlim, ylim = transformer.transform(lonlim, latlim)


urls = [
    "https://urbanismodrive.procempa.com.br/geopmpa/SPM/PUBLICO/PDDUA_ATUAL/SHP/Bairros_LC12112_16.zip",
    "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2022/UFs/RS/RS_Municipios_2022.zip",
    "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2022/UFs/RS/RS_RG_Imediatas_2022.zip",
    "https://geoftp.ibge.gov.br/organizacao_do_territorio/malhas_territoriais/malhas_municipais/municipio_2022/UFs/RS/RS_RG_Intermediarias_2022.zip"
]
for url in urls:
    download_and_extract_shapefile(url) 


    
    

# Initialize and attempt login
auth = Auth()
auth.login(strategy="netrc")

# Define datasets of interest with their identifiers
datasets = {
    "swot": {
        "id": {"doi": "10.5067/SWOT-PIXC-2.0"},
    },
    "imerg": {
        "id": {"doi": "10.5067/GPM/IMERG/3B-HH-E/06"}
    },
}

# Define the time range for data acquisition
time_range = ("2024-04-10", "2024-05-24")


# Loop through each dataset of interest
for key in datasets:
    print(f"{key}\n")

    path = f"../data/external/{key}/"
    # Create a directory for the current dataset if it doesn't exist

    if not os.path.exists(path):
        os.mkdir(path)


    # Search for data matching the dataset identifier and specified criteria
    results = earthaccess.search_data(
        **datasets[key]["id"],
        cloud_hosted = True,
        temporal = time_range,
    )

    if key == "swot":
        selected_results = []
        for item in tqdm(results):
            spatial_domain = item["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]
            if "BoundingRectangles" in list(spatial_domain["Geometry"]):
                for bounds in spatial_domain["Geometry"]["BoundingRectangles"]:
                    lons = [bounds["WestBoundingCoordinate"], bounds["EastBoundingCoordinate"]]
                    lats = [bounds["SouthBoundingCoordinate"], bounds["NorthBoundingCoordinate"]]
                    
                    for lon, lat in zip(lons, lats):
                        isin = (
                            (lon > lonlim[0]) & (lon < lonlim[1]) &
                            (lat > latlim[0]) & (lat < latlim[1])
                        )
                        if isin:
                            selected_results.append(item)
    else:
        selected_results = results

    # Initialize a store object for data retrieval
    store = Store(auth)

    # Download data files based on the search results and store them in the specified path
    files = store.get(selected_results, path)
        
        

        

