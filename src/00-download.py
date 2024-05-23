import earthaccess
from earthaccess import Auth, Store
import xarray as xr
import numpy as np
import os

from tools import xlim, ylim

# Initialize and attempt login
auth = Auth()
auth.login(strategy="netrc")

# Define datasets of interest with their identifiers
datasets = {
    "swot": {
        "id": {"short_name": "SWOT_L2_HR_Raster_2.0"},
    },
}

# Define the time range for data acquisition
time_ranges = {
    "before": ("2024-04-10", "2024-04-26"),
    "after": ("2024-04-28", "2024-05-14"),
}

for time_key in time_ranges:
    print(f"\n\n{time_key}\n")
    time_range = time_ranges[time_key]
    
    # Loop through each dataset of interest
    for key in datasets:
        print(f"{key}\n")
        
        path = f"../data/external/{key}/"
        # Create a directory for the current dataset if it doesn't exist

        if not os.path.exists(path):
            os.mkdir(path)
        
        path = f"{path}/{time_key}"

        if not os.path.exists(path):
            os.mkdir(path)
            
        # Search for data matching the dataset identifier and specified criteria
        results = earthaccess.search_data(
            **datasets[key]["id"],
            cloud_hosted = True,
            temporal = time_range,
            granule_name = '*_100m_*',
        )

        selected_results = []
        for item in results:
            spatial_domain = item["umm"]["SpatialExtent"]["HorizontalSpatialDomain"]
            if "GPolygons" in list(spatial_domain["Geometry"]):
                points = spatial_domain["Geometry"]["GPolygons"][0]["Boundary"]["Points"]

                for point in points:
                    lon = point["Longitude"]
                    lat = point["Latitude"]
                    isin = (
                        (lon > xlim[0]) & (lon < xlim[1]) &
                        (lat > ylim[0]) & (lat < ylim[1])
                    )
                    if isin:
                        selected_results.append(item)


        # Initialize a store object for data retrieval
        store = Store(auth)

        # Download data files based on the search results and store them in the specified path
        files = store.get(selected_results, path)