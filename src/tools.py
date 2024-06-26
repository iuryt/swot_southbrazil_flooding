import numpy as np
import os
import zipfile
import requests

latlim = np.array([-30.3, -29.5])
lonlim = np.array([-52, -50.5])


def download_and_extract_shapefile(url, target_dir="../data/external/"):
    """Downloads a shapefile from a URL, extracts it to a specified directory, and deletes the zip file."""

    # Create the target directory if it doesn't exist
    shapefile_dir = os.path.join(target_dir, "shapefiles")
    os.makedirs(shapefile_dir, exist_ok=True)
    
    shapefile_dir = os.path.join(shapefile_dir, url.split("/")[-1].split(".")[0])
    os.makedirs(shapefile_dir, exist_ok=True)

    # Get the filename from the URL (e.g., "Bairros_LC12112_16.zip")
    filename = url.split("/")[-1]
    filepath = os.path.join(shapefile_dir, filename)

    # Download the file
    with requests.get(url, stream=True) as r:
        r.raise_for_status()  # Check for HTTP errors
        with open(filepath, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    # Extract the contents
    with zipfile.ZipFile(filepath, "r") as zip_ref:
        zip_ref.extractall(shapefile_dir)

    # Delete the zip file
    os.remove(filepath)