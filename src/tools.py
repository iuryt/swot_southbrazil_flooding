import xarray as xr
import numpy as np

from pyproj import Transformer

ylim = [-30.3, -29.5]
xlim = [-52, -50.5]


def load_unsmoothed(fname, box):
    dp = 1
    ds = []
    for group in ["left", "right"]:
        dsi = xr.open_dataset(fname, group = group)

        dsi = dsi.assign_coords(longitude = (dsi['longitude'] + 180) % 360 - 180)

        inbox = (
            (dsi.longitude > box[0]) & (dsi.longitude < box[1]) &
            (dsi.latitude > box[2]) & (dsi.latitude < box[3])
        ).any("num_pixels")

        ind = np.argwhere(inbox.values).ravel()

        if len(ind)>1:
            dsi = dsi.isel(num_lines = ind)

            if group == "left":
                dsi = dsi.isel(num_pixels = slice(None, None, -1))

            dsi = dsi.assign_coords(
                num_lines = dsi.num_lines,
                num_pixels = dsi.num_pixels + dp
            )

            # load data into memory only num_lines within the box
            ds.append(dsi)
            dp = dp + dsi.num_pixels.size
            
        else:
            return None
        
    return xr.concat(ds, "num_pixels")


def load(fname, box):

    dsi = xr.open_dataset(fname)

    dsi = dsi.assign_coords(longitude = (dsi['longitude'] + 180) % 360 - 180)

    inbox = (
        (dsi.longitude > box[0]) & (dsi.longitude < box[1]) &
        (dsi.latitude > box[2]) & (dsi.latitude < box[3])
    ).any("num_pixels")

    ind = np.argwhere(inbox.values).ravel()

    dsi = dsi.isel(num_lines = ind)

    return dsi