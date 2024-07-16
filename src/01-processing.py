import numpy as np
import xarray as xr
from tools import load_ana_data
from glob import glob
import hvplot.xarray
import hvplot.pandas
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm



fnames = glob("../data/external/hydroweb/5-*")

ana = []
station = []
for fname in tqdm(fnames):
    ana.append(load_ana_data(fname))
    station.append(fname.split("-")[-4])
ana = xr.concat(ana, "station").assign_coords(station=station)



# Your longitude and latitude
longitude = ana.longitude.values
latitude = ana.latitude.values

# Create a DataFrame for your point
df = pd.DataFrame({"longitude": longitude, "latitude": latitude})

dx = dy = 30
dlon = dx/(111.2e3*np.cos(latitude.mean()*np.pi/180))
dlat = dy/(111.2e3)

fnames = glob(f"../data/external/swot/*")
fnames.sort()
fnames = np.array(fnames)

variables = [
    "height", "water_frac", "classification",
    "geoid", "illumination_time"
]

heights = []
for fname in tqdm(fnames):
    swot = xr.open_dataset(fname, group="pixel_cloud")[variables]
    swot.load()
    
    for lon, lat, station in zip(ana.longitude.values, ana.latitude.values, ana.station.values):

        where = (
            (swot.longitude>lon-dlon)&(swot.longitude<lon+dlon)&
            (swot.latitude>lat-dlat)&(swot.latitude<lat+dlat)
        ).values
        
        ind = np.argwhere(where).ravel()

        where = (
            (swot.classification>2)&(swot.classification<6)&
            (swot.water_frac>0.1)
        )
        dsi = swot.where(where).sel(points=ind)

        dsi["water_level"] = (dsi.height-dsi.geoid)
        dsi = dsi.dropna("points")
        
        if dsi.points.size>1:

            level = xr.merge([
                dsi.illumination_time.mean(),
                dsi["water_level"].median(),
            ])

            level["time"] = level.illumination_time.mean()
            level = level.set_coords("time").drop_vars("illumination_time").expand_dims("time")

            level = level.assign_coords(station=station).expand_dims("station")
            
            heights.append(level)

ana_swot = xr.merge(heights)
ana_swot = ana_swot.isel(time=np.argsort(ana_swot.time.values))
ana_swot = ana_swot.dropna("time", how="all")
ana_swot["time"] = ana_swot["time"] - np.timedelta64(3, "h")
ana_swot["time"].attrs["timezone"] = "UTC-3"

correction = (ana.height.interp(time=ana_swot.time)-ana_swot.water_level).median()
ana_swot["water_level"] += correction
ana_swot.attrs["correction (m)"] = correction.values
ana_swot["water_level"].attrs = {"units": "m", "long_name": "height above geoid"}

ana_swot.to_netcdf("../data/processed/swot_ana.nc")
ana.to_netcdf("../data/processed/ana.nc")


colors = ["#e66231ff", "#1f59ffff"]
style = dict(marker="*", lw=0, markersize=8, markeredgewidth=0.6, zorder=10, markeredgecolor="0.2")

before_i, after_i = 14, 15

fig, ax = plt.subplots(figsize=(8,4))

for time in ana_swot.time.values[[before_i, after_i]]:
    ax.axvline(time, ls="--", color="0.3", alpha=0.6)
    
for station, color in zip(ana.station, colors):
    ana_swot.water_level.sel(station=station).plot(ax=ax, x="time", color=color, **style)
    ana.sel(station=station).height.plot(ax=ax, x="time", color=color, label=station.values)
ax.grid(True, ls="--", alpha=0.5)
ax.legend()
ax.set(title="", xlabel="", ylabel="water level [m]")



swot = []
for fname in fnames[[before_i, after_i]]:
    pass
swoti = xr.open_dataset(fname, group="pixel_cloud")[variables]
swoti.load()

where = (
    (swoti.classification>2)&(swoti.classification<6)&
    (swoti.water_frac>0.1)
)
ind = np.argwhere(where.values).ravel()

swoti = swoti.isel(points=ind)

water_level = (swoti.height-swoti.geoid+correction).rename("water_level")

water_level.hvplot.points(x="longitude", y="latitude", color="water_level", geo=True, rasterize=True, clim=(0,7))*df.hvplot.points(x="longitude", y="latitude", geo=True, color="red")

