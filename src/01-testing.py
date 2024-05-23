import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm
from matplotlib.colors import LogNorm, SymLogNorm

from tools import xlim, ylim

transform = ccrs.UTM(22, southern_hemisphere = True)

projection = ccrs.PlateCarree()


fnames = glob("../data/external/swot/before/*")
fnames.sort()

kw = dict(vmin = 0, vmax = 20, add_colorbar = False)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection = projection)
ax.set_extent([*xlim, *ylim], crs = ccrs.PlateCarree())

ax.coastlines()
ax.add_feature(cfeature.BORDERS, linestyle='-')
    
for fname in tqdm(fnames):
    
    ds = xr.open_dataset(fname)
    where = True#(ds.water_frac>0.4) & (ds.wse_qual<=2)
    
    ds.wse.where(where).plot(transform = transform, **kw)
    
    # ax.set(title = fname)



# fname = fnames[4]

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection = projection)
# ax.set_extent([*xlim, *ylim], crs = ccrs.PlateCarree())

# ax.coastlines()
# ax.add_feature(cfeature.BORDERS, linestyle='-')

# ds = xr.open_dataset(fname)

# where = (
#     (ds.wse_qual<=2)&
#     (ds.water_frac>0.75)&
#     ((ds.wse_qual_bitwise<1e6))
# )
# ds.wse.where(where).plot(transform = transform, robust = True)
    
    
    
    
    
# for fname in tqdm(fnames):
#     ds = xr.open_dataset(fname)
#     ds.wse.where(ds.wse_qual <= 2).plot(add_colorbar = False)

# dataset = []
# for fname in tqdm(fnames):
#     dataset.append(load_unsmoothed(fname, [*xlim, *ylim]))

# dataset = [d for d in dataset if d!=None]

# kw = dict(vmin = 0, vmax = 20, s = 1, cmap = "viridis")
# var = "ssh_karin_2"





# for ds in tqdm(dataset):
#     p = ds[var].where(ds[f"{var}_qual"] <= 2)
#     ax.scatter(p.longitude, p.latitude, c = p, **kw)

    
    
# for ds in dataset:
#     var = ds.ssh_karin_2.where(ds.ssh_karin_2_qual <= 2)

#     C = ax.scatter(ds.longitude, ds.latitude, s = 2)
# fig.colorbar(C)


# for ds in dataset:
#     if ds!=None:
#         var = ds.ssh_karin_2.where(ds.ssh_karin_2_qual <= 4)

#         C = ax.scatter(ds.longitude, ds.latitude, s = 2)
# fig.colorbar(C)


# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection = ccrs.PlateCarree())
# ax.set_extent([*xlim, *ylim], crs = ccrs.PlateCarree())

# ax.coastlines()
# ax.add_feature(cfeature.BORDERS, linestyle='-')

# for i,ds in enumerate(dataset):
#     if ds!=None:
#         clon = ds.longitude.mean("num_pixels")
#         clat = ds.latitude.mean("num_pixels")

#         ax.plot(clon, clat, label = i)
# ax.legend()


# ds = dataset[10]

# var = ds.ssh_karin_2.where(ds.ssh_karin_2_qual <= 2)
# vmin, vmax = var.quantile([0.05, 0.95]).values

# fig = plt.figure()
# ax = fig.add_subplot(1, 1, 1, projection = ccrs.PlateCarree())
# ax.set_extent([*xlim, *ylim], crs = ccrs.PlateCarree())

# ax.coastlines()
# ax.add_feature(cfeature.BORDERS, linestyle='-')

# C = ax.scatter(ds.longitude, ds.latitude, c = var, s = 2, vmin = vmin, vmax = vmax)

# # ds = load_unsmoothed(fname, [-180, 180, ylim[0]-5, ylim[1]+5])

# # clon = ds.longitude.mean("num_pixels")
# # clat = ds.latitude.mean("num_pixels")

# # dist = np.sqrt(
# #     (ds.longitude-clon)**2 + 
# #     (ds.latitude-clat)**2
# # ).rename("dist")

# # ssh = ds.ssh_karin_2.load()
# # ssha = (ssh-ssh.median("num_pixels")).rename("ssha")

# # from xhistogram.xarray import histogram

# # bins = [
# #     np.arange(-5, 5, 0.005),
# #     np.arange(0, 0.7, 0.02),
# # ]

# # H = histogram(ssha, dist, bins = bins)

# # for fname in fnames:

# #     for group in ["left", "right"]:
        
# #         ds = xr.open_dataset(fname, group = group)

# #         latm = ds.latitude.mean("num_pixels")

# #         ind = np.argwhere((
# #             (latm > ylim[0]) & (latm < ylim[1])
# #         ).values).ravel()

# #         ds = ds.isel(num_lines = ind)

# #         ds = ds.assign_coords(longitude = (ds['longitude'] + 180) % 360 - 180)
    
        
# # plt.scatter(ds.longitude, ds.latitude, c = ds.ssh_karin_2)

