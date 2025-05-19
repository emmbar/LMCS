######################################################################
# Code to create table of MCS properties from ICON NGC4008 simulation
######################################################################

################
# IMPORTS      #
################

import intake
import pandas as pd
import numpy as np
import numpy.ma as ma
import xarray as xr
import healpy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from scipy.ndimage import label
from matplotlib import cm
import sys

#####################
# FUNCTIONS         #  
#####################

def get_nn_lon_lat_index(nside, lons, lats):
    """for subsetting HEALPix ICON out onto regular lat/lon grid"""
    lons2, lats2 = np.meshgrid(lons, lats)
    return xr.DataArray(
        healpy.ang2pix(nside, lons2, lats2, nest=True, lonlat=True),
        coords=[("lat", lats), ("lon", lons)],
    )

def dictionary():
    """setup dictionary for storm table"""
    dic = {}
    vars = ['date', 'month', 'hour', 'minute', 'year', 'day', 'area', '70area', 'tmin',
            'minlon', 'minlat', 'maxlon', 'maxlat', 'clon', 'clat', 'tminlon', 'tminlat',
            'tmin', 'tmean', 'tp1', 'tp5','precipitation_mean','precipitation_max','precipitation_p95','precipitation_p99']


    for v in vars:
        dic[v] = []
    return dic

def olr_to_bt(olr):
    """Application of Stefan-Boltzmann law - converts outgoing longwave to cloud top temperature"""
    sigma = 5.670373e-8
    tf = (olr/sigma)**0.25
    #Convert from bb to empirical BT (degC) - Yang and Slingo, 2001
    a = 1.228
    b = -1.106e-3
    Tb = (-a + np.sqrt(a**2 + 4*b*tf))/(2*b)
    return Tb - 273.15

##########################
# Read simulation data   #
##########################

cat = intake.open_catalog("https://data.nextgems-h2020.eu/catalog.yaml")

zoom = 9 # ~ 10km
ds = cat.ICON.ngc4008(zoom=zoom,time="PT15M").to_dask()

# MCS hotspot region boxes

region = 'WAf'

if region == 'WAf': # West Africa
    idx = get_nn_lon_lat_index(
        2**zoom, np.linspace(-18, 25, 430), np.linspace(4, 25, 210)) # WAf
elif region == 'Ind': # India
    idx = get_nn_lon_lat_index(
        2**zoom, np.linspace(70, 90, 200), np.linspace(5, 30, 250)) # india
elif region == 'Chi': # China
    idx = get_nn_lon_lat_index(
        2**zoom, np.linspace(105, 125, 100), np.linspace(25, 40, 150)) # china
elif region == 'USA': # US Great Plains
    idx = get_nn_lon_lat_index(
        2**zoom, np.linspace(-100, -90, 100), np.linspace(32, 47, 150)) # Great Plains
elif region == 'SAf': # South Africa
    idx = get_nn_lon_lat_index(
        2**zoom, np.linspace(20, 35, 150), np.linspace(-35, -15, 200)) # SAf
elif region == 'SAm': # South America
    idx = get_nn_lon_lat_index(
        2**zoom, np.linspace(-68, -47, 210), np.linspace(-40, -20, 200)) # SAm
elif region == 'AUS': # Northen Australia
    idx = get_nn_lon_lat_index(
        2**zoom, np.linspace(120, 140, 200), np.linspace(-23, -11, 120)) # Aus


#################
# Cloud loop    #
#################

sdex, fdex = 14592, 23423 # example for June - August 2020

dic = dictionary()

for tdex in range(sdex,fdex,4): # hourly sampling

  rlut_lon_lat = ds.rlut.isel(time=tdex, cell=idx) # outgoing longwave radition OLR
  pr_lon_lat = ds.pr.isel(time=tdex, cell=idx) # precipitation flux

  Tb_lon_lat = olr_to_bt(rlut_lon_lat) # convert OLR to brightness temperature

  datestr = str(pr_lon_lat.coords['time'].values)
  yr = datestr[0:4]
  mn = datestr[5:7]
  dy = datestr[8:10]
  hr = datestr[11:13]
  mt = datestr[14:16]

  Tb = Tb_lon_lat.values
  pr = pr_lon_lat.values
  lon_Tb = Tb_lon_lat.coords['lon'].values
  lat_Tb = Tb_lon_lat.coords['lat'].values

  lon, lat = np.meshgrid(lon_Tb, lat_Tb)

  Tb [Tb > -50] = 0 # cloud shield colder than -50C

# blob cutting

  labels, numL = label(Tb)
  u, inv = np.unique(labels, return_inverse=True)
  n=np.bincount(inv)
  goodinds= u[n > 50] # clouds larger than 5000 km2

  for gi in goodinds:
    if gi == 0: # first index is empty
      continue
                        
    inds = np.where(labels == gi)

    latmax, latmin = lat[inds].max(), lat[inds].min()
    lonmax, lonmin = lon[inds].max(), lon[inds].min()

    marg = 0.3 # cuts out box around cloud with margin = marg
    i, j = np.where( (lon>lonmin-marg) & (lon<lonmax+marg) & (lat>latmin-marg) & (lat<latmax+marg) )
    blat = lat[i.min():i.max()+1, j.min():j.max()+1]
    blon = lon[i.min():i.max()+1, j.min():j.max()+1]
    bcloud = Tb[i.min():i.max()+1, j.min():j.max()+1]
    brain = pr[i.min():i.max()+1, j.min():j.max()+1]

    area50 = np.sum(bcloud<=-50)*100 # 50C cloud shield
    area70 = np.sum(bcloud<=-70)*100 # 70C cloud shield

    CTTmin = np.min(bcloud)
    mindex_x = np.where(bcloud == bcloud.min())[0][0]
    mindex_y = np.where(bcloud == bcloud.min())[1][0]

    latCTTmin = blat[mindex_x,mindex_y]
    lonCTTmin = blon[mindex_x,mindex_y] # location of coldest pixel

    maxdex_x = np.where(brain == brain.max())[0][0]
    maxdex_y = np.where(brain == brain.max())[1][0]        

    rainbox = brain[maxdex_x-1:maxdex_x+2,maxdex_y-1:maxdex_y+2]
    test = np.copy(rainbox)
    test[test>-1]=1
        
    if int(np.sum(test) != 9):
      continue

    rainbox_max = np.mean(rainbox)*3600.0 # calcualte max rainfall over a box for a more realistic value

    bcloudnan = np.copy(bcloud)
    bcloudnan [bcloudnan > -50] = np.nan

    rainmask = np.ma.masked_where(bcloud > -50,brain)
    rainmask = np.ma.filled(rainmask,np.nan)

    if (np.nanmax(rainmask)*3600) < 1.0: # only include raining storms
      continue

    dic['date'].append(datestr)
    dic['month'].append(int(mn))
    dic['hour'].append(int(hr))
    dic['year'].append(int(yr))
    dic['day'].append(int(dy))
    dic['minute'].append(int(mt))

    dic['minlon'].append(lonmin)
    dic['minlat'].append(latmin)
    dic['maxlon'].append(lonmax)
    dic['maxlat'].append(latmax)
    dic['clon'].append(lonmin + (lonmax - lonmin)/2)
    dic['clat'].append(latmin + (latmax - latmin)/2)
    dic['area'].append(area50)
    dic['70area'].append(area70)
    dic['tmin'].append(CTTmin)
    dic['tminlat'].append(latCTTmin)
    dic['tminlon'].append(lonCTTmin)

    dic['tmean'].append(np.nanmean(bcloudnan))
    dic['tp1'].append(np.nanpercentile(bcloudnan, 1))
    dic['tp5'].append(np.nanpercentile(bcloudnan, 5))

    dic['precipitation_mean'].append(np.nanmean(rainmask)*3600) # rain under cloud shield mm hr-1
    dic['precipitation_max'].append(rainbox_max) 
    dic['precipitation_p95'].append(np.nanpercentile(rainmask,95)*3600)
    dic['precipitation_p99'].append(np.nanpercentile(rainmask,99)*3600)

  df = pd.DataFrame.from_dict(dic)
  df.to_csv("storm_tables/"+region+"_2020_MCS_5000km2_-50C_1mm_10km_ICON_hourly_JJA.csv")

print('finished')
