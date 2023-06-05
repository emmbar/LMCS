###############################################################
# Python 2.7
#
# To run:
# python unpack_tracks_add_ERA5.py "region" #"input_directory_tracks"  "input_directory_ERA5"  #"output_directory"
#
# Last Edited: 05/06/2023
# Author: E J Barton emmbar@ceh.ac.uk
#
# Read storm track tables, extract afternoon (1400 - 2000 LT, # initate after 1200 LT) convective rainfall (max rate > 8 
# mm.hr-1) cases, sample morning (1000 LT) atmopsheric 
# conditions from ERA5
#
# Assumes individual ERA5 files for each day in region 
# subdirectories with naming convention: 
# ERA_YYYY_MM_DD_region_pl.nc
# Filtering and atmospheric sampling based on largest 
# precipitation feature (pf1)
#
##############################################################

import sys
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import scipy.ndimage.interpolation as inter
from scipy.interpolate import griddata
import numpy.ma as ma
import copy
import math
from scipy.spatial.qhull import QhullError
from mpl_toolkits.basemap import maskoceans

try:
	region, idir_tracks,idir_ERA5, odir = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
except IOError:
	print "Please specify region: WAf, SAf, india, china, australia, sub_SA or GPlains"
	print "Please specify input directory for storm tracks table"
	print "Please specify input directory for ERA5 data"
	print "Please specify output directory"
	sys.exit()

#time difference UTC/LT
if region == 'WAf':
	ofset = 0
elif region == 'SAf':
	ofset = 2
elif region == 'india':
	ofset = 5
elif region == 'china':
	ofset = 8
elif region == 'australia':
	ofset = 9
elif region == 'sub_SA':
	ofset = -4
elif region == 'GPlains':
	ofset = -6

#define grid for atmospheric sampling
Rx, Ry = np.mgrid[-3:3:25j, -3:3:25j]


#count cases

count = 0

# loop through storm tables
for year in range(2000,2020):
	yr = str(year)
	print "processing", yr
	
#Read storm tracks variables into Pandas DataFrame

	try:
		iname_tracks = idir_tracks+region+'_initTime__mcs_tracks_extc_'+yr+'0101_'+yr+'1231.csv'
		df = pd.read_csv(iname_tracks)
	except IOError:
		print "Cannot find file", iname

# extract columns from pandas dataframe for filtering
	base_time = df['base_time']
	init_hour = df['init_hour']
	hour = df['hour']

	loclonpf1 = df['pf_lon1']
	loclatpf1 = df['pf_lat1']

	pf_maxrainrate1 = df['pf_maxrainrate1']
	pf_area1 = df['pf_area1']

# create new lists to be converted to colums 
	pf1_shear = []

# loop through existing columns
	for i in range(0,len(base_time)):

		ihr = init_hour[i] + ofset
		if ihr < 13:
			df = df.drop(labels=i,axis=0)
			continue 
		else:
			pass 

		hr = hour[i] + ofset
		pday = False
		if hr < 0:
			hr = hr + 24
			pday = True
		else:
			pass 

		if hr < 14 or hr > 20:
			df = df.drop(labels=i,axis=0)
			continue 
		else:
			pass 

		parea = pf_area1[i]
		if parea < 5000:
			df = df.drop(labels=i,axis=0)
			continue
		else:
			pass 

		maxrain_pf1 = pf_maxrainrate1[i]			
		if np.isnan(maxrain_pf1) == True:
			df = df.drop(labels=i,axis=0)
			continue

		elif maxrain_pf1 < 8:
			df = df.drop(labels=i,axis=0)
			continue
		else:
			pass 

		cdate = base_time[i]
		yr,mon,dy = cdate[0:4],cdate[5:7],cdate[8:10] 

		if pday == True:
			dy = int(dy) - 1
			if dy < 10:
				dy = '0' + str(dy)
			else:
				dy = str(dy)
		else:
			pass

#retrieve atmospheric data

		try:
			RAfile = idir_ERA5+region+'/ERA5_'+yr+'_'+mon+'_'+dy+'_'+region+'_pl.nc'
                        RA = Dataset(RAfile)
		except IOError:
			print 'no reanalysis file', RAfile
			df = df.drop(labels=i,axis=0)
			continue

		RAlat = RA['latitude'][:]
		RAlon = RA['longitude'][:]

		lon5, lat5 = np.meshgrid(RAlon, RAlat)

		u925 = RA['u'][10+ofset,0,:,:]	
		u650 = RA['u'][10+ofset,2,:,:]

		pf1lat = loclatpf1[i]
		pf1lon = loclonpf1[i]

# locate nearest pixel to precip feature in atmospheric data 

		latf5 = abs(RAlat - pf1lat)
		lonf5 = abs(RAlon - pf1lon)

		loc_lat = np.argmin(latf5)
		loc_lon = np.argmin(lonf5)

		box_lat5 = lat5[loc_lat-15:loc_lat+15,loc_lon-15:loc_lon+15]
		box_lon5 = lon5[loc_lat-15:loc_lat+15,loc_lon-15:loc_lon+15]
		box_u925 = u925[loc_lat-15:loc_lat+15,loc_lon-15:loc_lon+15]
		box_u650 = u650[loc_lat-15:loc_lat+15,loc_lon-15:loc_lon+15]

		box_lat5 = box_lat5 - pf1lat
		box_lon5 = box_lon5 - pf1lon

# regrid atmospheric data

		box_lon5,box_lat5 = box_lon5.flatten(), box_lat5.flatten()

		box_u925,box_u650 = box_u925.flatten(),box_u650.flatten()

		try:

                	u925_R = griddata((box_lon5,box_lat5),box_u925,(Rx,Ry),method='linear',fill_value=np.nan)
                	u650_R = griddata((box_lon5,box_lat5),box_u650,(Rx,Ry),method='linear',fill_value=np.nan)

                except QhullError:
                        df = df.drop(labels=i,axis=0)
			continue
                except ValueError:
                        df = df.drop(labels=i,axis=0)
                        continue
		except IndexError:
			df = df.drop(labels=i,axis=0)
			continue

# calculate shear

		ushear = u650_R - u925_R
		ushear = np.ma.masked_invalid(ushear)
		ushear_mean = np.ma.mean(ushear)

		pf1_shear.append(ushear_mean)

		count += 1

	df['pf1_650925_meanshear'] = pf1_shear

	oname = odir+region+'_initTime__mcs_tracks_extc_'+yr+'0101_'+yr+'1231_atmos.csv'

	df.to_csv(oname)


print 'total cases', count
		






