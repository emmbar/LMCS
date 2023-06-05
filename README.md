# LMCS
Code sharing for NERC project LMCS: LAST EDIT 05/06/2023

unpack_tracks_add_ERA5.py

reads .csv MCS tracks tables
filters to retain only afternoon convective rainfall cases that initate after midday
samples atmospheric conditions (from ERA5) in 6 x 6 degree box centered on largest precipitation feature
outputs .csv filtered MCS tracks table plus atmospheric conditions 
