import dva_NCP_all_combine_v2
import imp
import os
import subprocess
import h5py
import numpy as np
from astropy.time import Time
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import datetime
import matplotlib.dates as mdates
from matplotlib.dates import HourLocator as HourLocator
from matplotlib.dates import MinuteLocator as MinuteLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy import units as u
from astropy.time import TimeDelta
from astropy.modeling import models, fitting
from importlib import reload


days = ['Sep24_p2','Sep24_p3','Sep24_p4','Sep25_p1','Sep25_p2','Sep25_p3',
        'Sep25_p4','Sep26_p1','Sep26_p2','Sep26_p3','Sep26_p4']

for day in days:

    reload(dva_NCP_all_combine_v2)
    outfiles = '/home2/DATA/DVA_DATA/NCP_all_observations/'

    # Filetype = 1: no metadata, no noise state in .h5
    # Filetpye = 2: positions in metadata, no noise state in .h5
    # Filetype = 3: positions and noise state in metadata

    #day = 'Sep24_p1'

    dir_files = '/home2/DATA/DVA_DATA/'+day+'_NCP/'
    outname = day+'_NCP_median'
    noise_file = '/home/ordoga/Python/DVA2/DATA/noise_times/noise_'+day+'_NCP.txt'
    filetype = 3


    dva_NCP_all_combine_v2.combine(dir_files,outfiles,outname,filetype=filetype,
                            noise_file=noise_file,df=1.0e6,freq_avg=True,bintype='med')
    #dva_NCP_all_combine_v2.combine(dir_files,outfiles,outname,filetype=filetype,freq_avg=True)

    print(' ')
    print('===========================================')
    print(' Finished NCP for '+day)
    print(' ')
