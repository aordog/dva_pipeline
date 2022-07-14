# Code to combine DVA-2 data files into single, larger files.
#
# A. Ordog, July 2022
#
#--------------------------
# To use:
#--------------------------
# Import dva_sdhdf_combine_simple and call:
# dva_sdhdf_combine_v3.combine(dir_files,outfiles,subdirs,outname)
#
# dir_files = absolute path to directory containing the subdirectories of input 
#             .h5 files (make sure to include "/")
#             NOTE: this was intended for NCP data, which was collected over several
#                   days and therefore stored in several subdirectories
# outfile = absolute path to directory to save output file
# subdirs = subdirectory names inside dir_files
# outname = name of output file
#
# Optional arguments:
#
# transferfiles (default = False) - not yet implemented (doing this manually)
# freq_s (default = 1) - frequency channel steps (for average or downsampling)
# freq_avg (default = False) - average over frequency channels
#
# NOTE: This code is intended for use with the earlier (Winter 2022) versions of 
# the DVA dataset, which stored the time, azimuth and elevation coordinates in
# file['data']['beam_0']['band_SB3']['scan_0']['time'] and
# file['data']['beam_0']['band_SB3']['scan_0']['position']             
# rather than in 'metadata'.
# Also: 
#   - the noise diode state is NOT included
#   - the bottom frequency bands (below 350 MHz) ARE included in the raw data
#     and I remove them when combining.
################################################################################


import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess
import h5py
from astropy.time import Time
import matplotlib.cm as cm
import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable
import array
import matplotlib.dates as mdates
from matplotlib.dates import HourLocator as HourLocator
from matplotlib.dates import MinuteLocator as MinuteLocator

def combine(dir_files,outfiles,NCPdays,outname,transferfiles=False,freq_s=1,freq_avg=False,*args,**kwargs):
       
    print('')
    
    if transferfiles == True:
        print('-------------------------------------------------')
        print('Eventually will have code here for downloading files if needed')               
        print('-------------------------------------------------')
    
    all_files = []

    for NCPday in NCPdays:
        proc=subprocess.Popen('ls -1 '+dir_files+NCPday+'/*.h5', shell=True, stdout=subprocess.PIPE)
        all_files = all_files+proc.communicate()[0].decode().split('\n')[0:-1][:]

    print('Number of files: ',len(all_files))
    print('')
    
    freq,nf = get_frequencies(all_files,dir_files,freq_s,freq_avg)
    print(len(freq))
      
    t_set,az_set,el_set,ra_set,dec_set,nt = get_times_and_coords(all_files)

    
    RR_set,LL_set,reRL_set,imRL_set = get_data_products(all_files,nt,nf,len(freq),freq_avg,freq_s)

    make_new_file(outname,outfiles,all_files[0],RR_set,LL_set,reRL_set,imRL_set,
                  t_set,az_set,el_set,ra_set,dec_set,freq)
    
    return None


def get_frequencies(all_files,dir_files,step,avg_bands):
    
    freq = []
    file = h5py.File(all_files[0],'r')
    beam = file['data']['beam_0']
    print(beam['band_SB3']['frequency'][1]-beam['band_SB3']['frequency'][0])
    print('**********')
    print(beam['band_SB3']['scan_0']['data'].shape[2])
    nf_raw = beam['band_SB3']['scan_0']['data'].shape[2]/step
    nf = int(nf_raw)

    print('------------------------------------------------------------------------')
    print('Number of frequency slices or bins (CAUTION: SHOULD BE WHOLE NUMBER): ',nf_raw)
    print('------------------------------------------------------------------------')
    for i, band_id in enumerate(beam.keys()):
        if i > 1:
            print(i,band_id)
            band = beam[band_id]
            if avg_bands == False:
                freq = np.concatenate([freq,band.get('frequency')[::step]])
            else:
                freq = np.concatenate([freq,np.nanmean(band.get('frequency')[:].reshape(-1,step), axis=1)])
    
    file.close()
    
    return freq,nf

def get_times_and_coords(all_files):
    
    file = h5py.File(all_files[0],'r')
    beam = file['data']['beam_0']
    nt = beam['band_SB3']['scan_0']['data'].shape[0]
    print('Timestamps per file: ',nt)

    t_set = []
    az_set = []
    dec_set = []
    ra_set = []
    el_set = []
    
    for ifile in range(0,len(all_files)):
        
        print(ifile+1,all_files[ifile])
        file = h5py.File(all_files[ifile],'r')
        beam = file['data']['beam_0']
        pos = beam['band_SB3']['scan_0']['position'][:]
        t = beam['band_SB3']['scan_0']['time'][:]
        
        dec = []
        ra = []
        az = []
        el = []
        for j in range(0,len(pos)):
            dec = np.concatenate([dec,[pos[j][2]]])
            ra = np.concatenate([ra,[pos[j][3]]])
            el = np.concatenate([el,[pos[j][0]]])
        
        dec_set = np.concatenate([dec_set,dec])
        ra_set = np.concatenate([ra_set,ra])
        el_set = np.concatenate([el_set,el])
        az_set = np.concatenate([az_set,az])       
        t_set = np.concatenate([t_set,t])
        
    print(len(t_set))
    
    file.close()
    
    return t_set,az_set,el_set,ra_set,dec_set,nt


def get_data_products(all_files,nt,nf,nf_all,avg_bands,step):    

    file = h5py.File(all_files[0],'r')
    beam = file['data']['beam_0']
    
    RR_set = np.empty([nt*len(all_files),nf_all])
    LL_set = np.empty([nt*len(all_files),nf_all])
    reRL_set = np.empty([nt*len(all_files),nf_all])
    imRL_set = np.empty([nt*len(all_files),nf_all])
    
    for ifile in range(0,len(all_files)):

        file = h5py.File(all_files[ifile],'r') 
        #print('-----------------------------------------')
        print('File ',ifile+1,' out of ',len(all_files))
        #print('-----------------------------------------')
              
        for i, band_id in enumerate(beam.keys()):

            if i > 1:   
                j = i-2
                data = file['data']['beam_0'][band_id]['scan_0']['data']
                if avg_bands == False:
                    RR_set[ifile*nt:(ifile+1)*nt,j*nf:(j+1)*nf] = data[:,1,::step]
                    LL_set[ifile*nt:(ifile+1)*nt,j*nf:(j+1)*nf] = data[:,0,::step]
                    reRL_set[ifile*nt:(ifile+1)*nt,j*nf:(j+1)*nf] = data[:,2,::step]
                    imRL_set[ifile*nt:(ifile+1)*nt,j*nf:(j+1)*nf] = data[:,3,::step]
                else:
                    RR_set[ifile*nt:(ifile+1)*nt,j*nf:(j+1)*nf] = np.nanmean(data[:,1,:].reshape(-1,nf,step),axis=2)
                    LL_set[ifile*nt:(ifile+1)*nt,j*nf:(j+1)*nf] = np.nanmean(data[:,0,:].reshape(-1,nf,step),axis=2)
                    reRL_set[ifile*nt:(ifile+1)*nt,j*nf:(j+1)*nf] = np.nanmean(data[:,2,:].reshape(-1,nf,step),axis=2)
                    imRL_set[ifile*nt:(ifile+1)*nt,j*nf:(j+1)*nf] = np.nanmean(data[:,3,:].reshape(-1,nf,step),axis=2)
                    
        file.close()
       
    return RR_set,LL_set,reRL_set,imRL_set


def make_new_file(outname,outfiles,file_ex,RR_set,LL_set,reRL_set,imRL_set,
                  t_set,az_set,el_set,ra_set,dec_set,freq):

    cmd2 = 'cp '+file_ex+' '+outfiles+outname+'.h5'
    os.system(cmd2)
    file = h5py.File(outfiles+outname+'.h5','r+')

    for i in range(1,8):
        #print(file['data']['beam_0'].keys())
        del file['data']['beam_0']['band_SB'+str(i)]
    
    # Create band and scan groups:
    file['data']['beam_0'].create_group("band_SB0")
    file['data']['beam_0']['band_SB0'].create_group(f"scan_0")
    
    # Create power dataset:
    dat = np.empty((len(t_set), 4, len(freq)), dtype=float)
    file['data']['beam_0']['band_SB0']['scan_0'].create_dataset("data", data=dat)
    file['data']['beam_0']['band_SB0']['scan_0']['data'][:,0,:] = RR_set
    file['data']['beam_0']['band_SB0']['scan_0']['data'][:,1,:] = LL_set
    file['data']['beam_0']['band_SB0']['scan_0']['data'][:,2,:] = reRL_set
    file['data']['beam_0']['band_SB0']['scan_0']['data'][:,3,:] = imRL_set
    
    # Create metadata with timestamps and coordinates:
    metadata_content = [t_set,az_set,el_set,ra_set,dec_set]  
    #print(noise)
    col_names = ["utc", "azimuth", "elevation", "right_ascension", "declination"]
    col_types = np.dtype({'names':col_names,'formats':["S32", "f8", "f8", "f8", "f8"] } )
    rec_arr = np.rec.array(metadata_content,dtype=col_types)        
    file['data']['beam_0']['band_SB0']['scan_0'].create_dataset("metadata",data=rec_arr)
    
    # Create frequency dataset:
    file['data']['beam_0']['band_SB0'].create_dataset("frequency",dtype="f8",data=freq)
    
    # Create polarizations dataset:
    pol_labels = [b"ReRR",b"ReLL",b"ReRL",b"ImRL"]
    file['data']['beam_0']['band_SB0'].create_dataset("polarization",dtype="S32",data=pol_labels)
            
    file.close()
    
    return


if __name__ =='__main__': 
    combine()