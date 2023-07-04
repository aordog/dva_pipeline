# Code to combine DVA-2 data files into scans defined by
# input start and end times.
#
# A. Ordog, June 2022
#
#--------------------------
# To use:
#--------------------------
# Import dva_sdhdf_combine_v3 and call:
# dva_sdhdf_combine_v3.combine(dir_files,outfiles,t1,t2,outname)
#
# dir_files = absolute path to directory containing input .h5 files
#             (make sure to include "/")
# outfile = absolute path to directory to save output files
# t1, t2 = start and end times of scan in "isot" format
# outname = name of output file
#
# Optional arguments:
#
# transferfiles (default = False) - not yet implemented (doing this manually)
# freq_s (default = 1) - frequency channel steps (for average or downsampling)
# freq_avg (default = False) - average over frequency channels
# az_scan_trim (default = False) - set to True to flag bad points in az scans
#
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

def combine(dir_files,outfiles,t1,t2,outname,
            transferfiles=False,freq_s=1,freq_avg=False,az_scan_trim=False,bintype='mean',
            *args,**kwargs):
       
    print('')
    
    if transferfiles == True:
        print('-------------------------------------------------')
        print('Eventually will have code here for downloading files if needed')               
        print('-------------------------------------------------')
    
    d1 = t1[0:10]
    h1 = int(t1[11:13])
    m1 = int(t1[14:16])
    d2 = t2[0:10]
    h2 = int(t2[11:13])
    m2 = int(t2[14:16])
    days = [d1,d2]
    
    if m2 < 59:
        m2_use = m2+2
        h2_use = h2
    else:
        m2_use = 0
        h2_use = h2+1
        
    if m1 > 0:
        m1_use = m1-1
        h1_use = h1
    else:
        m1_use = 59
        h1_use = h1-1

    all_times = []

    if d1 == d2:       
        if h1_use == h2_use:
            for minute in range(m1_use,m2_use):
                all_times.append(d1+'T'+f"{h1_use:02}"+':'+f"{minute:02}")
        else:
            for hour in range(h1_use,h2_use+1):
                if hour == h1_use:
                    for minute in range(m1_use,60):
                        all_times.append(d1+'T'+f"{hour:02}"+':'+f"{minute:02}")
                if hour == h2_use:
                    for minute in range(0,m2_use):
                        all_times.append(d1+'T'+f"{hour:02}"+':'+f"{minute:02}")
                if ((hour>h1_use) & (hour<h2_use)):
                    for minute in range(0,60):
                        all_times.append(d1+'T'+f"{hour:02}"+':'+f"{minute:02}")
    else:
        for day in days:
            if day == d1:       
                for hour in range(h1_use,24):
                    if hour == h1_use:
                        for minute in range(m1_use,60):
                            all_times.append(day+'T'+f"{hour:02}"+':'+f"{minute:02}")
                    else:
                        for minute in range(0,60):
                            all_times.append(day+'T'+f"{hour:02}"+':'+f"{minute:02}")
            else:                   
                for hour in range(0,h2_use+1):
                    if hour == h2_use:
                        for minute in range(0,m2_use):
                            all_times.append(day+'T'+f"{hour:02}"+':'+f"{minute:02}")
                    else:
                        for minute in range(0,60):
                            all_times.append(day+'T'+f"{hour:02}"+':'+f"{minute:02}")
    #print(all_times)
    all_files = []

    for i in range(0,len(all_times)):
        proc=subprocess.Popen('ls -1 '+dir_files+all_times[i]+'*', shell=True, stdout=subprocess.PIPE)
        all_files = all_files+proc.communicate()[0].decode().split('\n')[0:-1][:]

    print('Number of files: ',len(all_files))
    print('')
    
    freq,nf = get_frequencies(all_files,dir_files,freq_s,freq_avg)
    print(len(freq))
      
    t_set,az_set,el_set,ra_set,dec_set,nt,noise,int_time,corrupt = get_times_and_coords(all_files)

    
    RR_set,LL_set,reRL_set,imRL_set = get_data_products(all_files,nt,nf,len(freq),freq_avg,freq_s,bintype='mean')
    
    if az_scan_trim == True:
        trim_flag, wkeep = trim_azimuth_scans(RR_set,LL_set,reRL_set,imRL_set,t1,t2,
                          t_set,az_set,el_set,ra_set,dec_set,noise,int_time,corrupt)
    else:
        trim_flag = np.zeros(len(t_set))
        wkeep = list(set(range(0,len(t_set))))
    
    #print(wkeep)
    make_new_file(outname,outfiles,all_files[0],RR_set,LL_set,reRL_set,imRL_set,
                  t_set,az_set,el_set,ra_set,dec_set,freq,noise,int_time,corrupt,trim_flag,wkeep)
    
    return None


def get_frequencies(all_files,dir_files,step,avg_bands):
    
    freq = []
    file = h5py.File(all_files[0],'r')
    beam = file['data']['beam_0']

    nf_raw = beam['band_SB3']['scan_0']['data'].shape[2]/step
    nf = int(nf_raw)

    print('------------------------------------------------------------------------')
    print('Number of frequency slices or bins (CAUTION: SHOULD BE WHOLE NUMBER): ',nf_raw)
    print('------------------------------------------------------------------------')
    for i, band_id in enumerate(beam.keys()):
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
    noise = []
    int_time = []
    corrupt = []
    
    for ifile in range(0,len(all_files)):
        print(ifile+1,all_files[ifile])
        file = h5py.File(all_files[ifile],'r')
        metadata = file['data']['beam_0']['band_SB3']['scan_0']['metadata']
        dec_set = np.concatenate([dec_set,metadata['declination']])
        ra_set = np.concatenate([ra_set,metadata['right_ascension']])
        el_set = np.concatenate([el_set,metadata['elevation']])
        az_set = np.concatenate([az_set,metadata['azimuth']])
        t_set = np.concatenate([t_set,metadata['utc']])
        noise = np.concatenate([noise,metadata['noise_state']])
        int_time = np.concatenate([int_time,metadata['integration_time']])
        corrupt = np.concatenate([corrupt,metadata['corrupted']])
        
    print(len(t_set))
    
    file.close()
    
    return t_set,az_set,el_set,ra_set,dec_set,nt,noise,int_time,corrupt


def get_data_products(all_files,nt,nf,nf_all,avg_bands,step,bintype='mean',*args,**kwargs):    

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
            #print(i,band_id)
            data = file['data']['beam_0'][band_id]['scan_0']['data']
            if avg_bands == False:
                RR_set[ifile*nt:(ifile+1)*nt,i*nf:(i+1)*nf] = data[:,1,::step]
                LL_set[ifile*nt:(ifile+1)*nt,i*nf:(i+1)*nf] = data[:,0,::step]
                reRL_set[ifile*nt:(ifile+1)*nt,i*nf:(i+1)*nf] = data[:,2,::step]
                imRL_set[ifile*nt:(ifile+1)*nt,i*nf:(i+1)*nf] = data[:,3,::step]
            else:
                if bintype == 'mean':
                    print('Using mean')
                    RR_set[ifile*nt:(ifile+1)*nt,i*nf:(i+1)*nf] = np.nanmean(data[:,1,:].reshape(-1,nf,step),axis=2)
                    LL_set[ifile*nt:(ifile+1)*nt,i*nf:(i+1)*nf] = np.nanmean(data[:,0,:].reshape(-1,nf,step),axis=2)
                    reRL_set[ifile*nt:(ifile+1)*nt,i*nf:(i+1)*nf] = np.nanmean(data[:,2,:].reshape(-1,nf,step),axis=2)
                    imRL_set[ifile*nt:(ifile+1)*nt,i*nf:(i+1)*nf] = np.nanmean(data[:,3,:].reshape(-1,nf,step),axis=2)
                if bintype == 'med':
                    print('Using median')
                    RR_set[ifile*nt:(ifile+1)*nt,i*nf:(i+1)*nf] = np.nanmedian(data[:,1,:].reshape(-1,nf,step),axis=2)
                    LL_set[ifile*nt:(ifile+1)*nt,i*nf:(i+1)*nf] = np.nanmedian(data[:,0,:].reshape(-1,nf,step),axis=2)
                    reRL_set[ifile*nt:(ifile+1)*nt,i*nf:(i+1)*nf] = np.nanmedian(data[:,2,:].reshape(-1,nf,step),axis=2)
                    imRL_set[ifile*nt:(ifile+1)*nt,i*nf:(i+1)*nf] = np.nanmedian(data[:,3,:].reshape(-1,nf,step),axis=2)
                    
        file.close()
       
    return RR_set,LL_set,reRL_set,imRL_set


def trim_azimuth_scans(RR_set,LL_set,reRL_set,imRL_set,t1,t2,
                      t_set,az_set,el_set,ra_set,dec_set,noise,int_time,corrupt):
    
    t_set_plt = Time(t_set, format='isot',scale='utc').mjd
    tstart_plt = Time(t1, format='isot',scale='utc').mjd
    tstop_plt = Time(t2, format='isot',scale='utc').mjd

    wbad_RA = []        
    for i in range(0,len(t_set)-1):
        if abs(ra_set[i]-ra_set[i+1]) > 0.05:
            wbad_RA.append(i)
            
    wbad_t = list(np.where( (t_set_plt < tstart_plt) | (t_set_plt > tstop_plt) ))[0]
    wkeep = list(np.where( (t_set_plt >= tstart_plt) & (t_set_plt <= tstop_plt) ))[0]
    
    wgood = list(set(range(0,len(t_set)))-set(wbad_RA)-set(wbad_t))
       
    fig1, axs = plt.subplots(1,1,figsize=(14,5))
    axs.scatter(t_set_plt,ra_set,s=40,color='lightgrey')
    axs.scatter(t_set_plt[wbad_RA],ra_set[wbad_RA],s=25,color='C1')
    axs.scatter(t_set_plt[wbad_t],ra_set[wbad_t],s=12,color='C2')
    axs.scatter(t_set_plt[wgood],ra_set[wgood],s=0.1,color='black')
    axs.tick_params(axis="x")
    axs.tick_params(axis="y")
    axs.set_xlabel('Time (UTC)')
    axs.set_ylabel('RA (hr)')
    ax2 = axs.twinx()
    ax2.scatter(t_set_plt,az_set,s=40,color='lightgrey')
    ax2.scatter(t_set_plt[wbad_RA],az_set[wbad_RA],s=25,color='C1')
    ax2.scatter(t_set_plt[wbad_t],az_set[wbad_t],s=12,color='C2')
    ax2.scatter(t_set_plt[wgood],az_set[wgood],s=0.1,color='darkblue')
    ax2.set_ylabel('Azimuth (deg)')
    ax2.plot([tstart_plt,tstop_plt],[0,0],color='black',linewidth=0.5)
    ax2.plot([tstart_plt,tstop_plt],[360,360],color='black',linewidth=0.5)
    axs.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    axs.fmt_xdata = mdates.DateFormatter('%H:%M:%S') 
    
    trim_flag = np.ones(len(t_set))
    trim_flag[wgood] = 0

    #print(trim_flag)

    return trim_flag, wkeep


def make_new_file(outname,outfiles,file_ex,RR_set,LL_set,reRL_set,imRL_set,
                  t_set,az_set,el_set,ra_set,dec_set,freq,noise,int_time,corrupt,trim_flag,wkeep):
    
    cmd2 = 'cp '+file_ex+' '+outfiles+outname+'.h5'
    os.system(cmd2)
    file = h5py.File(outfiles+outname+'.h5','r+')

    for i in range(3,8):
        del file['data']['beam_0']['band_SB'+str(i)]
    
    # Create band and scan groups:
    file['data']['beam_0'].create_group("band_SB0")
    file['data']['beam_0']['band_SB0'].create_group(f"scan_0")
    
    # Create power dataset:
    dat = np.empty((len(t_set[wkeep]), 4, len(freq)), dtype=float)
    file['data']['beam_0']['band_SB0']['scan_0'].create_dataset("data", data=dat)
    file['data']['beam_0']['band_SB0']['scan_0']['data'][:,0,:] = RR_set[wkeep,:]
    file['data']['beam_0']['band_SB0']['scan_0']['data'][:,1,:] = LL_set[wkeep,:]
    file['data']['beam_0']['band_SB0']['scan_0']['data'][:,2,:] = reRL_set[wkeep,:]
    file['data']['beam_0']['band_SB0']['scan_0']['data'][:,3,:] = imRL_set[wkeep,:]
    
    # Create metadata with timestamps and coordinates:
    metadata_content = [t_set[wkeep],az_set[wkeep],el_set[wkeep],ra_set[wkeep],
                        dec_set[wkeep],int_time[wkeep],corrupt[wkeep],noise[wkeep],trim_flag[wkeep]]  
    #print(noise)
    col_names = ["utc", "azimuth", "elevation", "right_ascension", "declination", "integration_time",
                 "corrupted","noise_state","trim_scan_flag"]
    col_types = np.dtype({'names':col_names,'formats':["S32", "f8", "f8", "f8", "f8", "f8", "?", "i8","i8"] } )
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