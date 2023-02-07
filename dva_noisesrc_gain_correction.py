import os
import h5py
import numpy as np
from astropy.time import Time
import matplotlib.pyplot as plt
#from scipy.interpolate import griddata
import datetime
import matplotlib.dates as mdates
from matplotlib.dates import HourLocator as HourLocator
from matplotlib.dates import MinuteLocator as MinuteLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from astropy import units as u
from astropy.time import TimeDelta
from scipy import interpolate
from operator import itemgetter
from itertools import groupby
#from astropy.convolution import convolve, Box1DKernel
import gc
from tqdm import tqdm
import warnings
import time
warnings.simplefilter('ignore')
import copy


def noise_source_correct(phase,day,Ne,Nm):
    
    ### Use these directories on elephant: ###################
    dir_in_rast  = '/srv/data/dva/survey_raster/'
    dir_out_rast = '/srv/data/dva/survey_raster_noise_corr/'
    dir_in_az    = '/srv/data/dva/survey_azimuth_scans/'
    dir_out_az    = '/srv/data/dva/survey_azimuth_scans_noise_corr/'
    ###########################################################
    
    log_file = open('/home/aordog/DVA_PLOTS/noise_gain_corr_new/noise_units_log_phase'+str(phase)+'_day'+f"{day:02}"+'.txt', 'a')
    
    st = time.time()
    
    # Read in persistent RFI mask:
    RFI_mask_idx = read_RFI_mask('/srv/data/dva/RFIpersist_mask/RFIpersist_mask.txt')
    
    # Read in start and end times of azimuth and raster scans
    # KEYS: 'scanid','azstart','azstop',
    #       'rast1start','rast2start','rast1stop','rast2stop'
    # (all times are in mjd)
    times = read_scan_start_stop(dir_in_az,dir_in_rast,phase,day,Ne,Nm,log_file)
    
    
    # Read in raw data along with start and stop indices for the raster scans
    # data KEYS: 'freq','RR','LL','reRL','imRL','dec','ra','el','az','t','tplt','noise'
    # rastix KEYS: 'idx1rast1,''idx2rast1,'idx1rast2','idx2rast2' 
    # (t in UTC; tplt in mjd)
    data,rastidx = read_in_files(times['scanid'],dir_in_az,dir_in_rast,phase,day,Ne,Nm,log_file)
    
    # Read in temperature data:
    temp_C, t_weath_plt = get_weather_data(phase)
    
    # Calculate noise source power deflection at each noise source instance **CHECK THIS CODE!
    # KEYS: 'LLdnoise','RRdnoise','reRLdnoise','imRLdnoise','tnoise'  
    # (tnoise in mjd)
    noisedata = noise_power_deflection(day,phase,data,log_file)
    
    # Remove outliers from noise source defelction data by setting them to NaN
    # KEYS: 'LLdnoise','RRdnoise','reRLdnoise','imRLdnoise','tnoise'  
    # (tnoise in mjd)
    noisedatafix = remove_noise_outliers(data,noisedata,phase,day,log_file)
    
    # Calculate polynomial fits to noise deflection vs time
    # KEYS: 'LLfit','RRfit','reRLfit','imRLfit',
    #       'LLfitpt','RRfitpt','reRLfitpt','imRLfitpt'
    # (pt = fits at noise source instances)
    fitvals = calculate_noise_fits(data,noisedatafix,phase,day,log_file)
    
    # Write out the noise-source-corrected versions of the data files  **CHECK THIS CODE!
    write_out_files(data,times,fitvals,rastidx,dir_in_az,dir_out_az,
                    dir_in_rast,dir_out_rast,phase,day,Ne,Nm,RFI_mask_idx,log_file)  
    
    # Plot fits and corrections:
    print_to_file('',log_file)
    print_to_file('---------------------------------------------------------------------------',log_file)
    print_to_file('Making plots for Phase '+str(phase)+', Day '+str(day)+'...',log_file)
    print_to_file('---------------------------------------------------------------------------',log_file)
    print_to_file('',log_file)
    plot_noise_spectrum(data,noisedata,phase,day)
    plot_fits(data,times,noisedata,noisedatafix,fitvals,t_weath_plt,temp_C,phase,day,Ne,Nm)   
    plot_example_correction(data,times,fitvals,phase,day,Ne,Nm)
    plot_before_after_spectrum(data,fitvals,phase,day,log_file)
    
    # Make waterfall plot:
    plot_waterfall(10*np.log10(data['LL']),10*np.log10(data['RR']),
                   data,times,phase,day,Ne,Nm,RFI_mask_idx,[65,65],[79,79],
                   ['LL power (dB)','RR power (dB)'],
                   ['viridis','viridis'],'waterfall_data_raw_total')
    print('done waterfall 1/5')
    
    plot_waterfall(data['reRL'],data['imRL'],
                   data,times,phase,day,Ne,Nm,RFI_mask_idx,[-1e6,-1e6],[1e6,1e6],
                   ['reRL (raw)','imRL(raw)'],
                   ['RdBu','RdBu'],'waterfall_power_raw_polar')
    print('done waterfall 2/5')
    
    plot_waterfall(10*np.log10(fitvals['LLfit']),10*np.log10(fitvals['RRfit']),
                   data,times,phase,day,Ne,Nm,RFI_mask_idx,[64,64],[74,74],
                   ['LL fitted noise deflection (dB)','RR fitted noise deflection (dB)'],
                   ['viridis','viridis'],'waterfall_noise_fit_total')
    print('done waterfall 3/5')
    
    plot_waterfall(fitvals['reRLfit'],fitvals['imRLfit'],
                   data,times,phase,day,Ne,Nm,RFI_mask_idx,[-1e7,-1e7],[1e7,1e7],
                   ['reRL fitted noise deflection (raw)','imRL fitted noise deflection (raw)'],
                   ['RdBu','RdBu'],'waterfall_noise_fit_polar')
    print('done waterfall 4/5')
    
    plot_waterfall(data['LL']/fitvals['LLfit'],data['RR']/fitvals['RRfit'],
                   data,times,phase,day,Ne,Nm,RFI_mask_idx,[1,1],[7,7],
                   ['LL power (noise units)','RR power (noise units)'],
                   ['viridis','viridis'],'waterfall_data_NSunits_total')
    print('done waterfall 5/5')
    
    et = time.time()
    
    print_to_file('',log_file)
    print_to_file('============================================================',log_file)
    print_to_file('Time for Day '+str(day)+', with '+str(Ne)+' raster(s) and '+str(len(times['scanid']))+' azimuth scans:',log_file)
    print_to_file(str((et-st)/60)+' minutes',log_file)
    print_to_file('============================================================',log_file)
    print_to_file('',log_file)

    log_file.close()
    
    return None


def print_to_file(print_string,log_file):
    print(print_string)
    print(print_string,file=log_file)
    
    return


def read_RFI_mask(RFI_mask_file):
    i = 0
    RFI_mask_idx = []
    with open(RFI_mask_file) as fp:
        for line in fp:
            if i>0: 
                RFI_mask_idx.append(int(line.split()[0]))
            i=i+1
    return RFI_mask_idx


def read_scan_start_stop(dir_in_az,dir_in_rast,phase,day,Ne,Nm,log_file):
    
    scan_id = []    # The scan id number
    scan_start = []  # Start time of the scan (UTC)
    scan_stop = []   # Stop time of the scan (UTC)

    raster1_start = []
    raster2_start = []
    raster1_stop = []
    raster2_stop = []

    rast_list = ['a','b']

    # Read in the azimuth scan data and store it in arrays:
    with open(dir_in_az+'DVAsurvey_phase'+str(phase)+'_day'+f"{day:03}"+'.txt') as fp:
        for line in fp:       
            scan_id.append(int(line.split()[0]))
            scan_start.append(line.split()[1]+'T'+line.split()[2][0:12])
            scan_stop.append(line.split()[3]+'T'+line.split()[4][0:12])
    
    if Ne == 0:
        raster1_start = ['0000-01-01T00:00:00.0']
        raster1_stop = ['0000-01-01T00:00:00.0']
    if Nm == 0:
        raster2_start = ['0000-01-01T00:00:00.0']
        raster2_stop = ['0000-01-01T00:00:00.0']
    
    for i in range(0,Ne):
    
        if Ne == 1:
            rast1name = 'dva_survey_phase'+str(phase)+'_day'+f"{day:03}"+'_raster1.txt'
        else:
            rast1name = 'dva_survey_phase'+str(phase)+'_day'+f"{day:03}"+'_raster1'+rast_list[i]+'.txt'
        
        with open(dir_in_rast+rast1name) as fp:
            for line in fp:  
                raster1_start.append(line.split()[3])
                raster1_stop.append(line.split()[4])
            
    for i in range(0,Nm):
    
        if Nm == 1:
            rast2name = 'dva_survey_phase'+str(phase)+'_day'+f"{day:03}"+'_raster2.txt'
        else:
            rast2name = 'dva_survey_phase'+str(phase)+'_day'+f"{day:03}"+'_raster2'+rast_list[i]+'.txt'
        
        with open(dir_in_rast+rast2name) as fp:
            for line in fp:  
                raster2_start.append(line.split()[3])
                raster2_stop.append(line.split()[4])
    
    print_to_file('',log_file)
    print_to_file('---------------------------------------------------------------------------',log_file)
    print_to_file('Reading in scan start and stop times for Phase '+str(phase)+', Day '+str(day)+'...',log_file)
    print_to_file('---------------------------------------------------------------------------',log_file)
    print_to_file('',log_file)
    print_to_file('evening raster:',log_file)
    print_to_file(str(raster1_start),log_file)
    print_to_file(str(raster1_stop),log_file)
    print_to_file('Azimuth scans:',log_file)
    for i in range(0,len(scan_id)):
        print_to_file(f"{i+1:02}"+' '+f"{scan_id[i]:04}"+' '+str(scan_start[i])+' '+str(scan_stop[i]),log_file)
    print_to_file('morning raster:',log_file)
    print_to_file(str(raster2_start),log_file)
    print_to_file(str(raster2_stop),log_file)

    # Convert start and stop times to Modified Julian Day (MJD) and create dictionary   
    times = {'scanid':scan_id,
             'azstart':Time(scan_start, format='isot',scale='utc').mjd,
             'azstop':Time(scan_stop,  format='isot',scale='utc').mjd,
             'rast1start':Time(raster1_start, format='isot',scale='utc').mjd,
             'rast2start':Time(raster2_start, format='isot',scale='utc').mjd,
             'rast1stop':Time(raster1_stop,  format='isot',scale='utc').mjd,
             'rast2stop':Time(raster2_stop,  format='isot',scale='utc').mjd}
        
    return times


def concatenate_data(file,RR,LL,reRL,imRL,dec,ra,el,az,t,noise):
    
    dataset = file['data']['beam_0']['band_SB0']['scan_0']
    
    idx1 = len(t)
    
    # Add the position and time data to the corresponding arrays:
    dec = np.concatenate([dec,dataset['metadata']['declination']])
    ra = np.concatenate([ra,dataset['metadata']['right_ascension']])
    el = np.concatenate([el,dataset['metadata']['elevation']])
    az = np.concatenate([az,dataset['metadata']['azimuth']])
    t = np.concatenate([t,dataset['metadata']['utc']])
    noise = np.concatenate([noise,dataset['metadata']['noise_state']])

    idx2 = len(t)
    
    # Add the spectrometer power data to the corresponding arrays:
    RR = np.concatenate([RR,dataset['data'][:,0,:]],axis=0)
    LL = np.concatenate([LL,dataset['data'][:,1,:]],axis=0)
    reRL = np.concatenate([reRL,dataset['data'][:,2,:]],axis=0)
    imRL = np.concatenate([imRL,dataset['data'][:,3,:]],axis=0)
    
    return RR,LL,reRL,imRL,dec,ra,el,az,t,noise,idx1,idx2


def read_in_files(scan_id,dir_in_az,dir_in_rast,phase,day,Ne,Nm,log_file):

    rast_list = ['a','b']
    t = []
    az = []
    dec = []
    ra = []
    el = []
    noise = []
    idx1_rast1 = []
    idx2_rast1 = []
    idx1_rast2 = []
    idx2_rast2 = []
    
    # Use one of the scans to get the list of frequencies:
    scan0 = f"{scan_id[0]:04}"
    file = h5py.File(dir_in_az+'dva_survey_phase'+str(phase)+'_raw_'+scan0+'.h5','r')
    freq = file['data']['beam_0']['band_SB0']['frequency'][:]/1e6

    # Create empty arrays for the power data:
    RR = np.empty([0,len(freq)])
    LL = np.empty([0,len(freq)])
    reRL = np.empty([0,len(freq)])
    imRL = np.empty([0,len(freq)])
    
    print_to_file('',log_file)
    print_to_file('---------------------------------------------------------------------------',log_file)
    print_to_file('Reading in files for Phase '+str(phase)+', Day '+str(day)+'...',log_file)
    print_to_file('---------------------------------------------------------------------------',log_file)
    print_to_file('',log_file)

    # Raster scan 1:
    print_to_file('evening raster:',log_file)
    for i in range(0,Ne):
        if Ne == 1:
            file = h5py.File(dir_in_rast+'dva_survey_phase'+str(phase)+'_day'+f"{day:03}"+'_raster1'+'.h5','r')
        else:
            file = h5py.File(dir_in_rast+'dva_survey_phase'+str(phase)+'_day'+f"{day:03}"+'_raster1'+rast_list[i]+'.h5','r')

        rast1_file_t = [Time(file['data']['beam_0']['band_SB0']['scan_0']['metadata']['utc'][0],
                            format='isot',scale='utc').mjd,
                        Time(file['data']['beam_0']['band_SB0']['scan_0']['metadata']['utc'][-1],
                            format='isot',scale='utc').mjd]
        RR,LL,reRL,imRL,dec,ra,el,az,t,noise,idx1,idx2 = concatenate_data(file,RR,LL,reRL,imRL,
                                                                          dec,ra,el,az,t,noise)
        print_to_file(str(file),log_file)
        idx1_rast1.append(idx1)
        idx2_rast1.append(idx2)

    # Loop through all the scans in the "scan_num" list:
    print_to_file('azimuth scans:',log_file)
    for i in scan_id:
        file = h5py.File(dir_in_az+'dva_survey_phase'+str(phase)+'_raw_'+f"{i:04}"+'.h5','r')
        print_to_file(str(i)+' '+str(file),log_file)
        RR,LL,reRL,imRL,dec,ra,el,az,t,noise,idx1,idx2 = concatenate_data(file,RR,LL,reRL,imRL,
                                                                          dec,ra,el,az,t,noise)
    
    # Raster scan 2:
    print_to_file('morning raster:',log_file)
    for i in range(0,Nm):
        if Nm == 1:
            file = h5py.File(dir_in_rast+'dva_survey_phase'+str(phase)+'_day'+f"{day:03}"+'_raster2'+'.h5','r')
        else:
            file = h5py.File(dir_in_rast+'dva_survey_phase'+str(phase)+'_day'+f"{day:03}"+'_raster2'+rast_list[i]+'.h5','r')

        rast2_file_t = [Time(file['data']['beam_0']['band_SB0']['scan_0']['metadata']['utc'][0],
                            format='isot',scale='utc').mjd,
                        Time(file['data']['beam_0']['band_SB0']['scan_0']['metadata']['utc'][-1],
                            format='isot',scale='utc').mjd]
        RR,LL,reRL,imRL,dec,ra,el,az,t,noise,idx1,idx2 = concatenate_data(file,RR,LL,reRL,imRL,
                                                                          dec,ra,el,az,t,noise)
        print_to_file(str(file),log_file)
        idx1_rast2.append(idx1)
        idx2_rast2.append(idx2)
        
    t_set_plt = Time(t, format='isot',scale='utc').mjd
        
    data = {'freq':freq, 'RR':RR, 'LL':LL, 'reRL':reRL, 'imRL':imRL,
            'dec':dec, 'ra':ra, 'el':el, 'az':az, 't':t, 'tplt':t_set_plt,'noise':noise}
    rastidx = {'idx1rast1':idx1_rast1, 'idx2rast1':idx2_rast1, 
               'idx1rast2':idx1_rast2, 'idx2rast2':idx2_rast2}
    
    del RR, LL, reRL, imRL
    gc.collect()
           
    return data,rastidx


def month_to_num(month_name):
    if month_name == 'Jan': month_num = '01'
    if month_name == 'Feb': month_num = '02'
    if month_name == 'Mar': month_num = '03'
    if month_name == 'Apr': month_num = '04'
    if month_name == 'May': month_num = '05'
    if month_name == 'Jun': month_num = '06'
    if month_name == 'Jul': month_num = '07'
    if month_name == 'Aug': month_num = '08'
    if month_name == 'Sep': month_num = '09'
    if month_name == 'Oct': month_num = '10'
    if month_name == 'Nov': month_num = '11'
    if month_name == 'Dec': month_num = '12'
    return(month_num)

def get_weather_data(phase):
    i = 0
    t_weath = []
    temp_C = []

    with open('/srv/data/dva/weather_data/weather_survey_phase'+str(phase)+'.txt') as fp:
        for line in fp:
            t_weath.append(str( line.split()[2]+'-'+month_to_num(line.split()[1])+'-'+line.split()[0]+'T'+line.split()[3]))
            temp_C.append(line.split()[4])        

    temp_C = np.array(temp_C,dtype=float)
    t_weath_fix = Time(t_weath, format='isot',scale='utc')
    t_weath_plt = t_weath_fix.mjd
    
    return temp_C, t_weath_plt


def noise_power_deflection(day,phase,data,log_file,n_off=5,*args,**kwargs):
    
    # Make arrays for noise source deflection:
    LL_dnoise = []
    RR_dnoise = []
    reRL_dnoise = []
    imRL_dnoise = []
    t_noise = []

    wnoise = np.where(data['noise'] == 1)[0]
    print_to_file('',log_file)
    print_to_file('Calculating noise source power deflection for',log_file)
    print_to_file(str(len(wnoise))+' noise source instances in Phase'+str(phase)+', Day '+str(day)+'...',log_file)
    print_to_file('',log_file)
    for k,g in groupby(enumerate(wnoise),lambda x:x[0]-x[1]):

        try:
            group = np.array(list(map(itemgetter(1),g)))
   
            t_noise.append(np.nanmedian(data['tplt'][group[1:-1]]))

            LL_dnoise.append(calc_on_off_diff(group,n_off,data['LL']))
            RR_dnoise.append(calc_on_off_diff(group,n_off,data['RR']))
            reRL_dnoise.append(calc_on_off_diff(group,n_off,data['reRL']))
            imRL_dnoise.append(calc_on_off_diff(group,n_off,data['imRL']))
        
        except:
            pass

    noisedata = {'LLdnoise':np.empty([len(LL_dnoise),len(LL_dnoise[0])]),
                 'RRdnoise':np.empty([len(RR_dnoise),len(RR_dnoise[0])]),
                 'reRLdnoise':np.empty([len(reRL_dnoise),len(reRL_dnoise[0])]),
                 'imRLdnoise':np.empty([len(imRL_dnoise),len(imRL_dnoise[0])]),
                 'tnoise':t_noise}

    for i in range(0,len(LL_dnoise)):
        noisedata['LLdnoise'][i,:] = LL_dnoise[i]
        noisedata['RRdnoise'][i,:] = RR_dnoise[i]
        noisedata['reRLdnoise'][i,:] = reRL_dnoise[i]
        noisedata['imRLdnoise'][i,:] = imRL_dnoise[i]
    
    del RR_dnoise, LL_dnoise, reRL_dnoise, imRL_dnoise
    gc.collect()
   
    return noisedata


def calc_on_off_diff(group,n_off,data_arr):
    
    #print_to_file(group)
    #print_to_file(group[1:-1])
    #middle = [group[int(np.floor((len(group)-1)/2))] ,group[int(np.ceil((len(group)-1)/2))]]
    #print_to_file(middle)
    offleft = [group[0]-n_off-1,group[0]-2]
    offright = [group[-1]+2,group[-1]+n_off+1]
    #print_to_file(offleft)
    #print_to_file(offright)
    #print_to_file('')
    
    data_noise = np.nanmedian(data_arr[group[1:-1],:],axis=0)
    leftoff  = np.nanmedian(data_arr[offleft[0]:offleft[-1],:],axis=0)
    rightoff = np.nanmedian(data_arr[offright[0]:offright[-1],:],axis=0)
    data_off = (leftoff + rightoff)/2.
    
    data_diff = data_noise - data_off
    
    return data_diff



def plot_noise_instances(times,data,noisedata,rastidx,phase,day):

    fig1, axs = plt.subplots(3,1,figsize=(16,8))

    ii = 7000
    #print_to_file('freq = ',data['freq'][ii])
    
    meannoise = np.nanmean(noisedata['LLdnoise'][:,ii])

    axs[0].set_xlim(times['rast1start'][0],times['rast1stop'][-1])
    axs2 = axs[0].twinx()
    axs2.plot(data['tplt'],data['dec'],color='k')
    axs2.set_ylim(5,65)
    axs2.set_ylabel('dec.')

    axs[1].set_xlim(times['azstart'][0],times['azstop'][-1])
    axs2 = axs[1].twinx()
    axs2.scatter(data['tplt'],data['az'],color='k',s=0.01)
    axs2.set_ylim(-20,380)
    axs2.set_ylabel('az.')

    axs[2].set_xlim(times['rast2start'][0],times['rast2stop'][-1])
    axs2 = axs[2].twinx()
    axs2.plot(data['tplt'],data['dec'],color='k')
    axs2.set_ylim(5,65)
    axs2.set_ylabel('dec.')

    for i in range(0,3):
        axs[i].scatter(data['tplt'],data['noise'],s=5)
        axs[i].scatter(noisedata['tnoise'],noisedata['LLdnoise'][:,ii]/meannoise,s=10,c='red')
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        axs[i].fmt_xdata = mdates.DateFormatter('%H:%M:%S')
        axs[i].set_ylim(-0.1,1.2)
        axs[i].set_ylabel('noise (fraction of mean)')
    axs[2].set_xlabel('UTC')
    
    plt.tight_layout()
    plt.savefig('/home/aordog/DVA_PLOTS/noise_gain_corr_new/noise_firing_phase'+str(phase)+'_day'+f"{day:02}"+'.png')
    plt.close()
    
    return


def remove_noise_outliers(data,noisedata,phase,day,log_file,window=10,*args,**kwargs):
    
    print_to_file('Removing noise source outliers for Phase'+str(phase)+', Day '+str(day)+'...',log_file)
    print_to_file('',log_file)
    
    noisedatafix = copy.deepcopy(noisedata)
    
    for i in tqdm(range(0,len(data['freq']))):
    #for i in range(0,len(data['freq'])):
        try:
            for j in range(0,len(noisedata['tnoise'])):

                if ((j >= window) & (j < len(noisedata['tnoise'])-window)):
                    LL_near = np.nanmedian(noisedata['LLdnoise'][j-window:j+window,i])
                    RR_near = np.nanmedian(noisedata['RRdnoise'][j-window:j+window,i])
                else:
                    if j < window:
                        LL_near = np.nanmedian(noisedata['LLdnoise'][0:j+window,i])
                        RR_near = np.nanmedian(noisedata['RRdnoise'][0:j+window,i])
                    if j >= len(noisedata['tnoise'])-window:
                        LL_near = np.nanmedian(noisedata['LLdnoise'][j-window:-1,i])
                        RR_near = np.nanmedian(noisedata['RRdnoise'][j-window:-1,i])
        
                if ( noisedata['LLdnoise'][j,i] > 1.1*LL_near):
                    noisedatafix['LLdnoise'][j,i] = np.nan
                    noisedatafix['reRLdnoise'][j,i] = np.nan
                    noisedatafix['imRLdnoise'][j,i] = np.nan
                if ( noisedata['LLdnoise'][j,i] < 0.9*LL_near):
                    noisedatafix['LLdnoise'][j,i] = np.nan
                    noisedatafix['reRLdnoise'][j,i] = np.nan
                    noisedatafix['imRLdnoise'][j,i] = np.nan
            
                if ( noisedata['RRdnoise'][j,i] > 1.1*RR_near):
                    noisedatafix['RRdnoise'][j,i] = np.nan
                    noisedatafix['reRLdnoise'][j,i] = np.nan
                    noisedatafix['imRLdnoise'][j,i] = np.nan
                if ( noisedata['RRdnoise'][j,i] < 0.9*RR_near):
                    noisedatafix['RRdnoise'][j,i] = np.nan
                    noisedatafix['reRLdnoise'][j,i] = np.nan
                    noisedatafix['imRLdnoise'][j,i] = np.nan
        except:
            pass
                
    return noisedatafix


def calculate_noise_fits(data,noisedatafix,phase,day,log_file,order=9,*args,**kwargs):
    
    print_to_file('',log_file)
    print_to_file('Calculating the noise deflection polynomial fits for Phase'+str(phase)+', Day '+str(day)+'...',log_file)
    print_to_file('',log_file)
    
    LL_fit = np.empty_like(data['LL'])
    RR_fit = np.empty_like(data['RR'])
    reRL_fit = np.empty_like(data['reRL'])
    imRL_fit = np.empty_like(data['imRL'])

    LL_fit_noise_pt = np.empty_like(noisedatafix['LLdnoise'])
    RR_fit_noise_pt = np.empty_like(noisedatafix['RRdnoise'])
    reRL_fit_noise_pt = np.empty_like(noisedatafix['reRLdnoise'])
    imRL_fit_noise_pt = np.empty_like(noisedatafix['imRLdnoise'])
    
    t_mean = np.nanmean(data['tplt'])

    for i in tqdm(range(0,len(data['freq']))):

        try:
            wuse = np.where(np.isfinite(noisedatafix['LLdnoise'][:,i]))[0]
            pL = np.polyfit(np.array(noisedatafix['tnoise'])[wuse]-t_mean,noisedatafix['LLdnoise'][wuse,i],order)
            LL_fit[:,i] = np.polyval(pL, data['tplt']-t_mean)
            LL_fit_noise_pt[:,i] = np.polyval(pL, noisedatafix['tnoise']-t_mean)

            wuse = np.where(np.isfinite(noisedatafix['RRdnoise'][:,i]))[0]
            pR = np.polyfit(np.array(noisedatafix['tnoise'])[wuse]-t_mean,noisedatafix['RRdnoise'][wuse,i],order)
            RR_fit[:,i] = np.polyval(pR, data['tplt']-t_mean)
            RR_fit_noise_pt[:,i] = np.polyval(pR, noisedatafix['tnoise']-t_mean)
    
            wuse = np.where(np.isfinite(noisedatafix['reRLdnoise'][:,i]))[0]
            pre = np.polyfit(np.array(noisedatafix['tnoise'])[wuse]-t_mean,noisedatafix['reRLdnoise'][wuse,i],order)
            reRL_fit[:,i] = np.polyval(pre, data['tplt']-t_mean)
            reRL_fit_noise_pt[:,i] = np.polyval(pre, noisedatafix['tnoise']-t_mean)
    
            wuse = np.where(np.isfinite(noisedatafix['imRLdnoise'][:,i]))[0]
            pim = np.polyfit(np.array(noisedatafix['tnoise'])[wuse]-t_mean,noisedatafix['imRLdnoise'][wuse,i],order)
            imRL_fit[:,i] = np.polyval(pim, data['tplt']-t_mean)
            imRL_fit_noise_pt[:,i] = np.polyval(pim, noisedatafix['tnoise']-t_mean)
        except:
            pass
    
    fitvals = {'LLfit':LL_fit, 'RRfit':RR_fit, 'reRLfit':reRL_fit, 'imRLfit':imRL_fit,
               'LLfitpt':LL_fit_noise_pt, 'RRfitpt':RR_fit_noise_pt,
               'reRLfitpt':reRL_fit_noise_pt, 'imRLfitpt':imRL_fit_noise_pt}
    
    del RR_fit, LL_fit, reRL_fit, imRL_fit
    del LL_fit_noise_pt, RR_fit_noise_pt, reRL_fit_noise_pt, imRL_fit_noise_pt
    gc.collect()
    
    return fitvals


def plot_fits(data,times,noisedata,noisedatafix,fitvals,t_weath_plt,temp_C,phase,day,Ne,Nm):
    
    fplot = 800
    df = data['freq'][1]-data['freq'][0]
    wf = np.where(abs(data['freq']-fplot)<df/2)[0][0]
    #print_to_file(data['freq'][wf])

    fig1, axs = plt.subplots(6,1,figsize=(16,20))

    axs[0].scatter(noisedata['tnoise'],10*np.log10(noisedata['RRdnoise'][:,wf]),s=20,color='red')
    axs[0].scatter(noisedatafix['tnoise'],10*np.log10(noisedatafix['RRdnoise'][:,wf]),s=20,color='C1',label='RR',zorder=10)
    axs[0].scatter(noisedata['tnoise'],10*np.log10(noisedata['LLdnoise'][:,wf]),s=20,color='blue')
    axs[0].scatter(noisedatafix['tnoise'],10*np.log10(noisedatafix['LLdnoise'][:,wf]),s=20,color='C0',label='LL',zorder=10)
    axs[0].plot(data['tplt'],10*np.log10(fitvals['LLfit'][:,wf]),color='blue',linewidth=2,zorder=100)
    axs[0].plot(data['tplt'],10*np.log10(fitvals['RRfit'][:,wf]),color='red',linewidth=2,zorder=100)
    axs[0].set_ylim(65,69)
    axs[0].set_ylabel('Noise source deflection (dB)')
    axs[0].legend(markerscale=2)
    ax2 = axs[0].twinx()
    ax2.plot(t_weath_plt,temp_C,color='purple',linewidth=2)
    ax2.set_ylim(-25,35)
    ax2.set_ylabel('Outside Temperature')
    
    axs[3].scatter(noisedata['tnoise'],noisedata['reRLdnoise'][:,wf],s=20,color='red')
    axs[3].scatter(noisedatafix['tnoise'],noisedatafix['reRLdnoise'][:,wf],s=20,color='C1',label='reRL',zorder=10)
    axs[3].scatter(noisedata['tnoise'],noisedata['imRLdnoise'][:,wf],s=20,color='blue')
    axs[3].scatter(noisedatafix['tnoise'],noisedatafix['imRLdnoise'][:,wf],s=20,color='C0',label='imRL',zorder=10)
    axs[3].plot(data['tplt'],fitvals['reRLfit'][:,wf],color='red',linewidth=2,zorder=100)
    axs[3].plot(data['tplt'],fitvals['imRLfit'][:,wf],color='blue',linewidth=2,zorder=100)
    axs[3].set_ylabel('Noise source deflection (dB)')
    axs[3].legend(markerscale=2)
    ax2 = axs[3].twinx()
    ax2.plot(t_weath_plt,temp_C,color='purple',linewidth=2)
    ax2.set_ylim(-25,35)
    ax2.set_ylabel('Outside Temperature')
    
    ii = [1,2,4,5]
    points = (noisedatafix['RRdnoise'],noisedatafix['LLdnoise'],
              noisedatafix['reRLdnoise'],noisedatafix['imRLdnoise'])
    fits = (fitvals['RRfitpt'],fitvals['LLfitpt'],fitvals['reRLfitpt'],fitvals['imRLfitpt'])
    color = ('C1','C0','C1','C0')
    labels = ('RR','LL','reRL','imRL')
    
    for i in range(0,4):
        axs[ii[i]].scatter(noisedatafix['tnoise'],(points[i][:,wf]-fits[i][:,wf])/points[i][:,wf],
               color=color[i],zorder=10,label=labels[i])
        axs[ii[i]].legend()
        axs[ii[i]].set_ylim(-0.15,0.15)
        axs[ii[i]].fill_between(data['tplt'],-0.05,0.05,zorder=0,color='grey',alpha=0.4)
        axs[ii[i]].fill_between(data['tplt'],-0.01,0.01,zorder=1,color='grey',alpha=0.4)
        axs[ii[i]].set_ylabel('Fractional residuals')

    for i in range(0,6):
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        axs[i].fmt_xdata = mdates.DateFormatter('%H:%M:%S')
        if Ne == 0:
            axs[i].set_xlim(times['azstart'][0],times['rast2stop'][-1])
        if Nm == 0:
            axs[i].set_xlim(times['rast1start'][0],times['azstop'][-1])        
        if ((Nm > 0) & (Ne > 0)):
            axs[i].set_xlim(times['rast1start'][0],times['rast2stop'][-1])        
        axs[i].grid()
    axs[5].set_xlabel('UTC')
    
    plt.tight_layout()
    plt.savefig('/home/aordog/DVA_PLOTS/noise_gain_corr_new/time_noise_fits_800MHz_phase'+str(phase)+'_day'+f"{day:02}"+'.png')
    plt.close()

    return
   
    
def plot_example_correction(data,times,fitvals,phase,day,Ne,Nm):
    
    fplot = 800
    df = data['freq'][1]-data['freq'][0]
    wf = np.where(abs(data['freq']-fplot)<df/2)[0][0]
    #print_to_file(data['freq'][wf])

    lims_raw1 = [66,66,66]
    lims_raw2 = [76,76,76]
    lims_cor1 = [1,1,1]
    lims_cor2 = [7,7,7]

    sz=1
    fig1, axs = plt.subplots(3,1,figsize=(20,12))

    
    if Ne == 0:
        j1 = 1
        j2 = 3
    if Nm == 0:
        j1 = 0
        j2 = 2
    if ((Nm > 0) & (Ne > 0)):
        j1 = 0
        j2 = 3
    
    for i in range(j1,j2):
        axs[i].scatter(data['tplt'],10*np.log10(data['LL'][:,wf]),s=sz,color='black')
        axs[i].set_ylim(lims_raw1[i],lims_raw2[i])
        
        ax2 = axs[i].twinx()
        ax2.scatter(data['tplt'],data['LL'][:,wf]/fitvals['LLfit'][:,wf],s=sz,color='purple')
        ax2.set_ylim(lims_cor1[i],lims_cor2[i])
        ax2.set_ylabel('Corrected power (noise src units; purple)')
    
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        axs[i].fmt_xdata = mdates.DateFormatter('%H:%M:%S')
        axs[i].grid()
        axs[i].set_ylabel('Raw power (dB; black)')

    if Ne == 0:    
        axs[2].set_xlim(times['rast2start'][0],times['rast2stop'][-1])
    if Nm == 0:    
        axs[0].set_xlim(times['rast1start'][0],times['rast1stop'][-1])
    if ((Nm > 0) & (Ne > 0)):
        axs[0].set_xlim(times['rast1start'][0],times['rast1stop'][-1])
        axs[2].set_xlim(times['rast2start'][0],times['rast2stop'][-1])
    axs[1].set_xlim(times['azstart'][0],times['azstop'][-1])
    
    
    for i in range(0,len(times['scanid'])):
        print(times['azstart'][i])
        print(min(abs(data['tplt']-times['azstart'][i])),max(abs(data['tplt']-times['azstart'][i])))
        print(min(data['tplt']),max(data['tplt']))
        print('')
        elhere = data['el'][abs(data['tplt']-times['azstart'][i])<1e-4][0]
        #elhere = data['el'][abs(data['tplt']-times['azstart'][i])<0.007][0]
        if abs(elhere-49.32)<0.5:
            clr = 'C1'
        elif abs(elhere-20.0)<0.5:
            clr = 'C2'
        axs[1].axvspan(times['azstart'][i],times['azstop'][i],color=clr,alpha=0.1)

    axs[0].set_title('Raw to noise source units at 800 MHz')
    plt.tight_layout()
    plt.savefig('/home/aordog/DVA_PLOTS/noise_gain_corr_new/time_datacorrect_800MHz_phase'+str(phase)+'_day'+f"{day:02}"+'.png')
    
    return



def plot_noise_spectrum(data,noisedata,phase,day,t_noise_idx=20,*args,**kwargs):
    
    fig1, axs = plt.subplots(2,1,figsize=(16,10))

    axs[0].scatter(data['freq'],10*np.log10(noisedata['LLdnoise'][t_noise_idx,:]),s=1,label='LL')
    axs[0].scatter(data['freq'],10*np.log10(noisedata['RRdnoise'][t_noise_idx,:]),s=1,label='RR')
    PI = np.sqrt((noisedata['reRLdnoise'][t_noise_idx,:])**2+(noisedata['imRLdnoise'][t_noise_idx,:])**2)
    axs[0].scatter(data['freq'],10*np.log10(PI),s=1,label='PI')
    axs[0].set_ylim(55,75)
    axs[0].set_ylabel('Power deflection (dB)')

    axs[1].scatter(data['freq'],noisedata['reRLdnoise'][t_noise_idx,:],s=1,label='reRL')
    axs[1].scatter(data['freq'],noisedata['imRLdnoise'][t_noise_idx,:],s=1,label='imRL')
    axs[1].set_ylim(-2e7,2e7)
    axs[1].set_xlabel('Frequency (MHz)')
    axs[1].set_ylabel('Power deflection (spectrometer units)')
    ax2 = axs[1].twinx()
    ax2.scatter(data['freq'],(180./np.pi)*0.5*np.arctan2(noisedata['imRLdnoise'][t_noise_idx,:],
                                                         noisedata['reRLdnoise'][t_noise_idx,:]),
                s=1,color='C2',label='pol. angle')
    ax2.set_ylim(-90,90)
    ax2.set_ylabel('Polarisation angle (degrees) [green]')
    
    for i in range(0,2):
        axs[i].grid()
        axs[i].set_xlim(350,1030)
        axs[i].legend(markerscale=10)

    plt.tight_layout()
    plt.savefig('/home/aordog/DVA_PLOTS/noise_gain_corr_new/spectrum_noise_example_phase'+str(phase)+'_day'+f"{day:02}"+'.png')
    plt.close()
    
    return


def plot_before_after_spectrum(data,fitvals,phase,day,log_file):

    fig1, axs = plt.subplots(2,1,figsize=(16,10))

    t_idx = 5000
    print_to_file('Timestamp for example data spectrum: '+str(data['t'][t_idx]),log_file)
    s = 0.5

    axs[0].scatter(data['freq'],10*np.log10(data['LL'][t_idx,:]),s=s,color='C0',label='LL')
    axs[0].scatter(data['freq'],10*np.log10(data['RR'][t_idx,:]),s=s,color='C1',label='RR')
    ax2 = axs[0].twinx()
    PI = np.sqrt((data['reRL'][t_idx,:])**2+(data['imRL'][t_idx])**2)
    ax2.scatter(data['freq'],10*np.log10(PI),s=s,color='C2',label='PI')   
    axs[0].set_ylim(65,80)
    ax2.set_ylim(20,80)   
    axs[0].set_ylabel('Raw power (dB)')
    ax2.legend(markerscale=10,loc='lower right')

    axs[1].scatter(data['freq'],data['LL'][t_idx,:]/fitvals['LLfit'][t_idx,:],s=s,color='blue',label='LL')
    axs[1].scatter(data['freq'],data['RR'][t_idx,:]/fitvals['RRfit'][t_idx,:],s=s,color='red',label='RR')
    ax2 = axs[1].twinx()
    PI = np.sqrt((data['reRL'][t_idx,:]/fitvals['reRLfit'][t_idx,:])**2+(data['imRL'][t_idx]/fitvals['imRLfit'][t_idx])**2)
    ax2.scatter(data['freq'],PI,s=s,color='green',label='PI')
    ax2.set_ylim(-1,2)
    axs[1].set_ylim(0,10)
    axs[1].set_ylabel('Corrected power (noise source units)')
    axs[1].set_xlabel('Frequency (MHz)')
    ax2.legend(markerscale=10,loc='lower right')
    
    for i in range(0,2):
        axs[i].grid()
        axs[i].set_xlim(350,1030)
        axs[i].legend(markerscale=10,loc='upper right')

    plt.tight_layout()
    plt.savefig('/home/aordog/DVA_PLOTS/noise_gain_corr_new/spectrum_data_example_phase'+str(phase)+'_day'+f"{day:02}"+'.png')
    plt.close()
    
    return


def write_out_files(data,times,fitvals,rastidx,dir_in_az,dir_out_az,
                    dir_in_rast,dir_out_rast,phase,day,Ne,Nm,RFI_mask_idx,log_file):
    
    rast_list = ['a','b']
    
    PA_cal = np.empty(data['LL'].shape)
    PI_cal = np.empty(data['LL'].shape)
    print_to_file('',log_file)
    print_to_file('---------------------------------------------------------------------------',log_file)
    print_to_file('Writing out files for Phase '+str(phase)+', Day '+str(day)+'...',log_file)
    print_to_file('---------------------------------------------------------------------------',log_file)
    print_to_file(str(PA_cal.shape),log_file)
    
    # Raster scan 1:
    for i in range(0,Ne):   
        if Ne == 1:
            inname  = 'dva_survey_phase'+str(phase)+'_day'+f"{day:03}"+'_raster1'+'.h5'
            outname = 'dva_survey_phase'+str(phase)+'_day'+f"{day:03}"+'_raster1_gain_corr'
        else:
            inname  = 'dva_survey_phase'+str(phase)+'_day'+f"{day:03}"+'_raster1'+rast_list[i]+'.h5'
            outname = 'dva_survey_phase'+str(phase)+'_day'+f"{day:03}"+'_raster1'+rast_list[i]+'_gain_corr'
        
        file = h5py.File(dir_in_rast+inname,'r')       
        cmd2 = 'cp '+dir_in_rast+inname+' '+dir_out_rast+outname+'.h5'
        print_to_file('',log_file)
        print_to_file('Evening raster(s)',log_file)
        os.system(cmd2)
        file_new = h5py.File(dir_out_rast+outname+'.h5','r+')
        
        i1 = rastidx['idx1rast1'][i]
        i2 = rastidx['idx2rast1'][i]
        PI_cal[i1:i2,:],PA_cal[i1:i2,:] = gain_correct(file_new,data,fitvals,i1,i2)

    # Raster scan 2:
    for i in range(0,Nm):   
        if Nm == 1:
            inname  = 'dva_survey_phase'+str(phase)+'_day'+f"{day:03}"+'_raster2'+'.h5'
            outname = 'dva_survey_phase'+str(phase)+'_day'+f"{day:03}"+'_raster2_gain_corr'
        else:
            inname  = 'dva_survey_phase'+str(phase)+'_day'+f"{day:03}"+'_raster2'+rast_list[i]+'.h5'
            outname = 'dva_survey_phase'+str(phase)+'_day'+f"{day:03}"+'_raster2'+rast_list[i]+'_gain_corr'
        
        file = h5py.File(dir_in_rast+inname,'r')       
        cmd2 = 'cp '+dir_in_rast+inname+' '+dir_out_rast+outname+'.h5'
        print_to_file('',log_file)
        print_to_file('Morning raster(s)',log_file)
        os.system(cmd2)
        file_new = h5py.File(dir_out_rast+outname+'.h5','r+')
        
        i1 = rastidx['idx1rast2'][i]
        i2 = rastidx['idx2rast2'][i]
        PI_cal[i1:i2,:],PA_cal[i1:i2,:] = gain_correct(file_new,data,fitvals,i1,i2)
        
    # Azimuth scans:
    for i in times['scanid']:
        inname  = 'dva_survey_phase'+str(phase)+'_raw_'+f"{i:04}"+'.h5'
        outname = 'dva_survey_phase'+str(phase)+'_raw_'+f"{i:04}"+'_gain_corr'
        
        file = h5py.File(dir_in_az+inname,'r')       
        cmd2 = 'cp '+dir_in_az+inname+' '+dir_out_az+outname+'.h5'
        #print_to_file(cmd2)
        os.system(cmd2)
        file_new = h5py.File(dir_out_az+outname+'.h5','r+')
        print_to_file('',log_file)
        print_to_file(str(i)+' '+str(file),log_file)
        t_az = Time(file['data']['beam_0']['band_SB0']['scan_0']['metadata']['utc'][:],
                        format='isot',scale='utc').mjd
        w = np.where((data['tplt'] >= t_az[0]) & (data['tplt'] <= t_az[-1]))[0]
        #print_to_file(w)
        #print_to_file('')
        PI_cal[w[0]:w[-1]+1,:],PA_cal[w[0]:w[-1]+1,:] = gain_correct(file_new,data,fitvals,w[0],w[-1]+1)
        
    plot_waterfall(PI_cal,PA_cal,
                   data,times,phase,day,Ne,Nm,RFI_mask_idx,[0,-90],[1e7,90],
                   ['noise diode PI (raw)','noise diode pol. angle (deg)'],
                   ['viridis','twilight'],'waterfall_noise_PI_PA')
    
    return


def gain_correct(file_new,data,fitvals,i1,i2):
    
    file_new['data']['beam_0']['band_SB0']['scan_0']['data'][:,0,:] = data['RR'][i1:i2,:]/fitvals['RRfit'][i1:i2,:]
    file_new['data']['beam_0']['band_SB0']['scan_0']['data'][:,1,:] = data['LL'][i1:i2,:]/fitvals['LLfit'][i1:i2,:]
       
    PAcal = 0.5*np.arctan2(fitvals['imRLfit'][i1:i2,:],fitvals['reRLfit'][i1:i2,:])
    PIcal = np.sqrt(fitvals['imRLfit'][i1:i2,:]**2 + fitvals['reRLfit'][i1:i2,:]**2)
    
    #print_to_file(data['freq'][5500])
    #print_to_file(min((180./np.pi)*PAcal[:,5500]),max((180./np.pi)*PAcal[:,5500]))

    reRL_corr = ( data['reRL'][i1:i2,:]*np.cos(2*PAcal) + data['imRL'][i1:i2,:]*np.sin(2*PAcal) )/PIcal
    imRL_corr = ( data['imRL'][i1:i2,:]*np.cos(2*PAcal) - data['reRL'][i1:i2,:]*np.sin(2*PAcal) )/PIcal
        
    file_new['data']['beam_0']['band_SB0']['scan_0']['data'][:,2,:] = reRL_corr
    file_new['data']['beam_0']['band_SB0']['scan_0']['data'][:,3,:] = imRL_corr
    
    file_new.close()
    
    return PIcal,PAcal*180./np.pi


def plot_waterfall(data_plot1,data_plot2,data,times,phase,day,Ne,Nm,RFI_mask_idx,vmin,vmax,
                   cbartitle,cmap,plotname,filetype='jpg',*args,**kwarg):
    
    fig1, axs = plt.subplots(2,1,figsize=(20,20))
    fs = 16

    data_plot1[:,RFI_mask_idx] = np.nan
    data_plot2[:,RFI_mask_idx] = np.nan

    for i in range(0,len(times['scanid'])):
        w = np.where((data['tplt']>=times['azstart'][i]) & (data['tplt']<=times['azstop'][i]))[0]
        extent = [times['azstart'][i],times['azstop'][i],data['freq'][0],data['freq'][-1]]   
        im0 = axs[0].imshow(data_plot1.T[:,w],aspect='auto',vmin=vmin[0],vmax=vmax[0],origin='lower',
                            extent=extent,cmap=cmap[0])
        im1 = axs[1].imshow(data_plot2.T[:,w],aspect='auto',vmin=vmin[1],vmax=vmax[1],origin='lower',
                            extent=extent,cmap=cmap[1])

    for i in range(0,Ne):
        w = np.where((data['tplt']>=times['rast1start'][i]) & (data['tplt']<=times['rast1stop'][i]))[0]
        extent = [times['rast1start'][i],times['rast1stop'][i],data['freq'][0],data['freq'][-1]] 
        im0 = axs[0].imshow(data_plot1.T[:,w],aspect='auto',vmin=vmin[0],vmax=vmax[0],origin='lower',
                            extent=extent,cmap=cmap[0])
        im1 = axs[1].imshow(data_plot2.T[:,w],aspect='auto',vmin=vmin[1],vmax=vmax[1],origin='lower',
                            extent=extent,cmap=cmap[1])

    for i in range(0,Nm):
        w = np.where((data['tplt']>=times['rast2start'][i]) & (data['tplt']<=times['rast2stop'][i]))[0]
        extent = [times['rast2start'][i],times['rast2stop'][i],data['freq'][0],data['freq'][-1]]   
        im0 = axs[0].imshow(data_plot1.T[:,w],aspect='auto',vmin=vmin[0],vmax=vmax[0],origin='lower',
                             extent=extent,cmap=cmap[0])
        im1 = axs[1].imshow(data_plot2.T[:,w],aspect='auto',vmin=vmin[1],vmax=vmax[1],origin='lower',
                             extent=extent,cmap=cmap[1])

    cbar0= fig1.colorbar(im0,ax=axs[0])
    cbar0.ax.tick_params(labelsize=fs) 
    cbar0.set_label(cbartitle[0], fontsize=fs)
    
    cbar1= fig1.colorbar(im1,ax=axs[1])
    cbar1.ax.tick_params(labelsize=fs) 
    cbar1.set_label(cbartitle[1], fontsize=fs)

    for i in range(0,2):
        axs[i].set_xlim(data['tplt'][0],data['tplt'][-1])
        axs[i].tick_params(axis='both', labelsize=fs,labelbottom=True)
        axs[i].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        axs[i].fmt_xdata = mdates.DateFormatter('%H:%M:%S')
        axs[i].set_ylim(data['freq'][0],data['freq'][-1])
        axs[i].set_ylabel('Frequency (MHz)',fontsize=fs)
    axs[1].set_xlabel('Time (UTC)',fontsize=fs)
    
    plt.savefig('/home/aordog/DVA_PLOTS/noise_gain_corr_new/'+plotname+'_phase'+str(phase)+'_day'+f"{day:02}"+'.'+filetype)
    plt.close()
    
    return

if __name__ =='__main__': 
    noise_source_correct()
