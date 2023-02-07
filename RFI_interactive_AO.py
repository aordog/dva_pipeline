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
from ipywidgets import interact
import matplotlib.patches as patches
from importlib import reload 
import DVA_RFI as rfi
from ipywidgets import interact, interactive, fixed, interact_manual, Layout
import ipywidgets as widgets
from mpl_point_clicker import clicker
from tqdm import tqdm



def compare_persistent_masks(phase,day):
    
    scan_pick, directory = pick_scan(phase,day)

    freq, t_set, t_plt, noise_idx, LL_set, PI_set = read_in_files(scan_pick, directory, phase)
    PI_set[noise_idx, :] = np.nan
    LL_set[noise_idx, :] = np.nan
    #RR_set[noise_idx, :] = np.nan

    #filename1 = '/home/ordoga/Python/DVA2/DATA/RFIpersist_mask.txt' 
    #filename2 = '/home/ordoga/Python/DVA2/DATA/PersistRFImaskNewJustIndexBad.txt'
    filename1 = '/home/ordoga/Python/DVA2/DATA/PersistRFImaskNewJustIndexBad.txt'
    filename2 = '/home/ordoga/Python/DVA2/DATA/PersistRFImaskNewJustIndexBad_v2.txt'
    RFI_mask_idx_old = persistent_mask(filename1)
    RFI_mask_idx_new = persistent_mask(filename2)
    
    data_plot_persist_old = LL_set.copy()
    data_plot_persist_old[:,RFI_mask_idx_old] = np.nan
    
    data_plot_persist_new = LL_set.copy()
    data_plot_persist_new[:,RFI_mask_idx_new] = np.nan

    make_the_plot(10*np.log10(LL_set.T),
                  10*np.log10(data_plot_persist_old.T),
                  10*np.log10(data_plot_persist_new.T),t_plt,t_set,freq)

    return






def do_RFI_excision(phase,day):
    
    scan_pick, directory = pick_scan(phase,day)

    freq, t_set, t_plt, noise_idx, LL_set, PI_set = read_in_files(scan_pick, directory, phase)
    PI_set[noise_idx, :] = np.nan
    LL_set[noise_idx, :] = np.nan
    #RR_set[noise_idx, :] = np.nan

    RFI_mask_idx = persistent_mask()

    total_OG_mask,total_baseline_mask,total_freq_mask = DVA_Plot_RFI(PI_set, LL_set, freq, t_plt, RFI_mask_idx)

    intermittent_mask = np.logical_or(total_baseline_mask, total_freq_mask)
    intermittent_mask[intermittent_mask==False] = 0
    intermittent_mask[intermittent_mask==True] = 1

    complete_mask = intermittent_mask.copy()
    #complete_mask = total_OG_mask.copy()
    #complete_mask = total_freq_mask.copy()
    complete_mask[:,RFI_mask_idx] = 1

    data_plot_good = LL_set.copy()
    data_plot_good[complete_mask==1] = np.nan
    
    data_plot_persist = LL_set.copy()
    data_plot_persist[:,RFI_mask_idx] = np.nan

    make_the_plot(10*np.log10(LL_set.T),
                  10*np.log10(data_plot_persist.T),
                  10*np.log10(data_plot_good.T),t_plt,t_set,freq)

    return


def pick_scan(phase,day):
    
    directory = '/media/ordoga/15m_band1_survey/dva_phase'+str(phase)+'/survey_phase'+str(phase)+'_day'+f"{day:02}"+'/'
    
    print('')
    scan_id = []
    with open(directory+'DVAsurvey_phase'+str(phase)+'_day'+f"{day:03}"+'.txt') as fp:
        for line in fp:     
            print('  '+f"{int(line.split()[0]):4}")
            scan_id.append(int(line.split()[0]))
    
    goodpick = False
    while goodpick == False:
        scan_pick = input("Pick a scan: ")
        if int(scan_pick) in np.array(scan_id):
            print('')
            print('Running RFI excision for Phase '+str(phase)+', day '+str(day)+', scan '+f"{int(scan_pick):04}")
            print('')
            goodpick = True
        else:
            print('')
            print('Selected scan not in list. Select again.')
            print('')

    return scan_pick, directory


def read_in_files(scan_choose, directory, phase):

    file = h5py.File(directory+'dva_survey_phase'+str(phase)+'_raw_'+f"{int(scan_choose):04}"+'.h5','r')
    print(file)

    # access the correct location in the file structure:
    dataset = file['data']['beam_0']['band_SB0']['scan_0']

    # get the list of frequencies:
    freq = file['data']['beam_0']['band_SB0']['frequency'][:]/1e6
    df = freq[1] - freq[0]

    # Add the position and time data to the corresponding arrays:
    #dec_set = dataset['metadata']['declination']
    #ra_set = dataset['metadata']['right_ascension']
    #el_set = dataset['metadata']['elevation']
    #az_set = dataset['metadata']['azimuth']
    t_set = dataset['metadata']['utc']
    noise_set = dataset['metadata']['noise_state']
    trim_flag = dataset['metadata']['trim_scan_flag']

    # Add the spectrometer power data to the corresponding arrays:
    RR_set = dataset['data'][:,0,:]
    LL_set = dataset['data'][:,1,:]
    reRL_set = dataset['data'][:,2,:]
    imRL_set = dataset['data'][:,3,:]

    PI_set = np.sqrt((reRL_set**2)+(imRL_set**2))

    noise_idx = np.array(np.where(noise_set == 1))
    t_plt = Time(t_set, format='isot',scale='utc').mjd

    return freq, t_set, t_plt, noise_idx, LL_set, PI_set

def persistent_mask(filename):

    i = 0
    RFI_mask_idx = []
    with open(filename) as fp:    
        for line in fp:
            if i>0: 
                RFI_mask_idx.append(int(line.split()[0]))
            i=i+1
    RFI_mask_idx = np.array(RFI_mask_idx)
    
    return RFI_mask_idx

def moving_average(arr, window):
    return np.convolve(arr, np.ones(window), 'same') / window

def DVA_Plot_RFI(PI_set, LL_set, freq, t_plt, RFI_mask_idx):
    
    reload(rfi)
    
    freq_threshold = 1e6
    base_mult = 3

    total_possible = 0
    total_confirmed = 0
    total_baseline_mask = np.empty_like(PI_set)
    total_OG_mask = np.empty_like(PI_set)
    total_freq_mask = np.empty_like(PI_set)

    for freq_idx in tqdm(range(0, len(freq))): 
        if freq_idx in RFI_mask_idx:
            pass
        else:
            #OG_mask = np.zeros(len(LL_set[:, freq_idx]))
            #[confirmed_RFI_results, number_of_possible, 
            #                        number_of_confirmed] = rfi.RFI_Detection(freq_slope_threshold=freq_threshold,
            #                                                                 freq_idx = freq_idx, 
            #                                                                 baseline_multiplier=base_mult, 
            #                                                                 polarized_set = PI_set, 
            #                                                                 df = (freq[1] - freq[0]), 
            #                                                                 apply_freq_verification = True)
            #confirmed_rfi_idxes = rfi.GenerateRfiIndexes(confirmed_RFI_results, t_plt)[0]
            #OG_mask[confirmed_rfi_idxes] = 1
            #total_OG_mask[:,freq_idx] = OG_mask


            baseline_mask = np.zeros(len(LL_set[:, freq_idx]))
            [baseline_RFI_results, number_of_possible, 
                                    number_of_confirmed] = rfi.RFI_Detection(freq_slope_threshold=freq_threshold,
                                                                             freq_idx = freq_idx, 
                                                                             baseline_multiplier=base_mult, 
                                                                             polarized_set = PI_set, 
                                                                             df = (freq[1] - freq[0]), 
                                                                             apply_freq_verification = True)
            baseline_RFI_idx = rfi.GenerateRfiIndexes(baseline_RFI_results, t_plt)[0]
            baseline_mask[baseline_RFI_idx] = 1
            total_baseline_mask[:,freq_idx] = baseline_mask
            total_possible += number_of_possible
            total_confirmed += number_of_confirmed

    for time_idx in tqdm(range(0, len(LL_set[:, 0]))):
        data_plot_L = 10*np.log10(LL_set[time_idx,:])
        data_plot_L[RFI_mask_idx] = np.nan
        window_size = 50 #20
        LL_smoothed = moving_average(data_plot_L, window_size)
        LL_diff = np.abs(LL_smoothed - data_plot_L)
        freq_mask = np.zeros(len(LL_diff))
        freq_mask_idxes = np.where(LL_diff > 0.5) #TODO: Make this 0.5 an explicit arbitrary variable
        freq_mask[freq_mask_idxes] = 1
        total_freq_mask[time_idx,:] = freq_mask

    return total_OG_mask,total_baseline_mask,total_freq_mask



def mouse_event(event,ax1,ax2,ax3,t_set,t_plt,freq,data_plot_bad,data_plot_persist,data_plot_good):
    
    fs = 10
    
    print('x: {} and y: {}'.format(event.xdata, event.ydata))   
    df = freq[1] - freq[0]
    dt = t_plt[1] - t_plt[0]
    fidx = np.where(abs(freq-event.ydata)<df/2)[0][0]
    tidx = np.where(abs(t_plt-event.xdata)<dt/2)[0][0]
    ax2.cla()
    ax3.cla()

    ax2.set_xlim(freq[0],freq[-1])
    ax2.set_ylim(66,80)
    #ax2.set_xlabel('Frequency (MHz)',fontsize=fs)
    ax2.set_ylabel('Power (dB)',fontsize=fs)
    ax2.set_title('Time = '+str(t_set[tidx])[13:23])

    ax3.set_xlim(t_plt[0],t_plt[-1])
    ax3.set_ylim(66,80)
    ax3.tick_params(axis='both', labelsize=fs)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax3.fmt_xdata = mdates.DateFormatter('%H:%M:%S')
    ax3.set_xlabel('Time (UTC)',fontsize=fs)        
    ax3.set_ylabel('Power (dB)',fontsize=fs)
    ax3.set_title('Frequency = '+str(round(event.ydata,3)))

    
    ax2.plot(freq,data_plot_bad[:,tidx],color='C0',linewidth=1,zorder=1)
    ax2.scatter(freq,data_plot_bad[:,tidx],s=5,color='C0',zorder=1)
    
    ax3.plot(t_plt,data_plot_bad[fidx,:],color='C0',linewidth=1,zorder=1)
    ax3.scatter(t_plt,data_plot_bad[fidx,:],s=5,color='C0',zorder=1)
    
    
    ax2.plot(freq,data_plot_persist[:,tidx],color='C1',linewidth=2,zorder=5)
    ax2.scatter(freq,data_plot_persist[:,tidx],s=10,color='C1',zorder=5)
    
    ax3.plot(t_plt,data_plot_persist[fidx,:],color='C1',linewidth=2,zorder=5)
    ax3.scatter(t_plt,data_plot_persist[fidx,:],s=10,color='C1',zorder=5)
    
    
    ax2.plot(freq,data_plot_good[:,tidx],color='k',linewidth=1,zorder=10)
    ax2.scatter(freq,data_plot_good[:,tidx],s=5,color='k',zorder=10)
    
    ax3.plot(t_plt,data_plot_good[fidx,:],color='k',linewidth=1,zorder=10)
    ax3.scatter(t_plt,data_plot_good[fidx,:],s=5,color='k',zorder=10)

    plt.show()

    return 

def make_the_plot(data_plot_bad,data_plot_persist,data_plot_good,t_plt,t_set,freq):

    fs = 10
    
    fig2 = plt.figure(1,figsize=(6,6))
    ax2 = plt.subplot(211)
    ax3 = plt.subplot(212)
    
    fig1 = plt.figure(2,figsize=(6,6))
    ax1 = plt.subplot(111)
    
    fig3 = plt.figure(3,figsize=(6,6))
    ax4 = plt.subplot(111)
    
    #vmin = 1e4
    #vmax = 2e6
    vmin = 66
    vmax = 76
        
    im = ax1.imshow(data_plot_good,aspect='auto',vmin=vmin,vmax=vmax,origin='lower',
                    extent=[t_plt[0],t_plt[-1],freq[0],freq[-1]],cmap='viridis')
    
    ax1.set_xlim(t_plt[0],t_plt[-1])
    ax1.set_ylim(freq[0], freq[-1])
    ax1.tick_params(axis='both', labelsize=fs)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax1.fmt_xdata = mdates.DateFormatter('%H:%M:%S')
    ax1.set_xlabel('Time (UTC)',fontsize=fs)        
    ax1.set_ylabel('Frequency (MHz)',fontsize=fs)
    
    
    im2 = ax4.imshow(data_plot_bad,aspect='auto',vmin=vmin,vmax=vmax,origin='lower',
                    extent=[t_plt[0],t_plt[-1],freq[0],freq[-1]],cmap='viridis')
    
    ax4.set_xlim(t_plt[0],t_plt[-1])
    ax4.set_ylim(freq[0], freq[-1])
    ax4.tick_params(axis='both', labelsize=fs)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax4.fmt_xdata = mdates.DateFormatter('%H:%M:%S')
    ax4.set_xlabel('Time (UTC)',fontsize=fs)        
    ax4.set_ylabel('Frequency (MHz)',fontsize=fs)
    
    cid = fig1.canvas.mpl_connect('button_press_event', 
                                  lambda event: mouse_event(event,ax1,ax2,ax3,t_set,t_plt,
                                                            freq,data_plot_bad,data_plot_persist,data_plot_good))

    klicker = clicker(ax1, ["s"], markers=["x"])
    ax1.get_legend().remove()
    plt.show()
            
    return



if __name__ =='__main__': 
    do_RFI_excision()