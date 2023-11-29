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
from astropy import units as u
from astropy.time import TimeDelta
import matplotlib.patches as patches
from importlib import reload 
import DVA_RFI as rfi


def run_batch():
    
    dir_lists = '/srv/data/dva/survey_azimuth_scans_noise_corr/'
    
    phase1_scans = np.genfromtxt(dir_lists+'phase1_scans.txt')
    phase2_scans = np.genfromtxt(dir_lists+'phase2_scans.txt')
    phase3_scans = np.genfromtxt(dir_lists+'phase3_scans.txt')
    
    #all_scans = ['dva_survey_phase3_raw_2879.h5']
    
    print(' Phase 1 scans ...')
    for i in range(0,len(phase1_scans)):
    #for i in range(0,2):
        scan = 'dva_survey_phase1_raw_'+'{0:04d}'.format(int(phase1_scans[i]))+'.h5'
        print('Scan '+str(i+1)+' of '+str(len(phase1_scans))+': '+scan)
        do_RFI_excision(scan)
    
    print(' Phase 2 scans ...')
    for i in range(0,len(phase2_scans)):
    #for i in range(0,2):
        scan = 'dva_survey_phase2_raw_'+'{0:04d}'.format(int(phase2_scans[i]))+'.h5'
        print('Scan '+str(i+1)+' of '+str(len(phase2_scans))+': '+scan)
        do_RFI_excision(scan)
    
    print(' Phase 3 scans ...')
    for i in range(0,len(phase3_scans)):
    #for i in range(0,2):
        scan = 'dva_survey_phase3_raw_'+'{0:04d}'.format(int(phase3_scans[i]))+'.h5'
        print('Scan '+str(i+1)+' of '+str(len(phase3_scans))+': '+scan)
        do_RFI_excision(scan)
    
    return


def do_RFI_excision(scan_pick):
    
    freq, t_set, t_plt, noise_idx, LL_set, PI_set = read_in_files(scan_pick)
    PI_set[noise_idx, :] = np.nan
    LL_set[noise_idx, :] = np.nan
    #RR_set[noise_idx, :] = np.nan

    filename2 = '/srv/data/dva/RFIpersist_mask/PersistRFImaskNewJustIndexBad_v2.txt'
    RFI_mask_idx = persistent_mask(filename2)

    total_OG_mask,total_baseline_mask,total_freq_mask = DVA_calc_RFI(PI_set, LL_set, freq, t_plt, RFI_mask_idx)

    intermittent_mask = np.logical_or(total_baseline_mask, total_freq_mask)
    intermittent_mask[intermittent_mask==False] = 0
    intermittent_mask[intermittent_mask==True] = 1

    complete_mask = intermittent_mask.copy()
    #complete_mask = total_OG_mask.copy()
    #complete_mask = total_freq_mask.copy()
    complete_mask[:,RFI_mask_idx] = 1

    out_name = scan_pick[0:18]+scan_pick[22:26]+'_RFI_mask.npy'
    
    np.save('/srv/data/dva/RFI_mask_full/'+out_name,complete_mask)

    return


def read_in_files(scan_choose):

    directory = '/srv/data/dva/survey_azimuth_scans/'
    file = h5py.File(directory+scan_choose,'r')

    # access the correct location in the file structure:
    dataset = file['data']['beam_0']['band_SB0']['scan_0']

    # get the list of frequencies:
    freq = file['data']['beam_0']['band_SB0']['frequency'][:]/1e6
    df = freq[1] - freq[0]

    # Add the position and time data to the corresponding arrays:
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


def DVA_calc_RFI(PI_set, LL_set, freq, t_plt, RFI_mask_idx):
    
    reload(rfi)
    
    freq_threshold = 1e6
    base_mult = 3

    total_possible = 0
    total_confirmed = 0
    total_baseline_mask = np.empty_like(PI_set)
    total_OG_mask = np.empty_like(PI_set)
    total_freq_mask = np.empty_like(PI_set)

    for freq_idx in range(0, len(freq)): 
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

    for time_idx in range(0, len(LL_set[:, 0])):
        data_plot_L = 10*np.log10(LL_set[time_idx,:])
        data_plot_L[RFI_mask_idx] = np.nan
        window_size = 50 #20
        LL_smoothed = moving_average(data_plot_L, window_size)
        LL_diff = np.abs(LL_smoothed - data_plot_L)
        freq_mask = np.zeros(len(LL_diff))
        #print(np.nanmedian(LL_diff))
        LL_diff[np.isnan(LL_diff)] = 0
        freq_mask_idxes = np.where(LL_diff > 0.5) #TODO: Make this 0.5 an explicit arbitrary variable
        freq_mask[freq_mask_idxes] = 1

        total_freq_mask[time_idx,:] = freq_mask

    return total_OG_mask,total_baseline_mask,total_freq_mask


if __name__ =='__main__': 
    run_batch()