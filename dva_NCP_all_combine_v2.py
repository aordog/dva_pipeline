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
import gc
#from matplotlib.dates import HourLocator as HourLocator
#from matplotlib.dates import MinuteLocator as MinuteLocator

def combine(dir_files,outfiles,outname,filetype=3,noise_file=None,freq_avg=False,
            df=1.0e6, fmin=351.0e6, fmax = 1030.0e6,bintype='mean',*args,**kwargs):
  
    # Filetype = 1: no metadata, no noise state in .h5
    # Filetpye = 2: positions in metadata, no noise state in .h5
    # Filetype = 3: positions and noise state in metadata
        
    proc=subprocess.Popen('ls -1 '+dir_files+'*.h5', shell=True, stdout=subprocess.PIPE)
    all_files = proc.communicate()[0].decode().split('\n')[0:-1][:]

    print('Number of files: ',len(all_files))
    print('')
    
    freq,nf = get_frequencies(all_files,dir_files,filetype)
    print('Frequencies:')
    print(freq/1e6)
    print(freq[1]/1e3-freq[0]/1e3)
    print(len(freq))
    print(' ')
    
    if (filetype == 1):
        t_set,az_set,el_set,nt = get_times_and_coords_old(all_files)
    else:
        t_set,az_set,el_set,nt,noise_set = get_times_and_coords_new(all_files)

    if (filetype == 1) or (filetype == 2):
        print('reading in noise source file...')
        t1_mjd, t2_mjd = read_in_noise(noise_file)
        print('making noise flag array...')
        noise_set = make_noise_mask(t_set,t1_mjd,t2_mjd)
        
    RR_set,LL_set,reRL_set,imRL_set = get_data_products(all_files,nt,nf,len(freq),filetype)
    
    RFI_mask_idx_new = get_RFI_mask('/home/ordoga/Python/DVA2/DATA/PersistRFI_v01_Jul13.txt',freq)
    #print('--------------')
    #print(RFI_idx)
    #print('--------------')
    RR_set[:,RFI_mask_idx_new] = np.nan
    LL_set[:,RFI_mask_idx_new] = np.nan
    reRL_set[:,RFI_mask_idx_new] = np.nan
    imRL_set[:,RFI_mask_idx_new] = np.nan
    
    if freq_avg == True:
        frq_arr,RR_avg,LL_avg,reRL_avg,imRL_avg = bin_data_and_freq(RR_set,LL_set,reRL_set,imRL_set,freq,df,fmin,fmax,bintype)
    
        make_new_file(outname,outfiles,all_files[0],RR_avg, LL_avg, reRL_avg, imRL_avg,
                      t_set,az_set,el_set,frq_arr,noise_set)
    else:
        make_new_file(outname,outfiles,all_files[0],RR_set, LL_set, reRL_set, imRL_set,
                      t_set,az_set,el_set,frq_arr,noise_set)

    return None


def get_RFI_mask(RFI_mask_file,freq):
    
    i = 0
    RFI_mask_frq = []
    RFI_mask_idx = []
    with open(RFI_mask_file) as fp:
        for line in fp:
            if i>0: 
                #print(line)
                RFI_mask_frq.append(float(line.split()[1]))
                RFI_mask_idx.append(int(line.split()[0]))
            i=i+1
            
    df = (freq[1]-freq[0])/1e6
    print(df)
    if RFI_mask_idx[1]-RFI_mask_idx[0] == 1:
        df_mask = RFI_mask_frq[1]-RFI_mask_frq[0]
        print(df_mask)
    else:
        print('Cannot find mask freq spacing!')
        
    RFI_mask_idx_new = []    
            
    for i in range(0,len(RFI_mask_idx)):
        wbad = np.where(abs(freq/1e6 - RFI_mask_frq[i]) <= df_mask/2)[0]
        if len(wbad) > 0:
            RFI_mask_idx_new.extend(list(wbad))
   
    return RFI_mask_idx_new



def read_in_noise(noise_file):
    
    t_noise1 = []
    t_noise2 = []

    with open(noise_file) as fp:
        for line in fp:       
            t_noise1.append(line.split()[0]+'T'+line.split()[1][0:12])
            t_noise2.append(line.split()[2]+'T'+line.split()[3][0:12])
        
    t_noise1_fix = Time(t_noise1, format='isot',scale='utc')
    t_noise2_fix = Time(t_noise2, format='isot',scale='utc')

    t_noise1_mjd = t_noise1_fix.mjd
    t_noise2_mjd = t_noise2_fix.mjd    
    
    return t_noise1_mjd, t_noise2_mjd


def make_noise_mask(t_set,t1_mjd,t2_mjd):
    
    t_set_fix = Time(t_set, format='isot',scale='utc')
    t_set_mjd = t_set_fix.mjd
    
    noise = np.zeros_like(t_set_mjd)
    
    noise_state = 0    
    j = 0
    t_arrs = [t1_mjd,t2_mjd]
    #t_bound = t1_mjd[0]
    
    for i in range(0,len(t_set_mjd)):
        try:
            t_bound = t_arrs[noise_state][j]  
            if t_set_mjd[i] >= t_bound:            
                noise_state = abs(noise_state-1)
                if noise_state == 0:
                    j = j+1
            noise[i] = noise_state
        except:
            pass
    
    return noise



def get_frequencies(all_files,dir_files,filetype):
    
    freq = []
    file = h5py.File(all_files[0],'r')
    beam = file['data']['beam_0']
    
    if filetype == 1:
        band_id_min = 2
    else:
        band_id_min = 0
    
    print(beam['band_SB3']['frequency'][1]-beam['band_SB3']['frequency'][0])
    print(beam['band_SB3']['scan_0']['data'].shape[2])
    nf = beam['band_SB3']['scan_0']['data'].shape[2]

    for i, band_id in enumerate(beam.keys()):
        if i >= band_id_min:
            print(i,band_id)
            band = beam[band_id]
            freq = np.concatenate([freq,band.get('frequency')[:]])
            #if avg_bands == False:
            #    freq = np.concatenate([freq,band.get('frequency')[::step]])
            #else:
            #    freq = np.concatenate([freq,np.nanmean(band.get('frequency')[:].reshape(-1,step), axis=1)])
    
    file.close()
    
    return freq,nf


def get_times_and_coords_old(all_files):
    
    file = h5py.File(all_files[0],'r')
    beam = file['data']['beam_0']
    nt = beam['band_SB3']['scan_0']['data'].shape[0]
    print('Timestamps per file: ',nt)

    t_set = []
    az_set = []
    el_set = []
    
    for ifile in range(0,len(all_files)):
        
        #print(ifile+1,all_files[ifile])
        file = h5py.File(all_files[ifile],'r')
        beam = file['data']['beam_0']
        pos = beam['band_SB3']['scan_0']['position'][:]
        t = beam['band_SB3']['scan_0']['time'][:]
        
        az = []
        el = []
        for j in range(0,len(pos)):
            el = np.concatenate([el,[pos[j][0]]])
            az = np.concatenate([az,[pos[j][1]]])

        el_set = np.concatenate([el_set,el])
        az_set = np.concatenate([az_set,az])       
        t_set = np.concatenate([t_set,t])
        
    print(len(t_set))
    
    file.close()
    
    return t_set,az_set,el_set,nt


def get_times_and_coords_new(all_files):
    
    file = h5py.File(all_files[0],'r')
    beam = file['data']['beam_0']
    nt = beam['band_SB3']['scan_0']['data'].shape[0]
    print('Timestamps per file: ',nt)

    t_set = []
    az_set = []
    el_set = []
    noise = []
    
    for ifile in range(0,len(all_files)):
        #print(ifile+1,all_files[ifile])
        file = h5py.File(all_files[ifile],'r')
        metadata = file['data']['beam_0']['band_SB3']['scan_0']['metadata']
        el_set = np.concatenate([el_set,metadata['elevation']])
        az_set = np.concatenate([az_set,metadata['azimuth']])
        t_set = np.concatenate([t_set,metadata['utc']])
        try:
            noise = np.concatenate([noise,metadata['noise_state']])
        except:
            pass
        
    print(len(t_set))
    
    file.close()
    
    return t_set,az_set,el_set,nt,noise


def get_data_products(all_files,nt,nf,nf_all,filetype):    

    file = h5py.File(all_files[0],'r')
    beam = file['data']['beam_0']
    
    RR_set = np.empty([nt*len(all_files),nf_all])
    LL_set = np.empty([nt*len(all_files),nf_all])
    reRL_set = np.empty([nt*len(all_files),nf_all])
    imRL_set = np.empty([nt*len(all_files),nf_all])
    
    if filetype == 1:
        band_id_min = 2
    else:
        band_id_min = 0
    
    for ifile in range(0,len(all_files)):

        file = h5py.File(all_files[ifile],'r') 
        print('File ',ifile+1,' out of ',len(all_files))
              
        for i, band_id in enumerate(beam.keys()):
            
            if filetype == 1:
                j = i-2
            else:
                j = i

            if i >= band_id_min:   
                data = file['data']['beam_0'][band_id]['scan_0']['data']
                RR_set[ifile*nt:(ifile+1)*nt,j*nf:(j+1)*nf] = data[:,1,:]
                LL_set[ifile*nt:(ifile+1)*nt,j*nf:(j+1)*nf] = data[:,0,:]
                reRL_set[ifile*nt:(ifile+1)*nt,j*nf:(j+1)*nf] = data[:,2,:]
                imRL_set[ifile*nt:(ifile+1)*nt,j*nf:(j+1)*nf] = data[:,3,:]
                
                del data
                gc.collect()
                
                   
        file.close()
       
    return RR_set,LL_set,reRL_set,imRL_set


def bin_data_and_freq(RR_set,LL_set,reRL_set,imRL_set,freq,df,fmin,fmax,bintype='mean',*args,**kwargs):
    
    frq_arr = np.arange(fmin,fmax,df)
    print(frq_arr)
    
    RR_avg = np.empty([RR_set.shape[0],len(frq_arr)])
    LL_avg = np.empty([LL_set.shape[0],len(frq_arr)])
    reRL_avg = np.empty([reRL_set.shape[0],len(frq_arr)])
    imRL_avg = np.empty([imRL_set.shape[0],len(frq_arr)])
    print(RR_avg.shape)
    
    print('')
    print('Binning in frequency...')
    for i in range(0,len(frq_arr)):
        print(frq_arr[i])
        w = np.where((freq >= frq_arr[i]-df/2.) & (freq <= frq_arr[i]+df/2.))[0]
        if bintype == 'mean':
            print('Using mean')
            RR_avg[:,i] = np.nanmean(RR_set[:,w],axis=1)
            LL_avg[:,i] = np.nanmean(LL_set[:,w],axis=1)
            reRL_avg[:,i] = np.nanmean(reRL_set[:,w],axis=1)
            imRL_avg[:,i] = np.nanmean(imRL_set[:,w],axis=1)
        if bintype == 'med':
            print('Using median')
            RR_avg[:,i] = np.nanmedian(RR_set[:,w],axis=1)
            LL_avg[:,i] = np.nanmedian(LL_set[:,w],axis=1)
            reRL_avg[:,i] = np.nanmedian(reRL_set[:,w],axis=1)
            imRL_avg[:,i] = np.nanmedian(imRL_set[:,w],axis=1)
    
    return frq_arr, RR_avg, LL_avg, reRL_avg, imRL_avg



def make_new_file(outname,outfiles,file_ex,RR,LL,reRL,imRL,
                  t_set,az_set,el_set,freq,noise):
    
    cmd2 = 'cp '+file_ex+' '+outfiles+outname+'.h5'
    os.system(cmd2)
    file = h5py.File(outfiles+outname+'.h5','r+')
    band_ids = file['data']['beam_0'].keys()

    #for i in range(3,8):
    for band_id in band_ids:
        del file['data']['beam_0'][str(band_id)]
    
    # Create band and scan groups:
    file['data']['beam_0'].create_group("band_SB0")
    file['data']['beam_0']['band_SB0'].create_group(f"scan_0")
    
    # Create power dataset:
    dat = np.empty((len(t_set), 4, len(freq)), dtype=float)
    file['data']['beam_0']['band_SB0']['scan_0'].create_dataset("data", data=dat)
    file['data']['beam_0']['band_SB0']['scan_0']['data'][:,0,:] = RR
    file['data']['beam_0']['band_SB0']['scan_0']['data'][:,1,:] = LL
    file['data']['beam_0']['band_SB0']['scan_0']['data'][:,2,:] = reRL
    file['data']['beam_0']['band_SB0']['scan_0']['data'][:,3,:] = imRL
    
    # Create metadata with timestamps and coordinates:
    metadata_content = [t_set,az_set,el_set,noise]  
    #print(noise)
    col_names = ["utc", "azimuth", "elevation", "noise_state"]
    col_types = np.dtype({'names':col_names,'formats':["S32", "f8", "f8", "i8"] } )
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
