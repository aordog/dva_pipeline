import numpy as np

#**************************************************************************************
#                                  Auxiliary Functions
#**************************************************************************************
def find_nearest_idx(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def Apply_Mask(mask, input_array): #TODO: This needs to be updated to implement all frequencies
    
    masked_copy = input_array.copy()
    mask_idx = np.where(mask == 1)
    masked_copy[mask_idx, :] = np.nan
    return(masked_copy)



#**************************************************************************************
#                                  Time-Domain Functions
#**************************************************************************************
def Combine_Into_RFI_Events(possible_RFI):
    Combined_RFI_events = [] 
    remaining_idxes = len(possible_RFI)
    if remaining_idxes > 0:
        current_idx = 0
        rfi_start = possible_RFI[current_idx]
        rfi_duration = 1

        while remaining_idxes >= 2:
            remaining_idxes = (len(possible_RFI) - current_idx)
            previous_idx = current_idx - 1
            idx_gap = (possible_RFI[current_idx] - possible_RFI[previous_idx])
            if(idx_gap <= 2):
                rfi_duration += 1
                current_idx += 1
            else:
                #Close current event
                Combined_RFI_events.append([rfi_start, rfi_duration])
                #Initiate next event starting on the current_idx
                rfi_start = possible_RFI[current_idx]
                rfi_duration = 1
                current_idx += 1

    return Combined_RFI_events #[idx, idx_duration]

#**************************************************************************************
#                                  Freq-Domain Functions
#**************************************************************************************
def ValidateStart(idx, delta):
    if (idx-delta) <= 0:
        validaded_start = 0
    else:
        validaded_start = idx-delta
    return validaded_start

def ValidateEnd(idx, delta, array):
    if (idx+delta) >= (len(array)-1):
        validaded_end = (len(array)-1)
    else:
        validaded_end = idx+delta
    return validaded_end


def Spectrum_Start_Found(spectrum_power_array, idx, slope_threshold):
    RFI_Start_Found = False
    spectrum_gradient = np.gradient(spectrum_power_array, 3)
    current_slope = spectrum_gradient[idx-1]
    next_slope = spectrum_gradient[idx]
    if((current_slope <= slope_threshold) and (next_slope >= slope_threshold)):
        RFI_Start_Found = True
        # print("RFI start information:\n - current slope:", current_slope,"next slope:", next_slope, "Slope threshold:", slope_threshold)
    return RFI_Start_Found


def Spectrum_End_Found(spectrum_power_array, rfi_end_idx, RFI_start_value):
    retVal = False
    if spectrum_power_array[rfi_end_idx] <= RFI_start_value:
        retVal = True
    return retVal

def Spectrum_Scan(spectrum_power_array, freq_idx, slope_threshold, df): #NOTE: This might be an issue if I have a nan value on my array
    scan_bandwidth = 10 #[MHz]
    #TODO: I have to make sure the start_idx and end_idx don't go over at the edges
    start_idx = ValidateStart(freq_idx, int((scan_bandwidth/df)/2))  #This changes the bandwidth from MHz to idxes
    end_idx = ValidateEnd(freq_idx, int((scan_bandwidth/df)/2), spectrum_power_array)
    RFI_confirmed = False
    RFI_spectral_thickness = 0
    for rfi_start_idx in range(start_idx, end_idx):                                             #Scan over the entire bandwith interval
        if(Spectrum_Start_Found(spectrum_power_array, rfi_start_idx, slope_threshold)):     #If I found the starting pattern
            RFI_start_value = spectrum_power_array[rfi_start_idx]
            rfi_minimum_length = 10     #This exists to avoid the algorithm from fake crossing near the start due to fluctuations.
            for rfi_end_idx in range(rfi_start_idx + rfi_minimum_length, end_idx):                                   #Finish looking at the bandwidth interval looking for the end
                # if(Spectrum_RFI_End_Found(time_array, spectrum_power_array, rfi_end_idx, slope_threshold)):     #If I found the ending pattern          TODO: I have a problem in the end found algorithm
                if(Spectrum_End_Found(spectrum_power_array, rfi_end_idx, RFI_start_value)):     #If the RFI crosses below the starting value         TODO: I have a problem in the end found algorithm
                    RFI_confirmed = True      
                    break
                    # RFI_spectral_thickness = (rfi_end_idx - rfi_start_idx)
                else:
                    continue
            if(RFI_confirmed):                                                                          #and stop looping throught the interval for efficiency
                break
        else:
            rfi_end_idx = end_idx
    return RFI_confirmed, rfi_start_idx, (rfi_end_idx)#RFI_spectral_thickness

#****************************************************************************
#                                  RFI excision
#****************************************************************************
def DVA_Find_Possible_RFI_Events(freq_idx, baseline_multiplier, polarized_set):
    scan_baseline = np.nanmedian(polarized_set[:,freq_idx])
    scan_threshold = scan_baseline*baseline_multiplier

    possible_RFI_idxes = np.where(polarized_set[:, freq_idx] >= scan_threshold)
    possible_RFI_events = Combine_Into_RFI_Events(possible_RFI_idxes[0])
    return possible_RFI_events  #Returns [time_idx, idx_duration]

def RFI_Verification(possible_RFI_events, freq_slope_threshold, event, freq_idx, polarized_set, df):
    rfi_confirmed = False
    for time_idx in range(possible_RFI_events[event][0], possible_RFI_events[event][0]+ possible_RFI_events[event][1]):
        event_verification_result = Spectrum_Scan(polarized_set[time_idx, :], freq_idx, freq_slope_threshold, df)
        if event_verification_result[0]:
            rfi_confirmed = True
            break
    return rfi_confirmed, event_verification_result

def DVA_Find_Possible_Event_Start(freq_idx, polarized_set, possible_RFI_events):
    scan_baseline = np.nanmedian(polarized_set[:,freq_idx])
    start_event_idxes = []
    for event in range(0, len(possible_RFI_events)-1):
        start_found = False
        event_start_idx = possible_RFI_events[event][0]
        while not start_found:
            event_start_idx  = event_start_idx - 1
            if (polarized_set[event_start_idx, freq_idx] <= scan_baseline):
                start_event_idxes.append(event_start_idx)
                start_found = True
    return start_event_idxes

def DVA_Find_Possible_Event_End(freq_idx, polarized_set, possible_RFI_events):
    scan_baseline = np.nanmedian(polarized_set[:,freq_idx])
    end_event_idxes = []
    for event in range(0, len(possible_RFI_events)-1):
        end_found = False
        event_end_idx = possible_RFI_events[event][0] + possible_RFI_events[event][1]
        while not end_found:
            if event_end_idx <= (len(polarized_set[:, freq_idx])-2):
                event_end_idx  = event_end_idx + 1
                if (polarized_set[event_end_idx, freq_idx] <= scan_baseline):
                    end_event_idxes.append(event_end_idx)
                    end_found = True
            else:
                end_event_idxes.append(event_end_idx)
                end_found = True
                break
                
    return end_event_idxes


#**************************************************************************************
#                                  Main Function
#**************************************************************************************
def RFI_Detection(freq_slope_threshold, freq_idx, baseline_multiplier, polarized_set, df, apply_freq_verification):
    confirmed_RFI_results = []
    possible_RFI_events = DVA_Find_Possible_RFI_Events(freq_idx, baseline_multiplier, polarized_set)
    possible_RFI_starts = DVA_Find_Possible_Event_Start(freq_idx, polarized_set, possible_RFI_events)
    possible_RFI_ends = DVA_Find_Possible_Event_End(freq_idx, polarized_set, possible_RFI_events)
    if len(possible_RFI_events) != 0:
        for event in range(0, len(possible_RFI_events)-1):
            if apply_freq_verification:
                rfi_confirmed, event_verification_result = RFI_Verification(possible_RFI_events, freq_slope_threshold, event, freq_idx, polarized_set, df)      
                if rfi_confirmed:  
                    t1_plt = possible_RFI_starts[event]                                             #Start time     [idx]
                    t2_plt = possible_RFI_ends[event]                                               #End time       [idx]
                    confirmed_RFI_results.append([t1_plt, t2_plt])
            else:
                t1_plt = possible_RFI_starts[event]                                             #Start time     [idx]
                t2_plt = possible_RFI_ends[event]                                               #End time       [idx]
                confirmed_RFI_results.append([t1_plt, t2_plt])

    return confirmed_RFI_results, len(possible_RFI_events), len(confirmed_RFI_results)
    
def GenerateRfiIndexes(confirmed_RFI_results, t_plt):
    RFI_time_mask = np.zeros(len(t_plt))
    for rfi_number in range(0, len(confirmed_RFI_results)-1):
        # DETERMINE RFI REGION --------------------------------------------------------------------------------------------------------
        t1_plt = confirmed_RFI_results[rfi_number][0]
        t2_plt = confirmed_RFI_results[rfi_number][1]
        for rfi_idx in range(t1_plt, t2_plt):
            RFI_time_mask[rfi_idx] = 1

    rfi_idxes = np.array(np.where(RFI_time_mask == 1))
    return rfi_idxes
#TODO: I hate the naming of this return
