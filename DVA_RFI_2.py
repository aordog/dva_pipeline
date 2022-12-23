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

def unique(list1):
 
    # initialize a null list
    unique_list = []
 
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    return unique_list

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
#                                  Validation Functions
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


#****************************************************************************
#                                  RFI excision
#****************************************************************************
def DVA_Find_Possible_RFI_Events(freq_idx, baseline_multiplier, polarized_set):
    scan_baseline = np.nanmedian(polarized_set[:,freq_idx])
    scan_threshold = scan_baseline*baseline_multiplier

    possible_RFI_idxes = np.where(polarized_set[:, freq_idx] >= scan_threshold)
    possible_RFI_events = Combine_Into_RFI_Events(possible_RFI_idxes[0])
    return possible_RFI_events  #Returns [time_idx, idx_duration]


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
def RFI_Detection(freq_slope_threshold, freq_idx, baseline_multiplier, polarized_set, df):
    confirmed_RFI_results = []
    possible_RFI_events = DVA_Find_Possible_RFI_Events(freq_idx, baseline_multiplier, polarized_set)
    possible_RFI_starts = DVA_Find_Possible_Event_Start(freq_idx, polarized_set, possible_RFI_events)
    possible_RFI_ends = DVA_Find_Possible_Event_End(freq_idx, polarized_set, possible_RFI_events)
    if len(possible_RFI_events) != 0:
        # print("Number of possible RFI event:", len(possible_RFI_events))
        for event in range(0, len(possible_RFI_events)-1):

            t1_plt = possible_RFI_starts[event]                                             #Start time     [idx]
            t2_plt = possible_RFI_ends[event]                                               #End time       [idx]


            confirmed_RFI_results.append([t1_plt, t2_plt])

        # print("Number of confirmed RFI regions:", len(confirmed_RFI_results))

    return confirmed_RFI_results, len(confirmed_RFI_results)

# confirmed_RFI_results = RFI_Detection(freq_slope_threshold = 1e5, freq_chosen = 844, baseline_multiplier = 3)

# interact(RFI_Detection(scan = 1045))

#TODO:Create GenerateRfiIndexes() Where I'm merging RFI's and creating an RFI mask
# def GenerateRfiIndexes(confirmed_RFI_results, freq):
#     RFI_freq_mask = np.zeros(len(freq))
#     for rfi_number in range(0, len(confirmed_RFI_results)-1):
#         # DETERMINE RFI REGION --------------------------------------------------------------------------------------------------------
#         t1_plt = confirmed_RFI_results[rfi_number][0]
#         t2_plt = confirmed_RFI_results[rfi_number][1]
#         for rfi_idx in range(t1_plt, t2_plt):
#             RFI_freq_mask[rfi_idx] = 1

#     rfi_idxes = np.array(np.where(RFI_freq_mask == 1))
#     return rfi_idxes
# #TODO: I hate the naming of this return
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
