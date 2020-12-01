'''
    Daniel Jacobs 2020
    OzGrav - University of Western Australia
'''

import math
import numpy as np
from pycbc.types.timeseries import TimeSeries
from scipy.signal.windows import tukey
import h5py

def find_nearest(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx


#Not my function - Ref: Gebhard, Kilbertus https://github.com/timothygebhard/ggwd
def fade_on(timeseries,
            alpha=0.25):
    """
    Take a PyCBC time series and use a one-sided Tukey window to "fade
    on" the waveform (to reduce discontinuities in the amplitude).

    Args:
        timeseries (pycbc.types.timeseries.TimeSeries): The PyCBC
            TimeSeries object to be faded on.
        alpha (float): The alpha parameter for the Tukey window.

    Returns:
        The `timeseries` which has been faded on.
    """

    # Save the parameters from the time series we are about to fade on
    delta_t = timeseries.delta_t
    epoch = timeseries.start_time
    duration = timeseries.duration
    sample_rate = timeseries.sample_rate

    # Create a one-sided Tukey window for the turn on
    window = tukey(M=int(duration * sample_rate), alpha=alpha)
    window[int(0.5*len(window)):] = 1

    # Apply the one-sided Tukey window for the fade-on
    ts = window * np.array(timeseries)

    # Create and return a TimeSeries object again from the resulting array
    # using the original parameters (delta_t and epoch) of the time series
    return TimeSeries(initial_array=ts,
                      delta_t=delta_t,
                      epoch=epoch)


#fixes signals all to the same length so they can be saved together in a HDF file
#extends shorter signals with 0s
def convert_cbc_array_to_np(signals, target_length):
    cbc_arrs=[]

    if type(signals)==dict:
        for det, cbc in list(signals.items()):
            cbc_arr = np.array(cbc)
            cbc_arrs.append(cbc_arr)
    elif type(signals)==list:
        for cbc in signals:
            cbc_arr = np.array(cbc)
            cbc_arrs.append(cbc_arr)

    if cbc_arrs[0].shape==cbc_arrs[1].shape==cbc_arrs[2].shape:
        return cbc_arrs

    corrected_cbc = []
    for cbc in cbc_arrs:
        zero_arr = np.zeros((target_length,))
        slice_idx = target_length if cbc.shape[0]>=target_length else cbc.shape[0]
        zero_arr[:cbc.shape[0]] = cbc[:slice_idx]
        corrected_cbc.append(zero_arr)

    return corrected_cbc


def get_param_arrays(sig_params):
    keys = {
        'm1': 'mass1',
        'm2': 'mass2',
        'x1': 'spin1z',
        'x2': 'spin2z',
        'coa': 'coa_phase',
        'ra': 'ra',
        'dec': 'dec',
        'inc': 'inclination',
        'pol': 'polarization',
        'snr': 'injection_snr',
        'h1_snr': 'h1_snr',
        'l1_snr': 'l1_snr',
        'v1_snr': 'v1_snr'
    }
    arrays = dict()
    for old, new in keys.items():
        l = [x[old] for x in sig_params]
        arr = np.array(l)
        arrays[new] = arr
    return arrays


#Take signal series and add them to the dataset hdf file
#If this is the first set, create groups and file structure
def to_hdf(file_path, sim_params, noisy_signals, pure_signals, sig_params, signal_len):
    #Initialise arrays to hold strain, sample_times, and parameters
    shape = (len(noisy_signals), int(signal_len))
    signal_len = int(signal_len)
    print(shape)
    h1_strain = np.zeros(shape)
    l1_strain = np.zeros(shape)
    v1_strain = np.zeros(shape)
    h1_time = np.zeros(shape)
    l1_time = np.zeros(shape)
    v1_time = np.zeros(shape)
    h1_pure_strain = np.zeros(shape)
    l1_pure_strain = np.zeros(shape)
    v1_pure_strain = np.zeros(shape)
    sig_param_arrs = get_param_arrays(sig_params)

    if noisy_signals[0]['H1'].shape[0]!=signal_len:
        print(('WARNING: Signal of len {0} is being truncated to {1}'.format(noisy_signals[0]['H1'].shape[0], signal_len)))

    for i in range(0, len(noisy_signals)):

        #Join signals as one Numpy array
        sig_set = convert_cbc_array_to_np(noisy_signals[i], signal_len)
        h1_strain[i,:sig_set[0].shape[0]] = sig_set[0]
        l1_strain[i,:sig_set[1].shape[0]] = sig_set[1]
        v1_strain[i,:sig_set[2].shape[0]] = sig_set[2]
        
        pure_sig_set = convert_cbc_array_to_np(pure_signals[i], signal_len)
        h1_pure_strain[i,:pure_sig_set[0].shape[0]] = pure_sig_set[0]
        l1_pure_strain[i,:pure_sig_set[1].shape[0]] = pure_sig_set[1]
        v1_pure_strain[i,:pure_sig_set[2].shape[0]] = pure_sig_set[2]
        #Join time as one numpy array
        time_set = convert_cbc_array_to_np([noisy_signals[i]['H1'].sample_times, noisy_signals[i]['L1'].sample_times,
                                            noisy_signals[i]['V1'].sample_times], signal_len)
        h1_time[i, :time_set[0].shape[0]] = time_set[0]
        l1_time[i, :time_set[1].shape[0]] = time_set[1]
        v1_time[i, :time_set[2].shape[0]] = time_set[2]

    signal_dict = {
        'H1':(h1_strain, h1_time, h1_pure_strain),
        'L1':(l1_strain, l1_time, l1_pure_strain),
        'V1':(v1_strain, v1_time, v1_pure_strain)
    }
    with h5py.File(file_path, 'a') as hdf:
        append = True if 'signals' in list(hdf.keys()) else False
        if not append:
            sig_group = hdf.create_group('signals')
            for det, (sig_strain, sig_time, pure_strain) in list(signal_dict.items()):
                #First dimension of maxshape needs to be None so that the datasets can be resized
                sig_group.create_dataset(name='{0} strain'.format(det),
                                            dtype='float32',
                                            shape=sig_strain.shape,
                                            data=sig_strain,
                                            maxshape=(None, signal_len))
                sig_group.create_dataset(name='{0} times'.format(det),
                                            dtype='float32',
                                            shape=sig_time.shape,
                                            data=sig_time,
                                            maxshape=(None, signal_len))
                sig_group.create_dataset(name='{0} pure strain'.format(det),
                                            dtype='float32',
                                            shape=pure_strain.shape,
                                            data=pure_strain,
                                            maxshape=(None, signal_len))
            inj_param_group = hdf.create_group('injection_parameters')
            for key, arr in sig_param_arrs.items():
                inj_param_group.create_dataset(name='{0}'.format(key),
                                                dtype='float32',
                                                shape=arr.shape,
                                                data=arr,
                                                maxshape=(None,))

            #store ranges used in this simulation run
            sim_ranges_group = hdf.create_group('simulation_ranges')
            for key, val in list(sim_params.items()):
                sim_ranges_group.attrs[key]=val
        else:
            sig_group = hdf['signals']
            for det, (sig_strain, sig_time, pure_strain) in list(signal_dict.items()):
                det_strain = sig_group['{0} strain'.format(det)]
                det_strain.resize((det_strain.shape[0]+sig_strain.shape[0], signal_len))
                det_strain[-sig_strain.shape[0]:] = sig_strain

                det_times = sig_group['{0} times'.format(det)]
                det_times.resize((det_times.shape[0]+sig_time.shape[0], signal_len))
                det_times[-sig_time.shape[0]:] = sig_time
                
                det_pure_strain = sig_group['{0} pure strain'.format(det)]
                det_pure_strain.resize((det_pure_strain.shape[0]+pure_strain.shape[0], signal_len))
                det_pure_strain[-pure_strain.shape[0]:] = pure_strain
            inj_param_group=hdf['injection_parameters']
            for key, arr in sig_param_arrs.items():
                param_group = inj_param_group[key]
                param_group.resize((param_group.shape[0]+arr.shape[0],))
                param_group[-arr.shape[0]:] = arr 


#doesnt have simulation ranges
def to_hdf_real(file_path, data, sig_params, signal_len):
    #Initialise arrays to hold strain, sample_times, and parameters
    shape = (len(data), int(signal_len))

    if data['H1'].shape[0]!=signal_len:
        print(('WARNING: Signal of len {0} is being truncated to {1}'.format(data['H1'].shape[0], signal_len)))
    
    with h5py.File(file_path, 'a') as hdf:
        sig_group = hdf.create_group('signals')
        for det, sig_strain in list(data.items()):
            #First dimension of maxshape needs to be None so that the datasets can be resized
            sig_group.create_dataset(name='{0} strain'.format(det),
                                        dtype='float32',
                                        shape=sig_strain.shape,
                                        data=sig_strain,
                                        maxshape=(None, signal_len))

        attr_mgr = sig_group.attrs
        for key, val in sig_params.items():
            attr_mgr.create(name=key, data=val)

