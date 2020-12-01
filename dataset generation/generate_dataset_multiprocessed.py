'''
    Daniel Jacobs 2020
    OzGrav - University of Western Australia
'''

from pycbc.waveform import td_approximants
from pycbc.waveform import get_td_waveform
from pycbc.detector import Detector
import pycbc.noise
import pycbc.psd
from pycbc.filter import sigma
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random as rand
import math
import argparse
import json
import h5py
import numpy as np
from generation_utils import fade_on, to_hdf
import time
import sys
import os.path
from lal import LIGOTimeGPS
from plot_signal import plot_sigs, plot_with_pure, find_nearest
import copy
import multiprocessing as mp
import time
import re
import pycbc.conversions
import pandas as pd
import pycbc.filter.resample
import pycbc.types.timeseries


#Defines ranges that parameters are sampled from
global sim_params
sim_params = {
    # Black hole mass range
    #'m_chirp_range':(7.5, 50),
    'm_chirp_range':(4.352,69.64),
    'eta_range':(0.055, 0.25),
    'spin_range': (-1,1), 
    'num_signals': 1,
    'inclination_range':(0, math.pi),
    'coa_phase_range':(0, 2*math.pi),
    'right_asc_range':(0, 2*math.pi),
    'declination_range':(0, 1),
    'polarisation_range':(0, 2*math.pi),
    'distance_range':(40, 3000),
    'snr_range':(30,40),
    'sample_freq':8192
}

detector_noise_dir = '/fred/oz016/djacobs/Datasets/'




param_pool = {'m_chirp':[], 'eta':[], 'm1':[], 'm2':[]}

#Generates a bunch of parameter sets and then cuts it down to ones fitting out m1 and m2 parameters
for x in range(500000):
    param_pool['m_chirp'].append(rand.uniform(sim_params['m_chirp_range'][0], sim_params['m_chirp_range'][1]))
    param_pool['eta'].append(round(rand.uniform(sim_params['eta_range'][0], sim_params['eta_range'][1]),4))
    param_pool['m1'].append(pycbc.conversions.mass1_from_mchirp_eta(param_pool['m_chirp'][-1], param_pool['eta'][-1]))
    param_pool['m2'].append(pycbc.conversions.mass2_from_mchirp_eta(param_pool['m_chirp'][-1], param_pool['eta'][-1]))
pool_df = pd.DataFrame(data={'m_chirp':np.array(param_pool['m_chirp']), 'eta':np.array(param_pool['eta']), 'm1':np.array(param_pool['m1']), 'm2':np.array(param_pool['m2'])})
pool_df = pool_df.loc[pool_df['m1']<80]
pool_df = pool_df.loc[pool_df['m2']>5]

global pools
pools = []
start_idx = 4
while start_idx<70:
    pool_temp = pool_df.loc[(pool_df['m_chirp']>=start_idx) & (pool_df['m_chirp']<(start_idx+2))]
    pools.append(pool_temp)
    print('Pool {0}-{1}: {2}'.format(start_idx, start_idx+2, len(pool_temp.index)))
    start_idx+=2

global signal_len
signal_len=0.25*sim_params['sample_freq']

def get_real_param_set(sig):
    real_params = {
        'gw151012':{'snr':10.0,'m1':23.2,'m2':13.6,'dist':1080,'x1':0.05,'x2':0.05},
        'gw151226':{'snr':13.1,'m1':13.7,'m2':7.7,'dist':450,'x1':0.23,'x2':0.091},
        'gw170104':{'snr':13.0,'m1':30.8,'m2':20.0,'dist':990,'x1':0.1,'x2':-0.2556},
        'gw170608':{'snr':14.9,'m1':11.0,'m2':7.6,'dist':320,'x1':0.08,'x2':-0.0424},
        'gw170729':{'snr':10.8,'m1':50.2,'m2':34.0,'dist':2840,'x1':0.2,'x2':0.621},
        'gw170809':{'snr':12.4,'m1':35.0,'m2':23.8,'dist':1030,'x1':0.1,'x2':0.05},
        'gw170814':{'snr':15.9,'m1':30.6,'m2':25.2,'dist':600,'x1':0.25,'x2':-0.148},
        'gw170818':{'snr':11.3,'m1':35.4,'m2':26.7,'dist':1060,'x1':0.1,'x2':-0.34},
        'gw170823':{'snr':11.5,'m1':39.5,'m2':29.0,'dist':1940,'x1':-0.1,'x2':0.348}
    }
    
    param_set = real_params[sig]
    param_set['inc'] = rand.uniform(sim_params['inclination_range'][0], sim_params['inclination_range'][1])
    param_set['coa'] = rand.uniform(sim_params['coa_phase_range'][0], sim_params['coa_phase_range'][1])
    param_set['ra'] = rand.uniform(sim_params['right_asc_range'][0], sim_params['right_asc_range'][1])
    param_set['dec'] = math.asin(1-(2*rand.uniform(sim_params['declination_range'][0], sim_params['declination_range'][1])))
    param_set['pol'] = rand.uniform(sim_params['polarisation_range'][0], sim_params['polarisation_range'][1])
    param_set['f'] = 8192
    yield param_set

#Yield a parameter set describing a signal uniformly samples from the sim_param ranges
def get_param_set(sim_params):
    param_set = {}
    for i in range(0, sim_params['num_signals']):
        target_pool = rand.choice(pools)
        print('t_pool: {0}'.format(len(target_pool.index)))
        row_choice = target_pool.sample().iloc[0]
        param_set['m1'] = row_choice['m1']
        param_set['m2'] = row_choice['m2']
        param_set['x1'] = rand.uniform(sim_params['spin_range'][0], sim_params['spin_range'][1])
        param_set['x2'] = rand.uniform(sim_params['spin_range'][0], sim_params['spin_range'][1])
        param_set['inc'] = rand.uniform(sim_params['inclination_range'][0], sim_params['inclination_range'][1])
        param_set['coa'] = rand.uniform(sim_params['coa_phase_range'][0], sim_params['coa_phase_range'][1])
        param_set['ra'] = rand.uniform(sim_params['right_asc_range'][0], sim_params['right_asc_range'][1])
        param_set['dec'] = math.asin(1-(2*rand.uniform(sim_params['declination_range'][0], sim_params['declination_range'][1])))
        param_set['pol'] = rand.uniform(sim_params['polarisation_range'][0], sim_params['polarisation_range'][1])
        param_set['dist'] = rand.randint(sim_params['distance_range'][0], sim_params['distance_range'][1])
        param_set['f'] = sim_params['sample_freq']
        param_set['snr'] = rand.uniform(sim_params['snr_range'][0], sim_params['snr_range'][1])

        yield param_set


#Generate and return projections of a signal described by param_set onto the Hanford, Livingston, Virgo detectors
def generate_signal(param_set):
    hp, hc = get_td_waveform(approximant='SEOBNRv4', #This approximant is only appropriate for BBH mergers
                            mass1=param_set['m1'],
                            mass2=param_set['m2'],
                            spin1z=param_set['x1'],
                            spin2z=param_set['x2'],
                            inclination_range=param_set['inc'],
                            coa_phase=param_set['coa'],
                            distance=param_set['dist'],
                            delta_t=1.0/param_set['f'],
                            f_lower=30)

    time = 100000000

    det_h1 = Detector('H1')
    det_l1 = Detector('L1')
    det_v1 = Detector('V1')

    hp = fade_on(hp,0.25)
    hc = fade_on(hc,0.25)

    sig_h1 = det_h1.project_wave(hp, hc, param_set['ra'], param_set['dec'], param_set['pol'])
    sig_l1 = det_l1.project_wave(hp, hc, param_set['ra'], param_set['dec'], param_set['pol'])
    sig_v1 = det_v1.project_wave(hp, hc, param_set['ra'], param_set['dec'], param_set['pol'])

    return {'H1':sig_h1,
            'L1':sig_l1,
            'V1':sig_v1}


#Reshape signals to desired length by appending and prepending zeros if necessary
def cut_sigs(signal_dict):
    cut_sigs = dict()
    zeroIdxs = {
        'H1':find_nearest(signal_dict['H1'].sample_times, 0),
        'L1':find_nearest(signal_dict['L1'].sample_times, 0),
        'V1':find_nearest(signal_dict['V1'].sample_times, 0)
    }

    for det in ['H1','L1','V1']:
        zIdx = zeroIdxs[det]
        sig = signal_dict[det]
        startIdx = int(zIdx-(math.floor(signal_len*0.8)))
        prep_zeros = 0
        endIdx = int(zIdx+(math.ceil(signal_len*0.2)))
        ap_zeros = 0
        if startIdx<0:
            prep_zeros = int(startIdx*-1)
            startIdx = 0
        if endIdx>sig.shape[0]:
            ap_zeros = endIdx-sig.shape[0]
            endIdx = sig.shape[0]-1
        res = sig[startIdx:endIdx]
        if res.shape[0]!=signal_len:
            res.prepend_zeros(prep_zeros)
            res.append_zeros(ap_zeros)
        resampled = res[()][::4]
        resampled = pycbc.types.timeseries.TimeSeries(resampled, delta_t=1.0/2048, 
                        epoch=res.sample_times[0])
        cut_sigs[det]=resampled
    return cut_sigs

#Gets O2 detector noise from each detector at a random time
#Chosen time is same for all detectors
def get_noise(sfreq):
    h1_files = os.listdir(detector_noise_dir+'h1_noise/')
    hasnans = True
    while hasnans:
        hasnans=False
        chosen_file = rand.choice(h1_files)   
        res = re.search('R1-(.*)-4096.hdf5', chosen_file)
        start_time = int(res.group(1))
        fps = {
            'H1': detector_noise_dir+'h1_noise/'+chosen_file,
            'L1': detector_noise_dir+'l1_noise/L-L1_GWOSC_O2_4KHZ_R1-'+str(start_time)+'-4096.hdf5',
            'V1': detector_noise_dir+'v1_noise/V-V1_GWOSC_O2_4KHZ_R1-'+str(start_time)+'-4096.hdf5'
        }
        start_slice = rand.randint(0, (4096*4096)-((16*sfreq)+1))
        end_slice = start_slice+(sfreq*16)
        noise = dict()
        for det in ['H1', 'L1', 'V1']: 
            fp = fps[det]
            with h5py.File(fp, 'r') as f:
                noise_strain = f['strain']['Strain'][start_slice:end_slice] 
                if np.isnan(np.sum(noise_strain)):
                    hasnans=True
                noise_ts = pycbc.types.timeseries.TimeSeries(initial_array=noise_strain, delta_t=1.0/4096)
                noise[det] = noise_ts
    if (np.isnan(np.sum(noise['H1'][()]))):
        print('bad noise')    
    return noise


#Inject a set of signals into Gaussian noise with the given SNR
def inject_signals_gaussian(signal_dict, inj_snr, sig_params):
    #trim signals to right length
    resized_sigs = cut_sigs(signal_dict)
    noise = dict()
    global sim_params

    for i, det in enumerate(('H1', 'L1', 'V1')):
        flow = 30.0
        delta_f = resized_sigs[det].delta_f
        flen = int(sim_params['sample_freq'] / delta_f) + 1 
        psd = pycbc.psd.from_txt('{0}_O2_PSD.dat'.format(det), 2049, 0.5, 30.0, is_asd_file=False)
        psd = pycbc.psd.interpolate(psd, delta_f)
        noise[det] = pycbc.noise.gaussian.noise_from_psd(length=sim_params['sample_freq']*16,
                                                        delta_t=1.0/2048, psd=psd)
        
        start_time = resized_sigs[det].start_time-8
        noise[det]._epoch = LIGOTimeGPS(start_time)

    psds = dict()
    dummy_strain = dict()
    snrs = dict()

    #using dummy strain and psds from the noise, calculate the snr of each signal+noise injection to find the 
    #network optimal SNR, used for injecting the real signal
    for det in ('H1', 'L1', 'V1'):
        delta_f = resized_sigs[det].delta_f
        dummy_strain[det] = noise[det].add_into(resized_sigs[det])
        
        psds[det] = dummy_strain[det].psd(0.2)
        psds[det] = pycbc.psd.interpolate(psds[det], delta_f=delta_f)
        snrs[det] = sigma(htilde=resized_sigs[det],
                            psd=psds[det],
                            low_frequency_cutoff=flow)
    nomf_snr = np.sqrt((snrs['H1']**2)+(snrs['L1']**2)+(snrs['V1']**2))
    scale_factor = 1.0* inj_snr/nomf_snr
    noisy_signals = dict()

    ret_snrs=dict()
    #inject signals with the correct scaling factor for the target SNR
    for det in ('H1', 'L1', 'V1'):
        delta_f = resized_sigs[det].delta_f
        noisy_signals[det] = noise[det].add_into(resized_sigs[det]*scale_factor)
        
        psds[det] = noisy_signals[det].psd(0.2)
        psds[det] = pycbc.psd.interpolate(psds[det], delta_f=delta_f)
        snr = sigma(htilde=resized_sigs[det]*scale_factor, 
                    psd=psds[det],
                    low_frequency_cutoff=flow)
        ret_snrs[det] = snr
        

        #Whiten signal
        noisy_signals[det] = noisy_signals[det].whiten(segment_duration=1,
                                                        max_filter_duration=1, 
                                                        remove_corrupted=False,
                                                        low_frequency_cutoff=30.0)

        #Cut down to desired length and cut off corrupted tails of signal
        noisy_signals[det] = noisy_signals[det].time_slice(-0.2, 0.05)
    return noisy_signals, ret_snrs, resized_sigs

#Master function to generate, inject and return a GW signal 
def generate_and_inject_signal():
    try:
        params = next(get_param_set(sim_params))
        signal = generate_signal(params)
        injected_signal, snrs, pure_sigs = inject_signals_gaussian(signal, params['snr'], params)
        params['h1_snr'] = snrs['H1']
        params['l1_snr'] = snrs['L1']
        params['v1_snr'] = snrs['V1']
        results = {
            'signal': injected_signal,
            'parameters': params,
            'pure_signal':pure_sigs
        }
        return results
    except RuntimeError:
        sys.exit('Runtime Error')


if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description='''Generate a GW signal dataset 
                                    with Gaussian noise''')
    parser.add_argument('-o', '--output')
    arguments = parser.parse_args()

    output_path = arguments.output if arguments.output!=None else 'output.hdf'
    if os.path.isfile(output_path):
        print('Output file already exists. Please remove this file or use a different file name.')
        sys.exit(1)

    start=time.time()
    print(('Starting generation of {0} signals...'.format(sim_params['num_signals'])))

    num_generated = 0
    
    #how many signals to keep in RAM before wwwriting to disk
    #5000 works well, too many and you'll run out of memory, too few and it'll be very slow since I/O is the bottleneck
    batch_size = 5000

    #Set processes = # of cores available
    #OR processes = # of cores * 2 if hyper threading capable
    #No damage setting it higher but won't improve performance
    pool = mp.Pool(processes=30)
    while num_generated<sim_params['num_signals']:
        remaining = sim_params['num_signals'] - num_generated
        to_gen = remaining if remaining<5000 else 5000
        res = [pool.apply_async(generate_and_inject_signal) for x in range(0, to_gen)]
        worker_results = [p.get() for p in res]
        num_generated += to_gen
        sig_params = [x['parameters'] for x in worker_results]
        signals = [x['signal'] for x in worker_results]
        pure_signals = [x['pure_signal'] for x in worker_results]
        to_hdf(output_path, sim_params, signals, pure_signals, sig_params, 512)

    end = time.time()
    print(('\nFinished! Took {0} seconds to generate and save {1} samples.\n'.format(float(end-start), sim_params['num_signals'])))
