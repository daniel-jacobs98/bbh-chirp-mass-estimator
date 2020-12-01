import numpy as np
import h5py
import os
import utils

#Iterates through set of files in a data folder (must all be .hdf files) and loads their signals,
#downsampling to 2048Hz
#Returns a dict with keys [H1, L1, V1]
def load_signals(data_directory):
    strains = []
    files = (f for f in os.listdir(data_directory) if os.path.isfile(os.path.join(data_directory, f)))
    for fp in files:
        print(fp)
        try:
            strain_x, sfreq = utils.load_strain(data_directory+fp, encoding_type=0)
        except:
            continue
        strains.append(strain_x)
    signals = {
        'H1':np.concatenate([x['H1'] for x in strains]),
        'L1':np.concatenate([x['L1'] for x in strains]),
        'V1':np.concatenate([x['V1'] for x in strains])
    }
    return signals

#Loads a single real signal
def load_real_sig(file_path):
    strain = utils.load_strain(file_path, encoding_type=0, sfreq=4096)
    signals = {
        'H1':strain['H1'],
        'L1':strain['L1'],
        'V1':strain['V1']
    }
    return signals

#Takes a list of datasets, shuffles them and splits them into training and test sets.
#Returns a dict (train, test) where each value is a list in the order the datasets were passed
#TODO: Throw exception if different sizes
def train_test_split(datasets, split=0.9):
    first_size = datasets[0].shape[0]
    ret = {
        'train':[],
        'test':[]
    }
    for dset in datasets:
        if dset.shape[0]!=first_size:
            print('WARNING: Datasets do not have the same number of elements\n...exiting')
            return
        rng = np.random.RandomState(seed=42)
        rng.shuffle(dset)
        idx = int(split*dset.shape[0])
        ret['train'].append(dset[:idx])
        ret['test'].append(dset[idx:])
    return ret

#Iterates through a set of data files (all .hdf) and gets their chirp masses
#Returns an numpy array (One CM per set of (Hanford, Livingston, Virgo) signals)
def get_chirp_masses_single(data_directory):
    chirp_masses = []
    files = (f for f in os.listdir(data_directory) if os.path.isfile(os.path.join(data_directory, f)))
    for fp in files:
        try:
            with h5py.File(data_directory+fp, 'r') as f:
                m1 = f['injection_parameters']['mass1'][()]
                m2 = f['injection_parameters']['mass2'][()]
                cm = ((m1*m2)**(3.0/5))/((m1+m2)**(1.0/5))
                chirp_masses.append(cm[:, np.newaxis])
        except:
            continue
    return np.concatenate(chirp_masses)

def get_detector_snrs(data_dir):
    snrs = {
        'H1':[],
        'L1':[],
        'V1':[]
    }
    files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
    for fp in files:
        with h5py.File(data_dir+fp, 'r') as f:
            snrs['H1'].append(f['injection_parameters']['h1_snr'][()])
            snrs['L1'].append(f['injection_parameters']['l1_snr'][()])
            snrs['V1'].append(f['injection_parameters']['v1_snr'][()])
    ret = {
        'H1':np.concatenate(snrs['H1']),
        'L1':np.concatenate(snrs['L1']),
        'V1':np.concatenate(snrs['V1'])
    }
    return ret           
