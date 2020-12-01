from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import linalg_ops_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.util.tf_export import tf_export

import h5py
import numpy as np
import math
import pandas as pd

import tensorflow.keras as keras

def load_strain(file_path, encoding_type, sfreq=None):
  strains = dict()
  with h5py.File(file_path, 'r') as f:
    if encoding_type==0:
      for det in ['H1', 'L1', 'V1']:
        strains[det] = f['signals']['{0} strain'.format(det)][()]
    elif encoding_type==1:
      for det in ['H1', 'L1', 'V1']:
        strains[det] = f['injection_samples']['{0}_strain'.format(det.lower())][()]  
    if sfreq==None:
      sfreq = f['simulation_ranges'].attrs['sample_freq']
      return (strains, sfreq)
    else:
      return strains

def downsample_strains(strains, original_sampling_freq, target_sampling_freq):
  subsample_step = int(original_sampling_freq/target_sampling_freq)
  downsampled_strains = dict()
  for det in ['H1', 'L1', 'V1']:
    ds = strains[det][:,::subsample_step].copy()
    downsampled_strains[det] = ds
  return downsampled_strains

def get_signal_parameters(file_path, encoding_type):
  parameter_dictionaries = []
  with h5py.File(file_path, 'r') as f:
    if encoding_type==0:
      signal_parameters = f['signals']['Signal parameters'][()]
      for parameter_set in signal_parameters:
        parameter_dict = dict()
        for parameter in parameter_set:
          if 'e' in parameter[1].decode('utf-8'):
            parameter[1] = float(parameter[1].decode('utf-8').replace('e-',''))*0.001
          parameter_dict[parameter[0].decode('utf-8')] = float(parameter[1].decode('utf-8'))
        parameter_dictionaries.append(parameter_dict)
    elif encoding_type==1:
      signal_parameters = f['injection_parameters']
      for i in range(0, signal_parameters['coa_phase'].shape[0]):
        param_dict = dict()
        param_dict['coa'] = signal_parameters['coa_phase'][i]
        param_dict['dec'] = signal_parameters['dec'][i]
        param_dict['inc'] = signal_parameters['inclination'][i]
        param_dict['snr'] = signal_parameters['injection_snr'][i]
        param_dict['m1'] = signal_parameters['mass1'][i]
        param_dict['m2'] = signal_parameters['mass2'][i]
        param_dict['pol'] = signal_parameters['polarization'][i]
        param_dict['ra'] = signal_parameters['ra'][i]
        param_dict['x1'] = signal_parameters['spin1z'][i]
        param_dict['x2'] = signal_parameters['spin2z'][i]
        parameter_dictionaries.append(param_dict)

  return parameter_dictionaries

def get_chirp_masses(parameter_dictionaries, single=False):
  chirp_masses = []
  for parameter_dict in parameter_dictionaries:
    m1 = parameter_dict['m1']
    m2 = parameter_dict['m2']
    cm = ((m1*m2)**(3.0/5))/((m1+m2)**(1.0/5))
    chirp_masses.append(cm)
  chirp_masses = np.array(chirp_masses)
  return np.concatenate((chirp_masses, chirp_masses, chirp_masses), axis=0)

def test_dist_predictions(predictions, y_true):
  within_1=0
  within_2=0
  within_3=0
  for i in range(0, predictions.shape[0]):
    t = y_true[i]
    mu = predictions[i][0]
    std = predictions[i][1]

    if t>=(mu-(std*3)) and t<=(mu+(std*3)):
      within_3+=1


    if t>=(mu-(std*2)) and t<=(mu+(std*2)):
      within_2+=1

    if t>=(mu-(std*1)) and t<=(mu+(std*1)):
      within_1+=1

  print('True value within 3 std deviations of mean: {:.2f}%'.format((within_3/predictions.shape[0])*100))
  print('True value within 2 std deviations of mean: {:.2f}%'.format((within_2/predictions.shape[0])*100))
  print('True value within 1 std deviations of mean: {:.2f}%'.format((within_1/predictions.shape[0])*100))

def prepare_dataset(paths, target_frequency):
  strain_arrs=[]
  cm_arrs=[]
  for path, encoding in paths:
    strain = load_strain(path, encoding)
    with h5py.File(path, 'r') as f:
      sample_freq = f['simulation_ranges'].attrs['sample_freq']
    strain = downsample_strains(strain, sample_freq, target_frequency)
    strain = np.concatenate((strain['H1'], strain['L1'], strain['V1']))
    strain_arrs.append(strain)

    cm = get_chirp_masses(get_signal_parameters(path, 1))
    cm = cm[:, np.newaxis]
    cm_arrs.append(cm)

  strain = np.concatenate(strain_arrs)
  cm = np.concatenate(cm_arrs)
  print(strain.shape)
  print(cm.shape)
  return (strain, cm)
