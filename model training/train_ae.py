import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Reshape, Conv1D, MaxPooling1D, Bidirectional, Dropout, Flatten, RepeatVector, Dense, Activation, LSTM
from tensorflow.keras.models import Sequential
import tensorflow.keras.backend as K
from tensorflow.keras.initializers import  VarianceScaling
import h5py
import nn_utils
import utils

def standardise_strain(a):
    mu = np.mean(a, axis=1)[:, np.newaxis]
    std = np.std(a, axis=1)[:, np.newaxis] 
    mu = np.concatenate([mu for x in range(512)], axis=1)
    std = np.concatenate([std for x in range(512)], axis=1)
    res = (a - mu)/std
    return res

data_dir = '/fred/oz016/djacobs/Datasets/new_snr/snr_20_30/'
strains = nn_utils.load_signals(data_dir)
#fp = 'snr_10_20_100k_6.hdf'
#strains, sfreq = utils.load_strain(data_dir+fp, encoding_type=0)
#strains = utils.downsample_strains(strains, sfreq, 2048)
#with h5py.File(data_dir+fp, 'r') as f:
#    pure_sigs = {
#        'H1':f['signals']['H1 pure strain'][()],
#        'L1':f['signals']['L1 pure strain'][()],
#        'V1':f['signals']['V1 pure strain'][()]
#    }
#    pure_sigs = utils.downsample_strains(pure_sigs, 4096, 2048)
#print(strains['H1'].shape)
#strains = {
#    'H1':standardise_strain(strains['H1']),
#    'L1':standardise_strain(strains['L1']),
#    'V1':standardise_strain(strains['V1'])
#}
tt_dict = nn_utils.train_test_split([strains['H1'], strains['L1'], strains['V1']])

x_train = tt_dict['train']
x_test = tt_dict['test']
#ps_dict = nn_utils.train_test_split([pure_sigs['H1'], pure_sigs['L1'], pure_sigs['V1']])
#y_train = ps_dict['train']
#y_test = ps_dict['test']
x_train = np.concatenate((x_train[0], x_train[1], x_train[2]))
x_test = np.concatenate((x_test[0], x_test[1], x_test[2]))
#y_train = np.concatenate((y_train[0], y_train[1], y_train[2]))
#y_test = np.concatenate((y_test[0], y_test[1], y_test[2]))

model = Sequential()
model.add(Reshape((512,1), input_shape=(512,)))
model.add(Conv1D(32,8, padding='same', input_shape=(512,1)))
model.add(Activation('tanh'))
model.add(MaxPooling1D(pool_size=2, padding='same'))
model.add(Conv1D(16,8, padding='same'))
model.add(Activation('tanh'))
model.add(MaxPooling1D(pool_size=2, padding='same'))
model.add(Conv1D(8,8, padding='same'))
model.add(Activation('tanh'))
model.add(MaxPooling1D(pool_size=2, padding='same'))
model.add(Flatten())
model.add(RepeatVector(1))
model.add(Bidirectional(LSTM(200, activation = 'tanh', kernel_initializer='glorot_normal', return_sequences=True)))
model.add(Dropout(rate = 0.2))
model.add(Bidirectional(LSTM(200, activation = 'tanh', kernel_initializer='glorot_normal', return_sequences=True)))
model.add(Dropout(rate = 0.2))
model.add(Bidirectional(LSTM(200, activation = 'tanh', kernel_initializer='glorot_normal', return_sequences=True)))
model.add(Dropout(rate = 0.2))
model.add(Bidirectional(LSTM(200, activation = 'tanh', kernel_initializer='glorot_normal', return_sequences=True)))
model.add(Dropout(rate = 0.2))
model.add(Bidirectional(LSTM(200, activation = 'tanh', kernel_initializer='glorot_normal')))
model.add(Dropout(rate = 0.2))
model.add(Dense(512))


model.compile(optimizer='adam',loss='mse',metrics=['accuracy'])
model.summary()

mc = keras.callbacks.ModelCheckpoint('/fred/oz016/djacobs/models/best_ae.h5', monitor='val_loss', verbose=1,save_best_only=True)
es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
model.fit(x=x_train, y=x_train, epochs=300, validation_split=0.15, callbacks=[mc, es], batch_size=2000)

model = keras.models.load_model('/fred/oz016/djacobs/models/best_ae.h5')
model.save('/fred/oz016/djacobs/models/ae_20-30_o2_psd.h5')


