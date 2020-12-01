import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import nn_utils
import matplotlib.pyplot as plt
import json

data_dir = '/fred/oz016/djacobs/Datasets/new_snr/'
strains = nn_utils.load_signals(data_dir)
chirp_masses = nn_utils.get_chirp_masses_single(data_dir)
tt_dict = nn_utils.train_test_split([strains['H1'], strains['L1'], strains['V1'], chirp_masses])
train = tt_dict['train']
test = tt_dict['test']

ae = keras.models.load_model('ae_20-30_o2_psd.h5')

#Clean signals and split them into train/test arrays
x_train = {
     'H1':ae.predict(train[0]),
     'L1':ae.predict(train[1]),
     'V1':ae.predict(train[2])
}
y_train = train[3]
x_test = {
     'H1':ae.predict(test[0]),
     'L1':ae.predict(test[1]),
     'V1':ae.predict(test[2])
}
y_test = test[3]

#Dropout rate
dr=0.2

#Dense layer size
ls=1024

h_in = keras.Input(shape=(512,))
l_in = keras.Input(shape=(512,))

h = keras.layers.Dense(ls, activation='elu')(h_in)
h = keras.layers.BatchNormalization()(h)
h = keras.layers.Dropout(dr)(h, training=True)
l = keras.layers.Dense(ls, activation='elu')(l_in)
l = keras.layers.BatchNormalization()(l)
l = keras.layers.Dropout(dr)(l, training=True)
for i in range(2):
    h = keras.layers.Dense(ls, activation='elu')(h)
    h = keras.layers.BatchNormalization()(h)
    h = keras.layers.Dropout(dr)(h, training=True)
    
    l = keras.layers.Dense(ls, activation='elu')(l)
    l = keras.layers.BatchNormalization()(l)
    l = keras.layers.Dropout(dr)(l, training=True)

m = keras.layers.Concatenate()([h, l])
for i in range (4):
    m = keras.layers.Dense(ls, activation='elu')(m)
    m = keras.layers.BatchNormalization()(m)
    m = keras.layers.Dropout(dr)(m, training=True)
m = keras.layers.Dense(1)(m)
model = keras.Model(inputs=[h_in, l_in], outputs=[m])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss='mse', metrics=['mae'])

cp = keras.callbacks.ModelCheckpoint('best_model.h5',monitor='val_loss', save_best_only=True, verbose=True)
es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15)
history = model.fit(x=[x_train['H1'], x_train['L1']], y=y_train, epochs=200, callbacks=[cp,es], validation_split=0.1, batch_size=2000)

plt.plot(history.history['val_loss'])
plt.savefig('val_loss.png')
json.dump(history.history, open('two_input_history.json', 'w'))
model = keras.models.load_model('best_model.h5')
model.save('two_input_mean_pred_o2psd_dr2.h5')
model.evaluate(x=[x_test['H1'], x_test['L1']])

