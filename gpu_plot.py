"""
on GPU0
"""
from keras.models import Sequential
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.utils import multi_gpu_model
from tensorflow.keras.applications import Xception

import numpy as np
import pandas as pd
import pylab as plt
from keras.layers import Dense, AveragePooling2D, MaxPooling2D, AveragePooling3D, MaxPooling3D, Reshape, Flatten, Conv2D,GRU, Input, Dropout, Activation, LeakyReLU
from keras.layers.noise import AlphaDropout
from keras.models import Model
from keras.utils import plot_model
from os.path import join
from os import listdir
import time, os
import tensorflow as tf
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras import optimizers


#K.set_image_dim_ordering('th')
K.set_image_data_format('channels_last')
#K.set_image_data_format('channels_first')
'''
#config = tf.ConfigProto()
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
#sconfig.gpu_options.per_process_gpu_memory_fraction = 0.9
#sess = tf.Session(config=config)
K.tensorflow_backend.set_session(tf.compat.v1.Session(config=config))
'''
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

##################
# set data
##################
X_train=np.load('./data/X_train.npz')['sequence_array']
y_train=np.load('./data/y_train.npz')['sequence_array']
X_test=np.load('./data/X_test.npz')['sequence_array']
y_test=np.load('./data/y_test.npz')['sequence_array']

def normalization(data):
    return data / np.max(abs(data))

def standardization(data):
    mu=np.mean(data)
    sigma=np.std(data)
    return (data-mu) / sigma

mask = np.isnan(X_train)
#X_train=X_train/1000
X_train[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), X_train[~mask])
#X_train = X_train / np.linalg.norm(X_train)
#x 172530/81=2130
mask2 = np.isnan(y_train)
#y_train[mask2] = 0
y_train[mask2] = np.interp(np.flatnonzero(mask2), np.flatnonzero(~mask2), y_train[~mask2])
#y_train = y_train / np.linalg.norm(y_train)
print(np.max(X_train),np.max(y_train))

X_train=standardization(X_train)
y_train=standardization(y_train)

print('Data is ready!')
'''
mask = np.isnan(X)
X[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), X[~mask])
mask2 = np.isnan(y)
y[mask2] = np.interp(np.flatnonzero(mask2), np.flatnonzero(~mask2), y[~mask2])
np.argwhere(np.isnan(y))
'''
'''
from sklearn.preprocessing import normalize
norm_X = X / np.linalg.norm(X)
#norm_X2 = normalize(X[:,np.newaxis], axis=[2,3]).ravel()
#print np.all(norm_X1 == norm_X2)
history=seq.fit(norm1[:79], y[:79], batch_size=10, epochs=30, validation_split=0.05)
'''

#tf.config.experimental.list_physical_devices()
#strategy = tf.distribute.MirroredStrategy()
##################
# define model
##################
def build_model(f,k1,k2,activation,dropout_rate,optimizer):
    model = Sequential()
    model.add(ConvLSTM2D(filters=f, kernel_size=k1,
                        activation=activation,
                       input_shape=(None, 11, 11, 4),
                       padding='same', return_sequences=True))
    #model.add(Conv2D(32, kernel_size=(3, 3), activation='linear',input_shape=(30, 30, 1)))
    #model.add(LeakyReLU(alpha=0.01))
    #model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(ConvLSTM2D(filters=f, kernel_size=k2,
                        activation=activation,
                       padding='same', return_sequences=True))
    #model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    model.add(BatchNormalization())   
    model.add(Dropout(dropout_rate))

    model.add(ConvLSTM2D(filters=f, kernel_size=k2,
                        activation=activation,
                       padding='same', return_sequences=True))
    #model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    model.add(ConvLSTM2D(filters=f, kernel_size=k1,
                        activation=activation,
                       padding='same', return_sequences=True))
    #model.add(MaxPooling3D(pool_size=(3, 3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))    
    #model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    model.add(Conv3D(filters=1, kernel_size=(3, 3, 3), activation='relu', padding='same', data_format='channels_last'))
    #model.add(AveragePooling3D(pool_size=(3, 3, 3), padding='same'))
    #model.add(AveragePooling3D((1, 80, 80)))
    #model.add(Reshape((-1, 40)))
    #model.add(Dense(1, kernel_initializer=initializers.RandomUniform(minval=0, maxval=0.05, seed=None)))
    #model.add(AveragePooling2D((2,2)))  #sum
    #model.add(Flatten())
    #model.add(Dense(100, activation='relu'))
    #model.add(Dropout(0.1))
    #model.add(Dense(1))
    model.compile(loss='mse', optimizer=optimizer)
    return model

##################
# act_func
##################
def gelu(x):
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))
get_custom_objects().update({'gelu': Activation(gelu)})
get_custom_objects().update({'leaky-relu': Activation(LeakyReLU(alpha=0.2))})
#act_func = ['sigmoid', 'relu', 'elu', 'leaky-relu', 'selu', 'gelu']
act_funcs = ['relu', 'tanh']
learning_rates = [0.004, 0.002, 0.001]
dropout_rates = [0.1, 0.3]
batch_sizes = [8,4,1]
##################
#loacte on GPU
##################
with tf.device('/GPU:1'):
    #result=[]
    results={}
    for activation in act_funcs:
        for lr in learning_rates: 
            for dr in dropout_rates:
                for bs in batch_sizes:
                    optimizer=optimizers.Adam(learning_rate=lr)  # beta_1=0.9, beta_2=0.999, amsgrad=False, clipvalue=0.5
                    print('\nTraining with:\nactivation function: {}\nlearning rate: {}\ndropout rate: {}\nbatch sizes: {}\n'.format(activation,lr,dr,bs))
                    
                    model = build_model(f=64,k1=(3,3),k2=(3,3),
                                  activation=activation,
                                  dropout_rate=dr,
                                  optimizer=optimizer #Adam(clipvalue=0.5)
                                  )
                    print('Model is built!')

                    history = model.fit(X_train[:100], y_train[:100],
                          validation_split=0.20,
                          batch_size=bs, # 128 is faster, but less accurate. 16/32 recommended
                          epochs=100,
                          verbose=1 #,
                          #validation_data=(x_test, y_test)
                          )
                    time_stamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
                    dirName = join(r'C:/Users/mu.yang/UM_Test','outputs',time_stamp)
                    os.mkdir(dirName)
                    hparams = activation + '_' + str(lr) + '_' + str(dr) + '_'+ str(bs)
                    model.save(join(dirName,hparams +'_model.h5'))
                    model.save_weights(join(dirName,hparams +'_weights.h5'))
                    
                    #result.append(history)
                    results[(activation,lr,dr,bs)]=history
                    K.clear_session()
                    del model

    print(results)


##########################
#   plot result
##########################
# act_funcs = ['sigmoid','relu', 'tanh','elu']
# learning_rates = [0.001, 0.0005, 0.0001]
# dropout_rates = [0.1, 0.3, 0.6, 0.9]
# batch_sizes = [8,4,2,1]

def plot_results(results, activation_functions = [], learning_rates = [], dropout_rates=[], batch_sizes = []):    
    plt.style.use('dark_background')
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    for k in results:
        axs[0].plot(results[k].history['loss'])
        axs[1].plot(results[k].history['val_loss'])

    fig.suptitle('Loss Plotting')
    #axs[0].set_title('Train loss')
    axs[0].set_ylabel('Train loss')
    axs[0].set_xlabel('Epoch')
    axs[0].legend(results.keys())

    axs[1].plot(results[k].history['val_loss'])
    #axs[1].set_title('Val. loss')
    axs[1].set_ylabel('Val. loss')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(results.keys())

    plt.savefig('./outputs/plot_' + time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime()) +'.png')  # dpi=300
    #plt.show()

#plot_act_func_results(new_results, new_act_arr)
plot_results(results, act_funcs, learning_rates,dropout_rates,batch_sizes)


# with tf.device('/GPU:1'):
#     model = model(f=256,k1=(5,5),k2=(5,5),d=0.3)
#     model.compile(loss='mse', optimizer='adam')
#     history=model.fit(X_train, y_train, batch_size=100, epochs=100, validation_split=0.05)
#     time_stamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
#     dirName = join(r'C:/Users/mu.yang/UM_Test','outputs',time_stamp)
#     os.mkdir(dirName)
#     model.save(join(dirName,'py_model.h5'))
#     model.save_weights(join(dirName,'py_weights.h5'))

# with tf.device('/CPU:0'):
#     model = model(f=64,k1=(2,2),k2=(2,2),d=0.1)
#     model.compile(loss='mse', optimizer='adam')
#     history=model.fit(X_train, y_train, batch_size=100, epochs=100, validation_split=0.05)
#     time_stamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
#     dirName = join(r'C:/Users/mu.yang/UM_Test','outputs',time_stamp)
#     os.mkdir(dirName)
#     model.save(join(dirName,'py_model.h5'))
#     model.save_weights(join(dirName,'py_weights.h5'))

'''
parallel_model = multi_gpu_model(model, gpus=2)
parallel_model.compile(loss='mse', optimizer='sgd')

history=parallel_model.fit(X, y, batch_size=100, epochs=100, validation_split=0.05)
'''
