import h5py as h5
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
# GPU on
gpus = tf.config.list_physical_devices('GPU')
print("The gpu' are:")
for gpu in gpus:
    print(gpu)
    tf.config.experimental.set_memory_growth(gpu, True)

import nn_builds.nn_constructor as nc
nc.tfl = tf.keras.layers
import nn_builds.nn as nn
nn.tf = tf
nn.tfl = tf.keras.layers
nn.nc = nc

from training.losses import focal_loss as loss

import training.ds_making as DS
DS.np = np
DS.h5 = h5
DS.tf = tf

import training.train as TR
TR.datetime = datetime
TR.loss = loss
TR.make_dataset = DS.make_dataset
TR.h5 = h5
TR.tf = tf
TR.plt = plt

# data and model's names
data_names = [n for n in os.listdir('./data/') if n.endswith('.h5')]
for i, h5n in enumerate(data_names):
    print(str(i + 1), ". " + h5n)
i = int(input("Which dataset do you want to use? Print it's number! \n"))
name = data_names[i - 1]
# name = 'baikal_multi_0523_flat_pureMC_h5s2_norm.h5'
path_to_h5 = './data/' + name

# scripts for NN
from training.train import train_model, make_train_figs

model_names = [n for n in dir(nn) if n.startswith('nn')]
for i, mn in enumerate(model_names):
    print(str(i + 1), ". " + mn)
i = int(input("Which model do you want to train? Print it's number! \n"))
model_name = model_names[i - 1]
# model_name = 'nn_rnn_model'
# making dir for model if necessary
try:
    os.makedirs('./trained_models/' + model_name)
    print('directory for the model is created')
except:
    print('directory for the model already exists')

# getting the shape of data
Shape = (None, 6)

# set hyperparams
lr_initial = 0.005  # tuned
batch_size = 256


# making model
model_func = getattr(nn, model_name)
model = model_func(Shape, u_list=[32, 16])
model_name = input("How do you want to call the model/trial?")
# model_name = 'test'

trigger = input("Do you want to see model's summary? Type only 'y' or 'n': \n")
#trigger = 'n'
if trigger == 'y':
    print(model.summary())
elif trigger == 'n':
    pass
else:
    print("Your input is incorrect. Summary will not be shown.")

# settings for training
epochs = int(input("Print max number of epochs that you want in the trial: \n"))
# epochs = 2
# trigger = input("Do you want to see verbose while training? Type only 'y' or 'n': \n")
trigger = 'y'
if trigger == 'y':
    v = 1
elif trigger == 'n':
    v = 0
else:
    print("Your input is incorrect. Verbose will not be shown.")
    v = 0

# training model and creating figs
history = TR.train_model(model, path_to_h5, batch_size, lr_initial, model_name, shape=Shape,
                         num_of_epochs=epochs, verbose=v)
TR.make_train_figs(history, model_name)
