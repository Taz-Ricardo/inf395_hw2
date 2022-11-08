import zipfile as zf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import filecmp as fcmp
import os
import scipy as sp
from PIL import Image

import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input, Dropout, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D, AveragePooling2D
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import gradient_descent_v2
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import BatchNormalization
from keras.utils import np_utils
from keras.callbacks import History, EarlyStopping

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import cross_val_score, KFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline

from pathlib import Path


data_dir  = Path.cwd()/'Grape'

X = list()
y = list()

for dirname, _, filenames in os.walk(data_dir):           
    for filename in filenames: 
        image = Image.open(os.path.join(dirname, filename))
        X.append(np.array(image, dtype=np.uint8))
        disease = dirname.split('\\')[-1][8:] # get the name of the disease
        y.append(disease)   
X = np.asarray(X)
y = np.asarray(y)

# Creating instance of labelencoder
le = LabelEncoder()

# Fit the transformer
le.fit(y)
y_le = le.transform(y)


# Creating instance of onehotencoder
ohe = OneHotEncoder(handle_unknown='ignore')

# Fit the transformer
ohe.fit(y_le.reshape(-1, 1))
y_ohe = ohe.transform(y_le.reshape(-1, 1)).toarray()

from sklearn.model_selection import StratifiedShuffleSplit

sf = StratifiedShuffleSplit(n_splits=1, test_size=0.9, random_state=42)

for train_index, val_index in sf.split(X, y_ohe):
    X_train, X_test = X[train_index], X[val_index]
    y_train, y_test = y_ohe[train_index], y_ohe[val_index]  

sf2 = StratifiedShuffleSplit(n_splits=1, test_size=8.5/9, random_state=42)

for train_index, test_index in sf2.split(X_train, y_train):
    X_train, X_val = X_train[train_index], X_train[test_index]
    y_train, y_val = y_train[train_index], y_train[test_index]  

# Perform Standarization
def prep_normalize(X_train, X_val, X_test):	
		
		x_means = X_train.mean(axis=(0,1,2), keepdims=True)
		x_std = X_train.std(axis=(0,1,2), keepdims=True)
		
		X_train	= ((X_train - x_means) / x_std).astype(np.float32)
		X_val   = ((X_val - x_means) / x_std).astype(np.float32)
		X_test  = ((X_test - x_means) / x_std).astype(np.float32)

		# Standarize each channel separately.
		return X_train, X_val, X_test

X_train, X_val, X_test = prep_normalize(X_train, X_val, X_test)

parameters_to_test = {'pooling_function': [MaxPooling2D(strides=(3, 3)), MaxPooling2D(strides=(5,5)) , AveragePooling2D(strides=(3, 3)), AveragePooling2D(strides=(5,5))],
                      'pool_size': [(2, 2)],
                      'dense_layers': [[16], [32], [64,32], [128,8]],
                      }

class parameters_nn:
    def __init__(self):
        self.inputs = Input(shape=X_train.shape[1:])
        self.num_classes = 4
        self.filters = 128
        self.kernel_size = (3,3)
        self.strides=(1,1)
        self.padding = 'same'
        self.dilation_rate=(1,1)
        self.activation = 'relu'
        self.pooling_function = MaxPooling2D(strides=(2,2))
        self.dense_layers = [32]

#cnn with variable amount of convolutional layers
def cnn_model(inputs,num_classes,conv_blocks,conv_filters):
    x = inputs
    # create model    
    for i in range(conv_blocks):
        conv1  = Conv2D(filters = conv_filters[i],
                        kernel_size = (3,3),
                        strides=(1, 1),
                        padding = 'valid',
                        activation = 'relu',
                        data_format="channels_last")(x)
        conv2  = Conv2D(filters = conv_filters[i],
                        kernel_size = (3,3),
                        strides=(1, 1),
                        padding = 'valid',
                        activation = 'relu',
                        data_format="channels_last")(conv1)

        pool1  = MaxPooling2D(pool_size = (2,2))(conv2)

        x = pool1

    dense1 = Dense(units=32, activation='relu')(Flatten()(x))
    

    dense2 = Dense(units=num_classes, activation='softmax')(dense1)

    model  = Model(inputs = inputs, outputs = dense2)    
    
    return model
def power2list(length, number):
    return [int(number/2**i) for i in range(length)]


print('----------------------\n\n\nbeginning grid testing\n\n\n----------------------')
