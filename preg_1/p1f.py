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
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import StratifiedShuffleSplit

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


print('----------------------\n\n\nbeginning grid testing\n\n\n----------------------')


def data_gen(X_train, y_train, X_val, y_val, X_test, y_test,rotation_range=0, width_shift_range=0, height_shift_range=0, vertical_flip=True):
    datagen_train = ImageDataGenerator(
                    rotation_range=rotation_range,
                    width_shift_range=width_shift_range,
                    height_shift_range=height_shift_range,
                    vertical_flip=vertical_flip,
                    fill_mode='nearest',
                    data_format="channels_last")

    datagen_val = ImageDataGenerator(
                    rotation_range=rotation_range,
                    width_shift_range=width_shift_range,
                    height_shift_range=height_shift_range,
                    vertical_flip=vertical_flip,
                    fill_mode='nearest',
                    data_format="channels_last")

    datagen_test = ImageDataGenerator(
                    rotation_range=rotation_range,
                    width_shift_range=width_shift_range,
                    height_shift_range=height_shift_range,
                    vertical_flip=vertical_flip,
                    fill_mode='nearest',
                    data_format="channels_last")
    datagen_train.fit(X_train)
    train_generaton = datagen_train.flow(X_train, y_train, batch_size=128)

    datagen_val.fit(X_val)
    val_generaton = datagen_val.flow(X_val, y_val, batch_size=64)

    datagen_test.fit(X_test)
    test_generaton = datagen_test.flow(X_test, y_test, batch_size=64)

    return train_generaton, val_generaton, test_generaton


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
        self.pooling_function = AveragePooling2D(strides=(5,5))
        self.dense_layers = [32]


def best_cnn(parameters_object):
    x = parameters_object.inputs
    # create model    
    conv1  =Conv2D(filters  = parameters_object.filters,
                    kernel_size = parameters_object.kernel_size,
                    strides = parameters_object.strides,
                    padding = parameters_object.padding,
                    dilation_rate = parameters_object.dilation_rate,
                    activation = parameters_object.activation,
                    data_format = 'channels_last')(x)

    pool1  = parameters_object.pooling_function(conv1)

    conv2  =Conv2D(filters  = parameters_object.filters,
                    kernel_size = parameters_object.kernel_size,
                    strides = parameters_object.strides,
                    padding = parameters_object.padding,
                    dilation_rate = parameters_object.dilation_rate,
                    activation = parameters_object.activation,
                    data_format = 'channels_last')(pool1)

    pool2  = parameters_object.pooling_function(conv2)    
    
    if len(parameters_object.dense_layers) == 1:
        dense1 = Dense(units= parameters_object.dense_layers[0], activation='relu')(Flatten()(pool2))    

        dense2 = Dense(units= parameters_object.num_classes, activation='softmax')(dense1)

        model  = Model(inputs = parameters_object.inputs, outputs = dense2)  
    else:
        dense1 = Dense(units= parameters_object.dense_layers[0], activation='relu')(Flatten()(pool2))    
        dense2 = Dense(units= parameters_object.dense_layers[1], activation='relu')(dense1)
        dense3 = Dense(units= parameters_object.num_classes, activation='softmax')(dense2)

        model  = Model(inputs = parameters_object.inputs, outputs = dense3)

    
    return model

# ...

import itertools  

flips = [True, False]
angles =  [20, 40]

parameters_object = parameters_nn()

for flips_, angles_ in itertools.product(flips, angles): 

    train_generaton, val_generaton, test_generaton = data_gen(X_train, y_train, X_val, y_val, X_test, y_test, rotation_range=angles_, vertical_flip=flips_)
    inputs = Input(shape=(256, 256, 3))
    num_classes = 4

    model = best_cnn(parameters_object)
    epochs = 200

    my_callbacks = [History(), EarlyStopping(patience=20, min_delta=0.01, monitor='val_accuracy', mode="max", restore_best_weights = True)]
    
    #compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'], run_eagerly=True)

    #fit model

    history = model.fit(train_generaton, validation_data=val_generaton, epochs=epochs, callbacks=my_callbacks)
    # save history of modedl
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(f'history_flips:{flips_}_angles:{angles_}_p1f.csv')

    #test model
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f'Loss in test data: {test_loss}\nAccuracy in test data: {test_acc}')
    # test model with augmented data
    gen_test_loss, gen_test_acc = model.evaluate(test_generaton)
    print(f'Loss in test data: {test_loss}\nAccuracy in test data: {test_acc}')
    # save model
    with open(f'test_results_flips:{flips_}_angles:{angles_}_p1f.csv', 'w') as f:
        f.write(f'Loss in test data: {test_loss}\nAccuracy in test data: {test_acc}')
        f.write(f'Loss in gen test data: {gen_test_loss}\nAccuracy in gen test data: {test_acc}')

    model.save(f'best_model_flips:{flips_}_angles:{angles_}_p1f')

    tf.keras.backend.clear_session()
    


