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

def p_model(parameters_object):     
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


print('----------------------\n\n\nbeginning grid testing\n\n\n----------------------')

parameters_object = parameters_nn()
funcs = ['max33', 'max55', 'avg33', 'avg55']
fun = 0
for parameter_name in parameters_to_test.keys():
    for parameter_value in parameters_to_test[parameter_name]:
        setattr(parameters_object, parameter_name, parameter_value)
        print(f'set: {parameter_name} = {parameter_value}')
        model = p_model(parameters_object)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        my_callbacks = [History(), EarlyStopping(patience=10, min_delta=0.01, monitor='val_accuracy', mode="max", restore_best_weights = True)]

        # Fit the model
        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=300, callbacks=my_callbacks)

        # save history to dataframe and save to csv
        history_df = pd.DataFrame(history.history)
        if parameter_name == 'pooling_function':
            parameter_value = funcs[fun]
            fun += 1
        history_df.to_csv(f'history_{parameter_name}_{parameter_value}.csv')
        

        # Test evaluation of the model
        test_loss, test_acc = model.evaluate(X_test, y_test)

        print(f'Loss in test data: {test_loss}\nAccuracy in test data: {test_acc}')
        
        # save test results to csv
        with open(f'test_results_{parameter_name}_{parameter_value}.csv', 'w') as f:
            f.write(f'Loss in test data: {test_loss}\nAccuracy in test data: {test_acc}')

        # save model
        model.save(f'model_{parameter_name}_{parameter_value}.h5')


        tf.keras.backend.clear_session()

    parameters_object = parameters_nn()