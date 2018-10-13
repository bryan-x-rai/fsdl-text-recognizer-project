from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import AlphaDropout, BatchNormalization, Conv2D, Dense, Dropout, Flatten, Input, Lambda, MaxPooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import initializers


def lenet(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> Model:
    num_classes = output_shape[0]

    ##### Your code below (Lab 2)
    
    
    model = Sequential()
    
    print(f'len(input_shape) is {len(input_shape)}') # this duplicates output from rexp.py btw
    if len(input_shape) < 3:
        model.add(Lambda(lambda x: tf.expand_dims(x, -1), input_shape = input_shape))
        input_shape = (input_shape[0], input_shape[1], 1)
    
    # selu option:
    
    model.add(Conv2D(32, kernel_size = (3, 3), kernel_initializer = 'lecun_normal', activation = 'selu', input_shape = input_shape))
    model.add(Conv2D(64, (3, 3), activation = 'selu'))
    '''
    # relu option:
    model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = input_shape))
    model.add(Conv2D(64, (3, 3), activation = 'relu'))
    '''
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(AlphaDropout(0.032))
    # model.add(Dropout(0.08))
    
    # added conv2d layer:
    # model.add(Conv2D(32, (3, 3), activation = 'selu'))
    # model.add(AlphaDropout(0.03))
    
    # model.add(MaxPooling2D(pool_size = (2, 2)))
    # model.add(AlphaDropout(0.07))
    
    model.add(Flatten())
    model.add(Dense(128, activation = 'selu'))
    model.add(AlphaDropout(0.1))
    model.add(Dense(num_classes, activation = 'softmax'))

    
    ''' 
    model = Sequential()
    if len(input_shape) < 3:
        model.add(Lambda(lambda x: tf.expand_dims(x, -1), input_shape=input_shape))
        input_shape = (input_shape[0], input_shape[1], 1)
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    '''
    
    ##### Your code above (Lab 2)

    return model

