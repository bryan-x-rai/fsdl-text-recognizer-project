from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, Lambda, MaxPooling2D
from tensorflow.keras.models import Sequential, Model

def lenet(input_shape: Tuple[int, ...],
          output_shape: Tuple[int, ...]
         ) -> Model:
    num_classes = output_shape[0]

    model = Sequential()
    model.add(Lambda(lambda x: tf.expand_dims(x, -1), input_shape = input_shape))
    model.add(Conv2D(32, (3, 3), activation = 'selu', input_shape = input_shape))
    model.add(Conv2D(64, (3, 3), activation = 'selu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation = 'selu'))
    model.add(Dense(num_classes, activation = 'softmax'))
    
    return model

'''

    ##### Your code below (Lab 2)
    
    # "Your code"  ;)
    
    model = Sequential()
    # model.add(Flatten(input_shape = input_shape))
    # model.add(Conv2D(32, (3, 3), activation = 'selu', input_shape = (28, 28, 1)))
    
    # if len(input_shape) < 3:
    
    model.add(Lambda(lambda x: tf.expand_dims(x, -1), input_shape = input_shape))
    # input_shape = (input_shape[0], input_shape[1], 1)
    
    model.add(Conv2D(32, (3, 3), activation = 'selu', input_shape = input_shape))
    # model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation = 'selu'))
    model.add(MaxPooling2D((2, 2)))
    # model.add(Conv2D(64, (3, 3), activation = 'selu'))
    model.add(Flatten())
    model.add(Dense(128, activation = 'selu'))
    model.add(Dense(num_classes, activation = 'softmax'))
    
    ##### Your code above (Lab 2)

    return model

'''