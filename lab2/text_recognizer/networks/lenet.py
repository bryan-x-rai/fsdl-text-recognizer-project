from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Input, Lambda, MaxPooling2D
from tensorflow.keras.models import Sequential, Model


def lenet(input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> Model:
    
    num_classes = output_shape[0]

    model = Sequential()
    
    # comment in either compact or conventional:
    # conventional:
    print(f'len(input_shape) is {len(input_shape)}') # this duplicates output from rexp.py btw
    if len(input_shape) < 3:
        model.add(Lambda(lambda x: tf.expand_dims(x, -1), input_shape = input_shape))
        input_shape = (input_shape[0], input_shape[1], 1)
    print('boo 1')
    # model.add(Conv2D(32, (3, 3), activation = 'selu'))
    model.add(Conv2D(32, (3, 3), activation = 'selu', input_shape = input_shape))
    model.add(Conv2D(64, (3, 3), activation = 'selu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(128, activation = 'selu'))
    model.add(Dense(num_classes, activation = 'softmax'))
    print('boo 2')
    
    return model
    
    # comment in either compact or conventional:
    # compact:
    '''
    # we know len(input_shape) in this context, so let's dispense with the if statement (do not use if context inapplicable):
    print(f'len(input_shape) is {len(input_shape)}') # this duplicates output from rexp.py btw
    model.add(Lambda(lambda x: tf.expand_dims(x, -1), input_shape = input_shape))
    model.add(Conv2D(32, (3, 3), activation = 'selu'))
    model.add(Conv2D(64, (3, 3), activation = 'selu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(128, activation = 'selu'))
    model.add(Dense(num_classes, activation = 'softmax'))
    
    return model
    '''

'''
example accs:
10906/10906 [==============================] - 162s 15ms/step - loss: 0.5523 - acc: 0.8171 - val_loss: 0.4675 - val_acc: 0.8399
Training took 162.104841 s
GPU utilization: 71.68 +- 5.36
Test evaluation: 0.8398940880135485

10906/10906 [==============================] - 161s 15ms/step - loss: 0.5525 - acc: 0.8175 - val_loss: 0.4745 - val_acc: 0.8416
Training took 161.639176 s
GPU utilization: 72.45 +- 1.55
Test evaluation: 0.841553261177927
~/fsdl-text-recognizer-project/lab2 (master)
'''
    
'''
    #conventional:
    print(f'len(input_shape) is {len(input_shape)}') # this duplicates output from rexp.py btw
    if len(input_shape) < 3:
        model.add(Lambda(lambda x: tf.expand_dims(x, -1), input_shape = input_shape))
        input_shape = (input_shape[0], input_shape[1], 1)
    model.add(Conv2D(32, (3, 3), activation = 'selu'))
    model.add(Conv2D(64, (3, 3), activation = 'selu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Dropout(0.1))
    model.add(Flatten())
    model.add(Dense(128, activation = 'selu'))
    model.add(Dense(num_classes, activation = 'softmax'))
    
    return model
    '''

'''

    ##### Your code below (Lab 2)
    
    # "Your code"  ;)
    
    model = Sequential()
    # model.add(Flatten(input_shape = input_shape))
    # model.add(Conv2D(32, (3, 3), activation = 'selu', input_shape = (28, 28, 1)))
    
    # if len(input_shape) < 3:
    
    model.add(Lambda(lambda x: tf.expand_dims(x, -1), input_shape = input_shape))
    # input_shape = (input_shape[0], input_shape[1], 1)
    
    # model.add(Conv2D(32, (3, 3), activation = 'selu', input_shape = input_shape))
    
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