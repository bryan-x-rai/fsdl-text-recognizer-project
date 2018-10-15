from typing import Tuple
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import AlphaDropout, BatchNormalization, Dense, Dropout, Flatten, Input, Lambda, MaxPooling2D, Conv2D

def mlp(input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        layer_size: int = 128,
        # dropout_amount_1: float = 0,
        dropout_amount_2: float = 0.04,
        dropout_amount_3: float = 0.08,
        num_layers: int = 3) -> Model:
    # Simple multi-layer perceptron: just fully-connected layers with softmax predictions; creates num_layers layers.
    
    num_classes = output_shape[0]
    model = Sequential()
    
    # model.add(Flatten(input_shape = input_shape))
    # new:
    if len(input_shape) < 3:
        model.add(Lambda(lambda x: tf.expand_dims(x, -1), input_shape = input_shape))
        input_shape = (input_shape[0], input_shape[1], 1)
    model.add(Conv2D(32, kernel_size = (3, 3), kernel_initializer = 'lecun_normal', strides = (1, 1), activation = 'selu', input_shape = input_shape))
    model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))
    model.add(Conv2D(64, (5, 5), activation = 'selu'))
    model.add(Conv2D(64, (5, 5), activation = 'selu'))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Flatten())
    model.add(Dense(layer_size, activation = 'selu'))
    # model.add(Dropout(dropout_amount_1))
    model.add(BatchNormalization())
    model.add(Dense(layer_size, activation = 'selu'))
    model.add(AlphaDropout(dropout_amount_2))
    model.add(BatchNormalization())
    model.add(Dense(layer_size, activation = 'selu'))
    model.add(AlphaDropout(dropout_amount_3))
    model.add(BatchNormalization())
    model.add(Dense(num_classes, activation = 'softmax'))
    
    return model

# first runs with conv:
# Test evaluation: 0.8640337680424336
# Test evaluation: 0.8658046989847236
'''
example run achieves #9 of 123 on the Leaderboard, without applying a convnet to Lab 1:
Leaderboard

Search
Assignment Leaderboard
 RANK
 SUBMISSION NAME
 ACCURACY
1	Al A Latif	0.8754760451501423
2	Kam	0.8728454389931484
3	B_145	0.872553149420149
4	Ignacio	0.8686502239453935
5	arai	0.8672489533454261
6	hoktay	0.865022394539343
7	Brenda Ng	0.8645839601798441
8	kishb	0.8616352741934097
9	bryan-x-rai	0.8490668225544389
10	skrsna	0.8489894517851155
'''

'''
def mlp(input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        layer_size: int=128,
        dropout_amount: float=0.12,
        num_layers: int=4) -> Model:
    """
    Simple multi-layer perceptron: just fully-connected layers with dropout between them, with softmax predictions.
    Creates num_layers layers.
    """
    num_classes = output_shape[0]

    model = Sequential()
    
    # Don't forget to pass input_shape to the first layer of the model # Your code below (Lab 1)

    '''    '''
    model.add(Flatten(input_shape=input_shape))
    for _ in range(num_layers):
        model.add(Dense(layer_size, activation='relu'))
        model.add(Dropout(dropout_amount))
    model.add(Dense(num_classes, activation='softmax'))
    
    '''    '''

    model.add(Flatten(input_shape=input_shape))
    for _ in range(num_layers - 1):
        model.add(Dense(layer_size, activation='relu'))
        model.add(Dropout(dropout_amount))
    model.add(Dense(layer_size, activation='relu'))
    model.add(Dropout(dropout_amount / 2))
    model.add(Dense(num_classes, activation='softmax'))

    '''    '''
    
    model.add(Flatten(input_shape=input_shape))
    for _ in range(num_layers):
        model.add(Dense(layer_size, activation='relu'))
        model.add(Dropout(dropout_amount))
    model.add(Dense(num_classes, activation='softmax'))
    
    
    # Your code above (Lab 1)

    return model
'''

