from typing import Tuple
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import BatchNormalization, Dense, Dropout, Flatten

def mlp(input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        layer_size: int = 128,
        dropout_amount: float = 0.223,
        num_layers: int = 3) -> Model:
    # Simple multi-layer perceptron: just fully-connected layers with softmax predictions; creates num_layers layers.
    
    num_classes = output_shape[0]
    model = Sequential()
    model.add(Flatten(input_shape = input_shape))
    # for _ in range(num_layers):
    model.add(BatchNormalization())
    model.add(Dense(layer_size, activation = 'selu'))
    model.add(BatchNormalization())
    model.add(Dense(layer_size, activation = 'selu'))
    # model.add(Dropout(dropout_amount))
    model.add(BatchNormalization())
    model.add(Dense(layer_size, activation = 'selu'))
    model.add(Dropout(dropout_amount))
    model.add(Dense(num_classes, activation = 'softmax'))
    
    return model


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

