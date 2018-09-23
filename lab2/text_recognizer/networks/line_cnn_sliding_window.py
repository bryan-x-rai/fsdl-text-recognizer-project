import pathlib
from typing import Tuple

from boltons.cacheutils import cachedproperty
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D, Permute, Reshape, TimeDistributed, Lambda, ZeroPadding2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model as KerasModel

from text_recognizer.models.line_model import LineModel
from text_recognizer.networks.lenet import lenet
from text_recognizer.networks.misc import slide_window

print('boo 3')

def line_cnn_sliding_window(
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        window_width: float=16,
        window_stride: float=10) -> KerasModel:
    
    image_height, image_width = input_shape
    output_length, num_classes = output_shape
    
    print('boo 4')

    image_input = Input(shape=input_shape)
    # (image_height, image_width)

    image_reshaped = Reshape((image_height, image_width, 1))(image_input)
    # (image_height, image_width, 1)
    
    print('boo 5')

    image_patches = Lambda(
        slide_window,
        arguments={'window_width': window_width, 'window_stride': window_stride}
    )(image_reshaped)
    # (num_windows, image_height, window_width, 1)
    
    print('boo 6')

    # Make a LeNet and get rid of the last two layers (softmax and dropout)
    convnet = lenet((image_height, window_width, 1), (num_classes,))
    print('boo 7')
    convnet = KerasModel(inputs=convnet.inputs, outputs=convnet.layers[-2].output)
    print('boo 8')

    convnet_outputs = TimeDistributed(convnet)(image_patches)
    print('boo 9')
    # (num_windows, 128)

    # Now we have to get to (output_length, num_classes) shape. One way to do it is to do another sliding window with
    # width = floor(num_windows / output_length)
    # Note that this will likely produce too many items in the output sequence, so take only output_length,
    # and watch out that width is at least 2 (else we will only be able to predict on the first half of the line)

    ##### Your code below (Lab 2)
    
    convnet_o_pl = Lambda(lambda x: tf.expand_dims(x, -1))(convnet_outputs)
    print('foo 1')
    num_win = int((image_width - window_width) / window_stride) + 1
    print('foo 2')
    width = int(num_win / output_length)
    print('foo 3')
    convd_convnet_o = Conv2D(num_classes, (width, 128), (width, 1), activation = 'softmax')(convnet_o_pl)
    print('foo 4')
    sqzd_convd_convnet_o = Lambda(lambda x: tf.squeeze(x, 2))(convd_convnet_o)
    print('foo 5')
    softmax_o = Lambda(lambda x: x[:, :output_length, :])(sqzd_convd_convnet_o)
    print('foo 6')

    ##### Your code above (Lab 2)

    model = KerasModel(inputs=image_input, outputs=softmax_o)
    model.summary()
    return model

