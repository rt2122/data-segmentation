import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from tensorflow.keras import Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy, sparse_categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from matplotlib.pyplot import figure
from skimage.io import imshow
from tensorflow.keras.layers import BatchNormalization

def unet(input_size = (2048,2048,4), filters=64, blocks=4):
    encoder = []
    inputs = Input(input_size)
    prev = inputs
    for i in range(blocks):
        cur = Conv2D(filters=filters, kernel_size=3, padding = 'same', kernel_initializer = 'he_normal')(prev)
        cur = BatchNormalization()(cur)
        cur = Activation(relu)(cur)

        cur = Conv2D(filters=filters, kernel_size=3, padding = 'same', kernel_initializer = 'he_normal')(cur)
        cur = BatchNormalization()(cur)
        cur = Activation(relu)(cur)

        encoder.append(cur)

        cur = MaxPooling2D()(cur)

        filters *= 2
        prev = cur
    print(prev.shape)
    for i in range(blocks - 1, -1, -1):
        cur = UpSampling2D()(prev)
        cur = Conv2D(filters=filters, kernel_size=3, padding='same')(cur)
        cur = Activation(relu)(cur)
        cur = concatenate([cur, encoder[i]], axis=3)

        cur = Conv2D(filters=filters, kernel_size=3, padding='same')(cur)
        cur = Activation(relu)(cur)
        cur = Conv2D(filters=filters, kernel_size=3, padding='same')(cur)
        cur = Activation(relu)(cur)

        prev = cur
        filters //= 2

    prev = Conv2D(1, kernel_size=1)(prev)
    prev = Activation(sigmoid)(prev)
    print(prev.shape)
    
    model = Model(inputs=inputs, outputs=prev)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model
