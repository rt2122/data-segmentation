import numpy as np 

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, concatenate, UpSampling2D
from tensorflow.keras import Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy, sparse_categorical_crossentropy
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

def iou(y_pred, y_true):
    iou_sum = 0
    for i in range(y_true.shape[-1]):
        inters = K.sum(y_pred[..., i] * y_true[..., i])
        union = K.sum((y_pred[..., i] + y_true[..., i])) - inters
        iou_sum += inters / union
    return iou_sum

def dice(y_pred, y_true, eps=0.1):
    dice_sum = 0
    for i in range(y_true.shape[-1]):
        inters = K.sum(y_pred[..., i] * y_true[..., i])
        union = K.sum((y_pred[..., i] + y_true[..., i])) - inters
        dice_sum += K.mean((2 * inters + eps) / (union + eps))
    return dice_sum


def unet(input_size = (512,512,1), filters=32, blocks=3, output_layers=50, weights=None):
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

    if not (weights is None):
        pt = Model(inputs=inputs, outputs=prev)
        pt.load_weights(weights)

    prev = Conv2D(output_layers, kernel_size=3)(prev)
    prev = Activation(sigmoid)(prev)
    
    model = Model(inputs=inputs, outputs=prev)
    model.compile(optimizer = Adam(lr = 1e-4), loss = categorical_crossentropy, metrics = ['accuracy', iou, dice])
    
    return model

