import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
sess = tf.Session()

from keras import regularizers
from keras import backend as K
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, TensorBoard


MAX_INT = np.iinfo(np.int32).max
data_format = 0


def dev_network_s(input_shape):
    '''
    network architecture with one hidden layer
    '''
    x_input = Input(shape=input_shape, name="input_layer")
    intermediate = Dense(20, activation='relu', 
                kernel_regularizer=regularizers.l2(0.01), name = 'hl1')(x_input)
    intermediate = Dense(1, activation='linear',  name = 'score')(intermediate)    
    return Model(x_input, intermediate)



def deviation_loss(y_true, y_pred):
    '''
    z-score-based deviation loss
    '''    
    confidence_margin = 5.     
    ## size=5000 is the setting of l in algdev_network_linearorithm 1 in the paper
    ref = K.variable(np.random.normal(loc = 0., scale= 1.0, size = 5000) , dtype='float32')
    dev = (y_pred - K.mean(ref)) / K.std(ref)
    inlier_loss = K.abs(dev) 
    outlier_loss = K.abs(K.maximum(confidence_margin - dev, 0.))
    return K.mean((1 - y_true) * inlier_loss + y_true * outlier_loss)


def deviation_network(input_shape):
    '''
    construct the deviation network-based detection model
    '''
    model = dev_network_s(input_shape)
    rms = RMSprop(clipnorm=1.) # RMSprop is the optimizer
    model.compile(loss=deviation_loss, optimizer=rms)
    return model