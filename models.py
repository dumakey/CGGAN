import os
import numpy as np
import tensorflow as tf


def swish(x, beta=1):
    return x * tf.keras.backend.sigmoid(beta * x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
        
def conv2D_block(X, num_channels, f, p, s, dropout, **kwargs):
    if kwargs:
        parameters = list(kwargs.values())[0]
        l2_reg = parameters['l2_reg']
        l1_reg = parameters['l1_reg']
        activation = parameters['activation']
    else:
        l2_reg = 0.0
        l1_reg = 0.0
        activation = 'relu'

    if p != 0:
        net = tf.keras.layers.ZeroPadding2D(p)(X)
    else:
        net = X
    net = tf.keras.layers.Conv2D(num_channels, kernel_size=f, strides=s, padding='valid',
                                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)
    net = tf.keras.layers.BatchNormalization()(net)

    if activation == 'leakyrelu':
        rate = 0.2
        net = tf.keras.layers.LeakyReLU(rate)(net)
    elif activation == 'swish':
        net = tf.keras.layers.Activation('swish')(net)
    elif activation == 'elu':
        net = tf.keras.layers.ELU(net)
    elif activation == 'tanh':
        net = tf.keras.activations.tanh(net)
    elif activation == 'sigmoid':
        net = tf.keras.activations.sigmoid(net)
    elif activation == 'linear':
        net = tf.keras.activations('linear')(net)
    else:
        net = tf.keras.layers.Activation('relu')(net)

    return net

def conv2Dtranspose_block(X, num_channels, f, s, dropout, **kwargs):
    if kwargs:
        parameters = list(kwargs.values())[0]
        l2_reg = parameters['l2_reg']
        l1_reg = parameters['l1_reg']
        activation = parameters['activation']
    else:
        l2_reg = 0.0
        l1_reg = 0.0
        activation = 'relu'

    net = tf.keras.layers.Conv2DTranspose(num_channels,kernel_size=f,strides=s,padding='same',
                                 kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(X)
    net = tf.keras.layers.BatchNormalization()(net)

    if activation == 'leakyrelu':
        rate = 0.2
        net = tf.keras.layers.LeakyReLU(rate)(net)
    elif activation == 'swish':
        net = tf.keras.layers.Activation('swish')(net)
    elif activation == 'elu':
        net = tf.keras.layers.ELU(net)
    elif activation == 'tanh':
        net = tf.keras.activations.tanh(net)
    elif activation == 'sigmoid':
        net = tf.keras.activations.sigmoid(net)    
    elif activation == 'linear':
        net = tf.keras.activations('linear')(net)
    else:
        net = tf.keras.layers.Activation('relu')(net)

    return net

def dense_layer(X, units, activation, dropout, l1_reg, l2_reg):

    net = tf.keras.layers.Dense(units=units,activation=None,kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(X)
    net = tf.keras.layers.BatchNormalization()(net)
    if activation == 'leakyrelu':
        rate = 0.1
        net = tf.keras.layers.LeakyReLU(rate)(net)
    elif activation == 'elu':
        net = tf.keras.layers.ELU()(net)
    else:
        net = tf.keras.layers.Activation(activation)(net)
    net = tf.keras.layers.Dropout(dropout)(net)

    return net
    
def loss_function():

    loss = tf.keras.losses.BinaryCrossentropy()
    #reg_loss = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)
        
    return loss

def optimizer(alpha):
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=alpha,beta_1=0.9,beta_2=0.999,amsgrad=False)

    return optimizer
    
def performance_metric(logits, labels):

    corrects = tf.equal(logits,labels)
    accuracy = tf.reduce_mean(tf.cast(corrects,tf.float32))

    return accuracy
        
def discriminator(X_input, activation, l2_reg, l1_reg, dropout):

    #input_shape = (image_shape[1],image_shape[0],1)
    #X_input = tf.keras.layers.Input(shape=input_shape)
    net = conv2D_block(X_input,num_channels=17,f=3,p=1,s=1,dropout=dropout,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'activation':activation})
    net = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)(net)
    net = conv2D_block(net,num_channels=39,f=3,p=1,s=1,dropout=dropout,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'activation':activation})
    net = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)(net)
    net = conv2D_block(net,num_channels=87,f=3,p=1,s=1,dropout=dropout,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'activation':activation})
    net = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)(net)
    net = tf.keras.layers.Flatten()(net)
    net = dense_layer(net,512,activation,dropout,l1_reg,l2_reg)
    net = tf.keras.layers.Dense(units=1,activation='sigmoid',kernel_regularizer=tf.keras.regularizers.L1L2(l1=l1_reg,l2=l2_reg))(net)

    return net

def generator(X_input, activation, l2_reg, l1_reg, dropout):

    dim_in = 10
    filt_in = 256
    #X_input = tf.keras.layers.Input(shape=(z_dim,))
    net = dense_layer(X_input,dim_in*dim_in*filt_in,activation,dropout,l1_reg,l2_reg)
    net = tf.keras.layers.Reshape((dim_in,dim_in,filt_in))(net)
    net = tf.keras.layers.UpSampling2D()(net)
    net = conv2Dtranspose_block(net,num_channels=int(filt_in/2),f=3,s=1,dropout=dropout,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'activation':activation})
    net = tf.keras.layers.UpSampling2D()(net)
    net = conv2Dtranspose_block(net,num_channels=int(filt_in/4),f=3,s=1,dropout=dropout,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'activation':activation})
    net = tf.keras.layers.UpSampling2D()(net)
    net = conv2Dtranspose_block(net,num_channels=int(filt_in/8),f=3,s=1,dropout=dropout,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'activation':activation})
    net = tf.keras.layers.UpSampling2D()(net)
    net = conv2Dtranspose_block(net,num_channels=1,f=3,s=1,dropout=dropout,kwargs={'l2_reg':l2_reg,'l1_reg':l1_reg,'activation':'sigmoid'})

    return net
