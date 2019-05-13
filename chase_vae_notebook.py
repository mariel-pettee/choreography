#!/usr/bin/env python

import os
import tensorflow as tf

# For GPU config:
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)

import keras
import keras.backend as K
import keras.layers as layers
from keras.models import Model, load_model
import numpy as np
# from IPython.display import HTML
import matplotlib.pyplot as plt
from functions.chase import load_data, animate_stick


ds_all, ds_all_centered, datasets, datasets_centered, ds_counts = load_data()
print(ds_all.shape)

# generator function to sample batches of contiguous sequences from a given dataset
# This one will safely avoid creating sequences that span the boundary between
# two different datasets
def gen_batches_safe(data, ds_counts, batch_size, seq_len, center=False):
    ds_offsets = np.zeros_like(ds_counts)
    ds_offsets[1:] = np.cumsum(ds_counts[:-1])
    
    batch_idxs = []
    for ds_len,ds_offset in zip(ds_counts, ds_offsets):
        ds_batch_idxs = np.arange(ds_len-seq_len).repeat(seq_len).reshape(-1,seq_len) + np.arange(seq_len)
        batch_idxs.append(ds_batch_idxs + ds_offset)
    
    batch_idxs = np.concatenate(batch_idxs)
    
    nbatch = batch_idxs.shape[0]//batch_size
    
    while True:
        np.random.shuffle(batch_idxs)
        for ibatch in range(nbatch):
            batch = data[batch_idxs[ibatch*batch_size:(ibatch+1)*batch_size]]
            yield batch, None
            #yield batch, batch

def sample_z(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


from keras import initializers
class BiasLayer(layers.Layer):

    def __init__(self, bias_init='zeros', bias_std=0.01, **kwargs):
        self.bias_init = bias_init
        self.bias_std = bias_std
        super(BiasLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        if self.bias_init == 'zeros':
            init = 'zeros'
        elif self.bias_init == 'normal':
            init = initializers.RandomNormal(stddev=self.bias_std)
        
        self.bias = self.add_weight(name='bias', 
                                      shape=(input_shape[1],),
                                      initializer=init,
                                      trainable=True)
        super(BiasLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        #return K.dot(x, self.kernel)
        return x + self.bias

    def compute_output_shape(self, input_shape):
        #return (input_shape[0], self.output_dim)
        return input_shape


class RotationLayer(layers.Layer):

    def __init__(self, theta, dim=3, learning_phase_only=True, **kwargs):
        self.theta = theta
        self.vec_dim = dim
        
        if dim == 3:
            self.R = K.constant([[1,0,0],[0,1,0],[0,0,0]])*tf.cos(theta)                        + K.constant([[0,-1,0],[1,0,0],[0,0,0]])*tf.sin(theta)                        + K.constant([[0,0,0],[0,0,0],[0,0,1]])
        elif dim == 2:
            self.R = K.constant([[1,0],[0,1]])*tf.cos(theta)                        + K.constant([[0,-1],[1,0]])*tf.sin(theta)
        
        self.uses_learning_phase = learning_phase_only
        
        super(RotationLayer, self).__init__(**kwargs)

    def call(self, x, training=None):
        if self.uses_learning_phase:
            return K.in_train_phase(K.dot(x, self.R), x, training=training)
        else:
            return K.dot(x, self.R)

def mk_seq_ae(X, seq_len, latent_dim=32, n_layers=2, n_units=32, use_dense=True, kl_weight=0, resolution=3e-3, do_rotations=True, extrap_len=8):
    K.clear_session()
    
    n_vtx = X.shape[1]
    
    encoder_input = layers.Input((seq_len, n_vtx, 3))
    H = encoder_input
    
    #H = layers.Flatten()(H)
    H = layers.Reshape((seq_len, n_vtx*3))(H)
    
    for i in range(n_layers-1):
        H = layers.LSTM(n_units, return_sequences=True)(H)
    
    if use_dense:
        H = layers.LSTM(n_units, return_sequences=False)(H)

        z_mean = layers.Dense(latent_dim, name='z_mean')(H)
        z_log_var = layers.Dense(latent_dim, name='z_log_var')(H)
    else:
        H = layers.LSTM(n_units, return_sequences=True)(H)
        
        H = layers.LSTM(2*latent_dim, return_sequences=False)(H)

        z_mean = layers.Lambda(lambda x: x[:,:latent_dim], name='z_mean')(H)
        z_log_var = layers.Lambda(lambda x: x[:,latent_dim:], name='z_log_var')(H)
        
    z_sample = layers.Lambda(sample_z, output_shape=(latent_dim,), name='z_sample')([z_mean, z_log_var])
    
    encoder_output = [z_sample, z_mean, z_log_var]
    
    encoder = Model(encoder_input, encoder_output)
    
    
    decoder_input = layers.Input((latent_dim,))
    H = decoder_input
    
    if use_dense:
        H = layers.Dense(n_units, activation='relu')(H)

    H = layers.RepeatVector(seq_len)(H)

    for i in range(n_layers-1):
        H = layers.LSTM(n_units, return_sequences=True)(H)

    H = layers.LSTM(n_vtx*3, return_sequences=True)(H)

    H = layers.Reshape((seq_len, n_vtx, 3))(H)
    decoder_output = H

    decoder = Model(decoder_input, decoder_output)
    
    
    autoencoder_input = layers.Input((seq_len,n_vtx,3))
    
    H = autoencoder_input
    
    if do_rotations:
        theta = K.cast(K.learning_phase(),'float')*K.random_uniform((1,), 0, 2*np.pi)
        H = RotationLayer(theta)(H)
    
    autoencoder_z, autoencoder_mean, autoencoder_log_var = encoder(H)
    H = decoder(autoencoder_z)
    
    if do_rotations:
        H = RotationLayer(-theta)(H)
    
    autoencoder_output = H

    autoencoder = Model(autoencoder_input, autoencoder_output)
    
    
    autoencoder.hp_resolution = K.variable(resolution)
    
    ae_loss = 0.5*K.mean(K.sum(K.square(autoencoder_input - autoencoder_output), axis=-1))
    autoencoder.add_loss(ae_loss/K.square(autoencoder.hp_resolution))

    if kl_weight:
        kl_loss = -0.5*K.mean(K.sum(1 + autoencoder_log_var - K.square(autoencoder_mean) - K.exp(autoencoder_log_var), axis=-1))

        autoencoder.hp_kl_weight = K.variable(kl_weight)
        autoencoder.add_loss(autoencoder.hp_kl_weight * kl_loss)

    autoencoder.compile(optimizer='adam')
    
    def calc_shift_residual(args):
        x0, x1 = args
        
        #return x0[:,seq_len//2:]-x1[:,:-seq_len//2]
        return x0[:,extrap_len:] - x1[:,:-extrap_len]
    shift_residual = layers.Lambda(calc_shift_residual, name='shift_residual')
    
    def mk_continuizer(nstep):
        continuizer_input = layers.Input((seq_len, n_vtx, 3))
        #continuizer_x0 = continuizer_input
        #continuizer_z0, continuizer_z0_mean, continuizer_z0_logvar = encoder(continuizer_x0)
        #continuizer_x0_clean = decoder(continuizer_z0_mean)
        _, continuizer_z0, _ = encoder(continuizer_input)
        continuizer_x0 = decoder(continuizer_z0)
        
        continuizer_z1 = BiasLayer()(continuizer_z0)
        continuizer_x1 = decoder(continuizer_z1)
        
        continuizer_output = [continuizer_x1, continuizer_z1]
        continuizer = Model(continuizer_input, continuizer_output)
        
        curve = np.exp(-np.arange(seq_len-1-extrap_len,-1,-1)/((seq_len-extrap_len)/8))
        curve /= curve.sum()
        curve = K.constant(curve.reshape((1,seq_len-extrap_len,1,1)))
        
        continuizer.add_loss(0.5/K.square(autoencoder.hp_resolution)*                             K.mean(K.sum(curve*K.square(shift_residual([continuizer_x0, continuizer_x1])), axis=-1)))
        
        decoder.trainable = False
        encoder.trainable = False
        continuizer.compile(optimizer='adam')
        #continuizer.compile(optimizer='rmsprop')
        decoder.trainable = True
        encoder.trainable = True
        return continuizer
    
    return encoder, decoder, autoencoder, mk_continuizer

seq_len      = 128
latent_dim   = 256
n_layers     = 3 #2
n_units      = 384 #256
use_dense    = True
kl_weight    = 1 #1e-2
resolution   = 3e-1 #1e-2
lr           = 3e-4
do_rotations = True
extrap_len   = seq_len//2
#do_shift     = False
#do_inplace   = False

encoder, decoder, autoencoder, mk_continuizer = mk_seq_ae(ds_all, seq_len=seq_len, latent_dim=latent_dim,
                                   n_units=n_units, n_layers=n_layers,
                                  use_dense=use_dense, kl_weight=kl_weight,
                                  resolution=resolution, do_rotations=do_rotations, extrap_len=extrap_len)
continuizer = mk_continuizer(1)
encoder.summary()
decoder.summary()
autoencoder.summary()

K.set_value(autoencoder.optimizer.lr, lr)

loss_history = []


# Train:

batch_size = 128 #32
epochs = 512

nstep = sum([c-seq_len for c in ds_counts])//batch_size

K.set_value(autoencoder.optimizer.lr, 1e-4)
K.set_value(autoencoder.hp_kl_weight, 2e-4)



class CustomCheckpoint(keras.callbacks.Callback):
    def __init__(self, filepath, encoder, decoder, autoencoder):
        self.monitor = 'loss'
        self.monitor_op = np.less
        self.best = np.Inf
        self.filepath = filepath
        self.encoder = encoder
        self.decoder = decoder
        self.autoencoder = autoencoder
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        if self.monitor_op(current, self.best):
            self.best = current
            self.encoder.save_weights(self.filepath+"_encoder.h5", overwrite=True)
            self.decoder.save_weights(self.filepath+"_decoder.h5", overwrite=True)
            self.autoencoder.save_weights(self.filepath+"_autoencoder.h5", overwrite=True)

checkpoint_callback = CustomCheckpoint(filepath="checkpoint_weights",encoder=encoder, decoder=decoder, autoencoder=autoencoder)

try:
    autoencoder.fit_generator(gen_batches_safe(ds_all_centered, ds_counts, batch_size, seq_len), steps_per_epoch=nstep, epochs=epochs, verbose=1, callbacks=[checkpoint_callback])
    
except KeyboardInterrupt:
    print("Interrupted.")

print("Updating loss history")
loss_history.extend(autoencoder.history.history['loss'])


# save_weights = True
# load_weights = False

# if save_weights:
#     print("Saving weights...")
#     encoder.save_weights('test_LSTM_enc_weights.h5')
#     decoder.save_weights('test_LSTM_dec_weights.h5')
#     autoencoder.save_weights('test_LSTM_autoencoder_weights.h5')
# if load_weights:
#     print("Loading weights...")
#     encoder.load_weights('seq_vae_enc_weights.h5')
#     decoder.load_weights('seq_vae_dec_weights.h5')
#     autoencoder.load_weights('seq_vae_autoencoder_weights.h5')


nskip = 2
xepochs = np.arange(len(loss_history))+1
plt.plot(xepochs[nskip:], loss_history[nskip:])
plt.savefig("loss.png")


