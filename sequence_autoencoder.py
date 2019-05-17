#!/usr/bin/env python

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, help="Learning rate")
args = parser.parse_args()
from functions.chase import *

ds_all, ds_all_centered, datasets, datasets_centered, ds_counts = load_data()


seq_len      = 128
latent_dim   = 256
n_layers     = 3 #2
n_units      = 384 #256
use_dense    = True
kl_weight    = 1e-4 #1e-2
resolution   = 3e-1 #1e-2
lr           = args.lr # range from 1e-5 to 1e-2
do_rotations = True
extrap_len   = seq_len//2
#do_shift     = False
#do_inplace   = False

batch_size = 128 #32
epochs = 100

print("Learning rate = {}".format(lr))

encoder, decoder, autoencoder, mk_continuizer = mk_seq_ae(ds_all, seq_len=seq_len, latent_dim=latent_dim,
                                   n_units=n_units, n_layers=n_layers,
                                  use_dense=use_dense, kl_weight=kl_weight,
                                  resolution=resolution, do_rotations=do_rotations, extrap_len=extrap_len)
continuizer = mk_continuizer(1)
encoder.summary()
decoder.summary()
autoencoder.summary()

K.set_value(autoencoder.optimizer.lr, lr)
K.set_value(autoencoder.hp_kl_weight, kl_weight)


loss_history = []


# encoder = model_from_json(open('models/vae_lstm_enc_model.json').read(), {'RotationLayer': RotationLayer})
# decoder = model_from_json(open('models/vae_lstm_dec_model.json').read(), {'RotationLayer': RotationLayer})
# auto = model_from_json(open('models/vae_lstm_auto_model.json').read(), {'RotationLayer': RotationLayer})

encoder.load_weights('weights/checkpoint_weights_vae_lstm_continued_lr_0.001_encoder.h5')
decoder.load_weights('weights/checkpoint_weights_vae_lstm_continued_lr_0.001_decoder.h5')
autoencoder.load_weights('weights/checkpoint_weights_vae_lstm_continued_lr_0.001_autoencoder.h5')

autoencoder.summary()

nstep = sum([c-seq_len for c in ds_counts])//batch_size


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

checkpoint_callback = CustomCheckpoint(filepath="weights/checkpoint_weights_vae_lstm_continued2_lr_"+str(lr),encoder=encoder, decoder=decoder, autoencoder=autoencoder)

try:
    autoencoder.fit_generator(gen_batches_safe(ds_all_centered, ds_counts, batch_size, seq_len), steps_per_epoch=nstep, epochs=epochs, verbose=1, callbacks=[checkpoint_callback])
    
except KeyboardInterrupt:
    print("Interrupted.")

print("Updating loss history")
loss_history.extend(autoencoder.history.history['loss'])
