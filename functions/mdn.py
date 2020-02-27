# adapted from https://raw.githubusercontent.com/omimo/Keras-MDN/master/kmdn/mdn.py
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.layers import Dense, Input, merge, concatenate, Dense, LSTM, CuDNNLSTM
from keras.engine.topology import Layer
from keras import optimizers
from keras import backend as K
import tensorflow_probability as tfp
import tensorflow as tf
from keras.models import load_model
import numpy as np
from keras.callbacks import TerminateOnNaN, ModelCheckpoint

# check tfp version, as tfp causes cryptic error if out of date
assert float(tfp.__version__.split('.')[1]) >= 5

class MDN(Layer):
  '''Mixture Density Network with unigaussian kernel'''
  def __init__(self, n_mixes, output_dim, **kwargs):
    self.n_mixes = n_mixes
    self.output_dim = output_dim

    with tf.name_scope('MDN'):
      self.mdn_mus    = Dense(self.n_mixes * self.output_dim, name='mdn_mus')
      self.mdn_sigmas = Dense(self.n_mixes, activation=K.exp, name='mdn_sigmas')
      self.mdn_alphas = Dense(self.n_mixes, activation=K.softmax, name='mdn_alphas')
    super(MDN, self).__init__(**kwargs)

  def build(self, input_shape):
    self.mdn_mus.build(input_shape)
    self.mdn_sigmas.build(input_shape)
    self.mdn_alphas.build(input_shape)
    self.trainable_weights = self.mdn_mus.trainable_weights + \
      self.mdn_sigmas.trainable_weights + \
      self.mdn_alphas.trainable_weights
    self.non_trainable_weights = self.mdn_mus.non_trainable_weights + \
      self.mdn_sigmas.non_trainable_weights + \
      self.mdn_alphas.non_trainable_weights
    self.built = True

  def call(self, x, mask=None):
    with tf.name_scope('MDN'):
      mdn_out = concatenate([
        self.mdn_mus(x),
        self.mdn_sigmas(x),
        self.mdn_alphas(x)
      ], name='mdn_outputs')
    return mdn_out

  def get_output_shape_for(self, input_shape):
    return (input_shape[0], self.output_dim)

  def get_config(self):
    config = {
      'output_dim': self.output_dim,
      'n_mixes': self.n_mixes,
    }
    base_config = super(MDN, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_loss_func(self):
    def unigaussian_loss(y_true, y_pred):
      mix = tf.range(start = 0, limit = self.n_mixes)
      out_mu, out_sigma, out_alphas = tf.split(y_pred, num_or_size_splits=[
        self.n_mixes * self.output_dim,
        self.n_mixes,
        self.n_mixes,
      ], axis=-1, name='mdn_coef_split')

      def loss_i(i):
        batch_size = tf.shape(out_sigma)[0]
        sigma_i = tf.slice(out_sigma, [0, i], [batch_size, 1], name='mdn_sigma_slice')
        alpha_i = tf.slice(out_alphas, [0, i], [batch_size, 1], name='mdn_alpha_slice')
        mu_i = tf.slice(out_mu, [0, i * self.output_dim], [batch_size, self.output_dim], name='mdn_mu_slice')
        dist = tfp.distributions.Normal(loc=mu_i, scale=sigma_i)
        loss = dist.prob(y_true) # find the pdf around each value in y_true
        loss = alpha_i * loss
        return loss

      result = tf.map_fn(lambda  m: loss_i(m), mix, dtype=tf.float32, name='mix_map_fn')
      result = tf.reduce_sum(result, axis=0, keepdims=False)
      result = -tf.math.log(result)
      result = tf.reduce_mean(result)
      return result

    with tf.name_scope('MDNLayer'):
      return unigaussian_loss

class LSTM_MDN:
  def __init__(self, n_verts=15, n_dims=3, n_mixes=2, look_back=1, cells=[32,32,32,32], use_mdn=True, use_pca=False, lr=1e-3):
    self.n_verts = n_verts
    self.n_dims = n_dims
    self.n_mixes = n_mixes
    self.look_back = look_back
    self.cells = cells
    self.use_mdn = use_mdn
    self.use_pca = use_pca
    self.lr = lr
    self.LSTM = CuDNNLSTM if len(tf.config.list_physical_devices('GPU')) > 0 else LSTM
    self.model = self.build_model()
    if use_mdn:
      self.model.compile(loss=MDN(n_mixes, n_verts*n_dims).get_loss_func(), optimizer=optimizers.Adam(lr=lr), metrics=['accuracy'])
    else:
      self.model.compile(loss='mse', optimizer=optimizers.Adam(lr=lr), metrics=['accuracy'])
    
  def build_model(self):
    i = Input((self.look_back, self.n_verts*self.n_dims))
    h = self.LSTM(self.cells[0], return_sequences=True)(i) # return sequences, stateful
    h = self.LSTM(self.cells[1], return_sequences=True)(h)
    h = self.LSTM(self.cells[2])(h)
    h = Dense(self.cells[3])(h)
    if self.use_mdn:
      o = MDN(self.n_mixes, self.n_verts*self.n_dims)(h)
    else:
      o = Dense(self.n_verts*self.n_dims)(h)
    return Model(inputs=[i], outputs=[o])
  
  def prepare_inputs(self, X, look_back=2):
    '''
    Prepare inputs in shape expected by LSTM
    @returns:
      numpy.ndarray train_X: has shape: n_samples, lookback, verts * dims
      numpy.ndarray train_Y: has shape: n_samples, verts * dims
    '''
    # prepare data for the LSTM_MDN
    X = X.swapaxes(0, 1) # reshape to time, vert, dim
    n_time, n_verts, n_dims = X.shape
    
    # validate shape attributes
    if n_verts != self.n_verts: raise Exception(' ! got', n_verts, 'vertices, expected', self.n_verts)
    if n_dims != self.n_dims: raise Exception(' ! got', n_dims, 'dims, expected', self.n_dims)
    if look_back != self.look_back: raise Exception(' ! got', look_back, 'for look_back, expected', self.look_back)
    
    # center each frame along the x and y axes to simplify training
    X[:,:,0] = X[:,:,0] - np.mean(X[:,:,0], axis=0) + 0.5*np.ones(self.n_verts)
    X[:,:,1] = X[:,:,1] - np.mean(X[:,:,1], axis=0) + 0.5*np.ones(self.n_verts)
    
    # lstm expects data in shape [samples_in_batch, timestamps, values]
    train_X = []
    train_Y = []
    for i in range(look_back, n_time, 1):
      train_X.append( X[i-look_back:i,:,:].reshape(look_back, n_verts * n_dims) ) # look_back, verts * dims
      train_Y.append( X[i,:,:].reshape(n_verts * n_dims) ) # verts * dims
    train_X = np.array(train_X) # n_samples, lookback, verts * dims
    train_Y = np.array(train_Y) # n_samples, verts * dims
    return [train_X, train_Y]
  
  def predict_positions(self, input_X):
    '''
    Predict the output for a series of input frames. Each prediction has shape (1, y), where y contains:
      mus = y[:n_mixes*n_verts*n_dims]
      sigs = y[n_mixes*n_verts*n_dims:-n_mixes]
      alphas = softmax(y[-n_mixes:])
    @param numpy.ndarray input_X: has shape: n_samples, look_back, n_verts * n_dims
    @returns:
      numpy.ndarray X: has shape: verts, time, dims
    '''
    predictions = []
    for i in range(input_X.shape[0]):
      y = self.model.predict( train_X[i:i+1] ).squeeze()
      mus = y[:n_mixes*n_verts*n_dims]
      sigs = y[n_mixes*n_verts*n_dims:-n_mixes]
      alphas = self.softmax(y[-n_mixes:])

      # find the most likely distribution then pull out the mus that correspond to that selected index
      alpha_idx = np.argmax(alphas) # 0
      alpha_idx = 0
      predictions.append( mus[alpha_idx*self.n_verts*self.n_dims:(alpha_idx+1)*self.n_verts*self.n_dims] )
    predictions = np.array(predictions).reshape(train_X.shape[0], self.n_verts, self.n_dims).swapaxes(0, 1)
    return predictions # shape = n_verts, n_time, n_dims
    
def softmax(x):
    ''''Compute softmax values for vector `x`'''
    r = np.exp(x - np.max(x))
    return r / r.sum()

