
from keras import backend as K
import tensorflow as tf
import numpy as np
import os
import glob
from itertools import combinations

# use tensorflow backend
os.environ['KERAS_BACKEND'] = 'tensorflow'

# allow dynamic GPU memory allocation
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# identify available GPU's
gpus = K.tensorflow_backend._get_available_gpus()
print("# of GPUs found: {}".format(len(gpus)))

# set random seeds
tf.set_random_seed(1)
np.random.seed(1)

labels = ['ARIEL', 'C7', 'CLAV', 'LANK', 'LBHD', 'LBSH', 'LBWT', 'LELB', 'LFHD', 'LFRM', 'LFSH', 'LFWT', 'LHEL', 'LIEL', 'LIHAND', 'LIWR', 'LKNE', 'LKNI', 'LMT1', 'LMT5', 'LOHAND', 'LOWR', 'LSHN', 'LTHI', 'LTOE', 'LUPA', 'LabelingHips', 'MBWT', 'MFWT', 'RANK', 'RBHD', 'RBSH', 'RBWT', 'RELB', 'RFHD', 'RFRM', 'RFSH', 'RFWT', 'RHEL', 'RIEL', 'RIHAND', 'RIWR', 'RKNE', 'RKNI', 'RMT1', 'RMT5', 'ROHAND', 'ROWR', 'RSHN', 'RTHI', 'RTOE', 'RUPA', 'STRN', 'SolvingHips', 'T10']    

bad_labels = ['SolvingHips', 'LabelingHips']

edge_groups = [
  # head
  ['ARIEL', 'RFHD', 'RBHD', 'LFHD', 'LBHD'],
  ['ARIEL', 'CLAV'],
  # right arm
  ['CLAV', 'RELB'],
  ['RELB', 'RIWR'],
  # left arm
  ['CLAV', 'LELB'],
  ['LELB', 'LIWR'],
  # body
  ['CLAV', 'STRN'],
  # right leg
  ['STRN', 'RKNE'],
  ['RKNE', 'RMT5'],
  # left leg
  ['STRN', 'LKNE'],
  ['LKNE', 'LMT5'],
]

infile_glob = sorted(glob.glob('data/mariel_*'))


class Data:
  def __init__(self, file_glob, labels, bad_labels, edge_groups):
    # params
    self.files = file_glob
    self.labels = labels
    self.bad_labels = bad_labels
    self.edge_groups = edge_groups
    self.filtered_labels = [i for i in labels if i not in bad_labels]
    
    # parsed structures
    self.all_vertices = self.get_all_vertices()
    
    # fully composed datasets; each has `X`, `labels`, and `edges` attrs
    self.full = self.get_full_dataset()
    self.selected = self.get_selected_dataset()

  def get_all_vertices(self):
    '''Load each of the data files'''
    # load all vertices dataset
    file_data = []
    for i in self.files:
      file_data.append(np.swapaxes(np.load(i), 0, 1))
    X = np.swapaxes(np.vstack(file_data), 0, 1) # stack the time frames then make time 1st dim
    X = self.rotate(X, -np.pi/2, 'x') # rotate the matrix into proper orientation
    # filter out the bad vertices using the bad labels
    bad_indices = [idx for idx, i in enumerate(self.labels) if i in self.bad_labels]
    idx_mask = np.ones(X.shape[0], dtype=bool)
    idx_mask[bad_indices] = 0
    X = X[idx_mask]
    return X

  def get_full_dataset(self):
    '''Load the full 53 vertex dataset with edges from rigidity analysis'''
    X = self.scale(self.all_vertices)
    return Dataset(X, self.filtered_labels, self.get_rigid_edges(X))

  def get_rigid_edges(self, X):
    '''Get the edges discovered through rigidity analysis on X'''
    vdist_var = np.zeros((X.shape[0], X.shape[0]))
    for i in range(X.shape[0]):
      for j in range(i+1, X.shape[0]):
        vdist = np.sum((X[i]-X[j])**2, axis=-1)
        vdist_var[i,j] = vdist_var[j,i] = vdist.var(ddof=1)
    upper_triangle = np.triu_indices_from(vdist_var, k=1)
    vtx_pairs = sorted(zip(*upper_triangle), key=lambda p: vdist_var[p[0], p[1]])
    return vtx_pairs

  def get_selected_dataset(self):
    '''Load just the selected vertices and their associated edges'''
    selected_edges = self.get_selected_edges() # [[self.all_vertices_idx_i, self.all_vertices_idx_j]]
    selected_vertices = sorted(list(set([j for i in selected_edges for j in i]))) # flatten and sort
    # create a copy of X with fewer vertices; shape = (verts, time, dims)
    idx_mask = np.zeros(self.all_vertices.shape[0], dtype=bool)
    idx_mask[selected_vertices] = 1
    X = self.all_vertices[idx_mask]
    # create a copy of selected_edges that indexes into _X (aka X with reduced vertex count)
    d = {i: idx for idx, i in enumerate(selected_vertices)} # maps vert idx in self.all_vertices to idx in X
    edges = [ [d[i], d[j]] for i, j in selected_edges ]
    labels = [self.filtered_labels[d[i]] for i in d.keys()]
    return Dataset(X, labels, edges)
  
  def get_selected_edges(self):
    '''Get just the edges required to compose the frame for select vertices'''
    selected_edges = []
    label_to_idx = {i: idx for idx, i in enumerate(self.filtered_labels)}
    for g in self.edge_groups:
      for i, j in combinations(g, 2):   
        i = label_to_idx[i]
        j = label_to_idx[j]
        selected_edges.append([i, j])
    return selected_edges

  def scale(self, X):
    '''Scale all dimensions in X 0:1'''
    # center the data for visualization
    X -= np.amin(X, axis=(0, 1))
    X /= np.amax(X, axis=(0, 1))
    X[:,:,2] *= -1
    X[:,:,2] += 1
    return X
    
  def rotate(self, X, theta, axis='x'):
    '''Rotate multidimensional array `X` `theta` degrees around axis `axis`'''
    c, s = np.cos(theta), np.sin(theta)
    if axis == 'x': return np.dot(X, np.array([
      [1.,  0,  0],
      [0 ,  c, -s],
      [0 ,  s,  c]
    ]))
    elif axis == 'y': return np.dot(X, np.array([
      [c,  0,  -s],
      [0,  1,   0],
      [s,  0,   c]
    ]))
    elif axis == 'z': return np.dot(X, np.array([
      [c, -s,  0 ],
      [s,  c,  0 ],
      [0,  0,  1.],
    ]))

class Dataset:
  def __init__(self, X, labels, edges):
    self.X = X
    self.labels = labels
    self.edges = edges
    self.diffs = self.get_diffs()

  def get_diffs(self):
    '''Return vertices stored in relative change from last frame, not absolute coords'''
    return np.diff(self.X, axis=1)

  def from_diffs(self):
    '''Return vertices in absolute coords, not relative change from last frame'''
    initial = X[:,0:1,:]
    return np.cumsum( np.concatenate([initial, diffs], axis=1), axis=1 )


data = Data(infile_glob, labels, bad_labels, edge_groups)
X_train = data.selected.X[:,:,:]
print("Training dataset shape: {}".format(X_train.shape))
print("Preparing to train...")


# adapted from https://raw.githubusercontent.com/omimo/Keras-MDN/master/kmdn/mdn.py
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.layers import Dense, Input, merge, concatenate, Dense, LSTM, CuDNNLSTM
from keras.engine.topology import Layer
from keras import backend as K
import tensorflow_probability as tfp
import tensorflow as tf

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
    self.trainable_weights = self.mdn_mus.trainable_weights +       self.mdn_sigmas.trainable_weights +       self.mdn_alphas.trainable_weights
    self.non_trainable_weights = self.mdn_mus.non_trainable_weights +       self.mdn_sigmas.non_trainable_weights +       self.mdn_alphas.non_trainable_weights
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
      result = -tf.log(result)
      result = tf.reduce_mean(result)
      return result

    with tf.name_scope('MDNLayer'):
      return unigaussian_loss

class LSTM_MDN:
  def __init__(self, n_verts=15, n_dims=3, n_mixes=2, look_back=1, cells=[32,32,32,32], use_mdn=True):
    self.n_verts = n_verts
    self.n_dims = n_dims
    self.n_mixes = n_mixes
    self.look_back = look_back
    self.cells = cells
    self.use_mdn = use_mdn
    self.LSTM = CuDNNLSTM if len(gpus) > 0 else LSTM
    self.model = self.build_model()
    if use_mdn:
      self.model.compile(loss=MDN(n_mixes, n_verts*n_dims).get_loss_func(), optimizer='adam', metrics=['accuracy'])
    else:
      self.model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    
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
    
  def softmax(self, x):
    ''''Compute softmax values for vector `x`'''
    r = np.exp(x - np.max(x))
    return r / r.sum()


X = data.selected.X
n_verts, n_time, n_dims = X.shape
n_mixes = 3
look_back = 10

lstm_mdn = LSTM_MDN(n_verts=n_verts, n_dims=n_dims, n_mixes=n_mixes, look_back=look_back)
train_X, train_Y = lstm_mdn.prepare_inputs(X, look_back=look_back)


from keras.callbacks import TerminateOnNaN

lstm_mdn = LSTM_MDN(n_verts=n_verts, n_dims=n_dims, n_mixes=n_mixes, look_back=look_back)
history = lstm_mdn.model.fit(train_X, train_Y, epochs=10000, batch_size=128, shuffle=False, callbacks=[TerminateOnNaN()])

from keras.models import load_model
lstm_mdn.model.save_weights('weights/mdn_nopca_10000epochs_gpu_weights.h5')
print("Weights saved!")