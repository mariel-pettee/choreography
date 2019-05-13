import keras
from keras import backend as K
import tensorflow as tf
import numpy as np
import os
import glob
from itertools import combinations
from keras.models import load_model
import argparse

def setup():
    # use tensorflow backend
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    # set random seeds
    # tf.set_random_seed(1)
    # np.random.seed(1)
    # identify available GPU's
    gpus = K.tensorflow_backend._get_available_gpus()
    # allow dynamic GPU memory allocation
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    print("GPUs found: {}".format(len(gpus)))
    return()
    
def load_data(files):
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
    infile_glob = sorted(glob.glob(files))
    data = Data(infile_glob, labels, bad_labels, edge_groups)
    print("Files loaded: {}".format(infile_glob))
    return data

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

class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()