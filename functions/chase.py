import os
from glob import glob
import tensorflow as tf
import keras
from keras.models import model_from_json
import keras.backend as K
import keras.layers as layers
from keras.models import Model, load_model
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import juggle_axes
from IPython.display import HTML
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['animation.ffmpeg_path'] = '/gpfs/loomis/project/hep/demers/mnp3/conda_envs/choreo/bin/ffmpeg' # for using html5 video in Jupyter notebook
# print(matplotlib.animation.writers.list()) # check that ffmpeg is loaded. if it's not there, use .to_jshtml() instead of .to_html5_video().

def setup_gpus():
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    gpus = K.tensorflow_backend._get_available_gpus()
    print("GPUs found: {}".format(len(gpus)))

def load_data(pattern="vae_data/mariel_*.npy"):
   # load up the six datasets, performing some minimal preprocessing beforehand
    datasets = {}
    ds_all = []
    
    exclude_points = [26,53]
    point_mask = np.ones(55, dtype=bool)
    point_mask[exclude_points] = 0
    
    for f in sorted(glob(pattern)):
        ds_name = os.path.basename(f)[7:-4]
        print("loading:", ds_name)
        ds = np.load(f).transpose((1,0,2))
        ds = ds[500:-500, point_mask]
        print("\t Shape:", ds.shape)

        ds[:,:,2] *= -1
        print("\t Min:", np.min(ds,axis=(0,1)))
        print("\t Max:", np.max(ds, axis=(0,1)))

        #ds = filter_points(ds)

        datasets[ds_name] = ds
        ds_all.append(ds)

    ds_counts = np.array([ds.shape[0] for ds in ds_all])
    ds_offsets = np.zeros_like(ds_counts)
    ds_offsets[1:] = np.cumsum(ds_counts[:-1])

    ds_all = np.concatenate(ds_all)
    print("Full data shape:", ds_all.shape)
    # print("Offsets:", ds_offsets)

    # print(ds_all.min(axis=(0,1)))
    low,hi = np.quantile(ds_all, [0.01,0.99], axis=(0,1))
    xy_min = min(low[:2])
    xy_max = max(hi[:2])
    xy_range = xy_max-xy_min
    ds_all[:,:,:2] -= xy_min
    ds_all *= 2/xy_range
    ds_all[:,:,:2] -= 1.0

    # it's also useful to have these datasets centered, i.e. with the x and y offsets
    # subtracted from each individual frame

    ds_all_centered = ds_all.copy()
    ds_all_centered[:,:,:2] -= ds_all_centered[:,:,:2].mean(axis=1,keepdims=True)

    datasets_centered = {}
    for ds in datasets:
        datasets[ds][:,:,:2] -= xy_min
        datasets[ds] *= 2/xy_range
        datasets[ds][:,:,:2] -= 1.0
        datasets_centered[ds] = datasets[ds].copy()
        datasets_centered[ds][:,:,:2] -= datasets[ds][:,:,:2].mean(axis=1,keepdims=True)

    # print(ds_all.min(axis=(0,1)))
    low,hi = np.quantile(ds_all, [0.01,0.99], axis=(0,1))
    return ds_all, ds_all_centered, datasets, datasets_centered, ds_counts


# these are the ordered label names of the 53 vertices
# (after the Labeling/SolvingHips points have been excised)
point_labels = ['ARIEL', 'C7',
          'CLAV', 'LANK',
          'LBHD', 'LBSH',
          'LBWT', 'LELB',
          'LFHD', 'LFRM',
          'LFSH', 'LFWT',
          'LHEL', 'LIEL',
          'LIHAND', 'LIWR',
          'LKNE', 'LKNI',
          'LMT1', 'LMT5',
          'LOHAND', 'LOWR',
          'LSHN', 'LTHI',
          'LTOE', 'LUPA',
          #'LabelingHips',
          'MBWT',
          'MFWT', 'RANK',
          'RBHD', 'RBSH',
          'RBWT', 'RELB',
          'RFHD', 'RFRM',
          'RFSH', 'RFWT',
          'RHEL', 'RIEL',
          'RIHAND', 'RIWR',
          'RKNE', 'RKNI',
          'RMT1', 'RMT5',
          'ROHAND', 'ROWR',
          'RSHN', 'RTHI',
          'RTOE', 'RUPA',
          'STRN',
          #'SolvingHips',
          'T10']

# This array defines the points between which skeletal lines should
# be drawn. Each segment is defined as a line between a group of one
# or more named points -- the line will be drawn at the average position
# of the points in the group
skeleton_lines = [
    # ( (start group), (end group) ),
    (('LHEL',), ('LTOE',)), # toe to heel
    (('RHEL',), ('RTOE',)),
    (('LKNE','LKNI'), ('LHEL',)), # heel to knee
    (('RKNE','RKNI'), ('RHEL',)),
    (('LKNE','LKNI'), ('LFWT','RFWT','LBWT','RBWT')), # knee to "navel"
    (('RKNE','RKNI'), ('LFWT','RFWT','LBWT','RBWT')),
    (('LFWT','RFWT','LBWT','RBWT'), ('STRN','T10',)), # "navel" to chest
    (('STRN','T10',), ('CLAV','C7',)), # chest to neck
    (('CLAV','C7',), ('LFSH','LBSH',),), # neck to shoulders
    (('CLAV','C7',), ('RFSH','RBSH',),),
    (('LFSH','LBSH',), ('LELB', 'LIEL',),), # shoulders to elbows
    (('RFSH','RBSH',), ('RELB', 'RIEL',),),
    (('LELB', 'LIEL',), ('LOWR','LIWR',),), # elbows to wrist
    (('RELB', 'RIEL',), ('ROWR','RIWR',),),
    (('LFHD',), ('LBHD',)), # draw lines around circumference of the head
    (('LBHD',), ('RBHD',)),
    (('RBHD',), ('RFHD',)),
    (('RFHD',), ('LFHD',)),
    (('LFHD',), ('ARIEL',)), # connect circumference points to top of head
    (('LBHD',), ('ARIEL',)),
    (('RBHD',), ('ARIEL',)),
    (('RFHD',), ('ARIEL',)),
]

skeleton_idxs = []
for g1,g2 in skeleton_lines:
    entry = []
    entry.append([point_labels.index(l) for l in g1])
    entry.append([point_labels.index(l) for l in g2])
    skeleton_idxs.append(entry)

# calculate the coordinates for the lines
def get_line_segments(seq, zcolor=None, cmap=None):
    xline = np.zeros((seq.shape[0],len(skeleton_idxs),3,2))
    if cmap:
        colors = np.zeros((len(skeleton_idxs), 4))
    for i,(g1,g2) in enumerate(skeleton_idxs):
        xline[:,i,:,0] = np.mean(seq[:,g1], axis=1)
        xline[:,i,:,1] = np.mean(seq[:,g2], axis=1)
        if cmap is not None:
            colors[i] = cmap(0.5*(zcolor[g1].mean() + zcolor[g2].mean()))
    if cmap:
        return xline, colors
    else:
        return xline
    
# put line segments on the given axis, with given colors
def put_lines(ax, segments, color=None, lw=2.5, alpha=None):
    lines = []
    for i in range(len(skeleton_idxs)):
        if isinstance(color, (list,tuple,np.ndarray)):
            c = color[i]
        else:
            c = color
        l = ax.plot(np.linspace(segments[i,0,0],segments[i,0,1],2),
                np.linspace(segments[i,1,0],segments[i,1,1],2),
                np.linspace(segments[i,2,0],segments[i,2,1],2),
                color=c,
                alpha=alpha,
                lw=lw)[0]
        lines.append(l)
    return lines

# animate a video of the stick figure.
# `ghost` may be a second sequence, which will be superimposed
# on the primary sequence.
# If ghost_shift is given, the primary and ghost sequence will be separated laterally
# by that amount.
# `zcolor` may be an N-length array, where N is the number of vertices in seq, and will
# be used to color the vertices. Typically this is set to the avg. z-value of each vtx.
def animate_stick(seq, ghost=None, ghost_shift=0, figsize=None, zcolor=None, pointer=None, ax_lims=(-0.4,0.4), speed=45,
                  dot_size=20, dot_alpha=0.5, lw=2.5, cmap='cool_r', pointer_color='black'):
    if zcolor is None:
        zcolor = np.zeros(seq.shape[1])
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    ax.axis('off')
    
    if ghost_shift and ghost is not None:
        seq = seq.copy()
        ghost = ghost.copy()
        seq[:,:,0] -= ghost_shift
        ghost[:,:,0] += ghost_shift
    
    cm = matplotlib.cm.get_cmap(cmap)
    
    pts = ax.scatter(seq[0,:,0],seq[0,:,1],seq[0,:,2], c=zcolor, s=dot_size, cmap=cm, alpha=dot_alpha)
    
    ghost_color = 'blue'

    if ghost is not None:
        pts_g = ax.scatter(ghost[0,:,0],ghost[0,:,1],ghost[0,:,2], c=ghost_color, s=dot_size, alpha=dot_alpha)
    
    if ax_lims:
        ax.set_xlim(*ax_lims)
        ax.set_ylim(*ax_lims)
        ax.set_zlim(0,ax_lims[1]-ax_lims[0])
        #ax.set_zlim(*ax_lims)
    plt.close(fig)
    xline, colors = get_line_segments(seq, zcolor, cm)
    lines = put_lines(ax, xline[0], colors, lw=lw)
    
    if ghost is not None:
        xline_g = get_line_segments(ghost)
        lines_g = put_lines(ax, xline_g[0], ghost_color, lw=lw, alpha=1.0)
    
    if pointer is not None:
        vR = 0.15
        dX,dY = vR*np.cos(pointer), vR*np.sin(pointer)
        zidx = point_labels.index('CLAV')
        X = seq[:,zidx,0]
        Y = seq[:,zidx,1]
        Z = seq[:,zidx,2]
        #Z = seq[:,2,2]
        quiv = ax.quiver(X[0],Y[0],Z[0],dX[0],dY[0],0, color=pointer_color)
        ax.quiv = quiv
    
    def update(t):
        pts._offsets3d = juggle_axes(seq[t,:,0], seq[t,:,1], seq[t,:,2], 'z')
        for i,l in enumerate(lines):
            l.set_data(xline[t,i,:2])
            l.set_3d_properties(xline[t,i,2])
        
        if ghost is not None:
            pts_g._offsets3d = juggle_axes(ghost[t,:,0], ghost[t,:,1], ghost[t,:,2], 'z')
            for i,l in enumerate(lines_g):
                l.set_data(xline_g[t,i,:2])
                l.set_3d_properties(xline_g[t,i,2])
        
        if pointer is not None:
            ax.quiv.remove()
            ax.quiv = ax.quiver(X[t],Y[t],Z[t],dX[t],dY[t],0,color=pointer_color)
     
    return animation.FuncAnimation(
        fig,
        update,
        len(seq),
        interval=speed,
        blit=False,
   ).to_html5_video()
    
# draw a "comic strip" style rendering of the given sequence of poses
def draw_comic(frames, angles=None, figsize=None, window_size=0.45, dot_size=20, lw=2.5, zcolor=None,cmap='cool_r'):
    fig = plt.figure(figsize=figsize)
    ax = p3.Axes3D(fig)
    ax.view_init(30, 0)
    shift_size=window_size
    
    ax.set_xlim(-window_size,window_size)
    ax.set_ylim(-window_size,len(frames)*window_size)
    ax.set_zlim(-0.1,0.6)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    
    cm = matplotlib.cm.get_cmap(cmap)
    
    if angles is not None:
        vR = 0.15
        zidx = point_labels.index("CLAV")
        X = frames[:,zidx,0]
        Y = frames[:,zidx,1]
        dX,dY = vR*np.cos(angles), vR*np.sin(angles)
        Z = frames[:,zidx,2]
        #Z = frames[:,2,2]
 
    
    for iframe,frame in enumerate(frames):
        ax.scatter(frame[:,0],
                       frame[:,1]+iframe*shift_size,
                       frame[:,2],
                       alpha=0.3,
                       c=zcolor,
                       cmap=cm,
                       s=dot_size,
                       depthshade=True)
        
        if angles is not None:
            ax.quiver(X[iframe],iframe*shift_size+Y[iframe],Z[iframe],dX[iframe],dY[iframe],0, color='black')
        
        for i,(g1,g2) in enumerate(skeleton_lines):
            g1_idx = [point_labels.index(l) for l in g1]
            g2_idx = [point_labels.index(l) for l in g2]

            if zcolor is not None:
                color = cm(0.5*(zcolor[g1_idx].mean() + zcolor[g2_idx].mean()))
            else:
                color = None

            x1 = np.mean(frame[g1_idx],axis=0)
            x2 = np.mean(frame[g2_idx],axis=0)
            
            ax.plot(np.linspace(x1[0],x2[0],10),
                    np.linspace(x1[1],x2[1],10)+iframe*shift_size,
                    np.linspace(x1[2],x2[2],10),
                    color=color,
                    lw=lw)

# Rotate a (?,...,?,3) tensor about the z-axis
def rotate(X, theta):
    c,s = np.cos(theta), np.sin(theta)
    R = np.array([
        [c, -s, 0],
        [s, c, 0],
        [0, 0, 1.]
        ])
    return np.dot(X, R)

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
            self.R = K.constant([[1,0,0],[0,1,0],[0,0,0]])*tf.cos(theta) \
                       + K.constant([[0,-1,0],[1,0,0],[0,0,0]])*tf.sin(theta) \
                       + K.constant([[0,0,0],[0,0,0],[0,0,1]])
        elif dim == 2:
            self.R = K.constant([[1,0],[0,1]])*tf.cos(theta) \
                       + K.constant([[0,-1],[1,0]])*tf.sin(theta)
        
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
    
    
    auto_input = layers.Input((seq_len,n_vtx,3))
    
    H = auto_input
    
    if do_rotations:
        theta = K.cast(K.learning_phase(),'float')*K.random_uniform((1,), 0, 2*np.pi)
        H = RotationLayer(theta)(H)
    
    auto_z, auto_mean, auto_log_var = encoder(H)
    H = decoder(auto_z)
    
    if do_rotations:
        H = RotationLayer(-theta)(H)
    
    auto_output = H

    auto = Model(auto_input, auto_output)
    
    
    auto.hp_resolution = K.variable(resolution)
    
    ae_loss = 0.5*K.mean(K.sum(K.square(auto_input - auto_output), axis=-1))
    auto.add_loss(ae_loss/K.square(auto.hp_resolution))

    if kl_weight:
        kl_loss = -0.5*K.mean(K.sum(1 + auto_log_var - K.square(auto_mean) - K.exp(auto_log_var), axis=-1))

        auto.hp_kl_weight = K.variable(kl_weight)
        auto.add_loss(auto.hp_kl_weight * kl_loss)

    auto.compile(optimizer='adam')
    
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
        
        continuizer.add_loss(0.5/K.square(auto.hp_resolution)*\
                             K.mean(K.sum(curve*K.square(shift_residual([continuizer_x0, continuizer_x1])), axis=-1)))
        
        decoder.trainable = False
        encoder.trainable = False
        continuizer.compile(optimizer='adam')
        #continuizer.compile(optimizer='rmsprop')
        decoder.trainable = True
        encoder.trainable = True
        return continuizer
    
    return encoder, decoder, auto, mk_continuizer

