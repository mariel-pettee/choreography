from keras.models import Model
from keras.layers import Input, Reshape, Dense, Flatten, Dropout, LeakyReLU, Add, Subtract, Lambda
import numpy as np

class Autoencoder:
  def __init__(self,
               n_verts=0,
               n_dims=3,
               latent_dim=2,
               n_layers=2,
               n_units=128,
               relu=False,
               add_random_offsets=False,
               dropout=False):
    if not n_verts: raise Exception('Please provide the number of vertices `n_verts`')
    self.n_verts = n_verts # input vert count
    self.n_dims = n_dims # input dimensions
    self.relu = relu # whether to add relu layers in encoder/decoder
    self.dropout = dropout # whether to add dropout layers in encoder/decoder
    self.latent_dim = latent_dim
    self.n_layers = n_layers
    self.n_units = n_units
    self.encoder = self.build_encoder()
    self.decoder = self.build_decoder()
    # attach the encoder and decoder
    i = Input((self.n_verts, self.n_dims))
    if add_random_offsets:
        random_offsets = K.cast(K.learning_phase(),'float')*K.random_uniform((K.shape(i)[0],1,3))*K.constant([[[1,1,0]]])
        offset_layer = Lambda(lambda x: x + random_offsets)
        offset_layer.uses_learning_phase = True
        i_offset = offset_layer(i)
    else: 
        i_offset = i
    z = self.encoder(i_offset) # push observations into latent space
    o = self.decoder(z) # project from latent space to feature space
    if add_random_offsets:
        o = Lambda(lambda x: x - random_offsets)(o)
    self.model = Model(inputs=[i], outputs=[o])
    self.model.compile(loss='mse', optimizer='adam')
    
  def build_encoder(self):
    i = Input((self.n_verts, self.n_dims))
    h = i
    h = Flatten()(h)
    for _ in range(self.n_layers):
      h = Dense(self.n_units)(h)
      if self.relu: h = LeakyReLU(alpha=0.2)(h)
      if self.dropout: h = Dropout(0.4)(h)
    o = Dense(self.latent_dim)(h)
    return Model(inputs=[i], outputs=[o])
  
  def build_decoder(self):
    i = Input((self.latent_dim,))
    h = i
    for _ in range(self.n_layers):
      h = Dense(self.n_units)(h)
      if self.relu: h = LeakyReLU(alpha=0.2)(h)
      if self.dropout: h = Dropout(0.4)(h)
    h = Dense(self.n_verts * self.n_dims)(h)
    o = Reshape((self.n_verts, self.n_dims))(h) # predict 1 frame
    return Model(inputs=[i], outputs=[o])

  def train(self, X, n_epochs=10000):
    for idx in range(n_epochs):
      i = np.random.randint(0, X.shape[1]-1) # sample idx
      frame = np.expand_dims( X[:,i:i+1,:].squeeze(), axis=0) # shape = 1 sample, v verts, d dims
      loss = self.model.train_on_batch(frame, frame)
      if idx == 0: print(frame.shape)
      if idx % 1000 == 0:
        print(' * training idx', idx, 'loss', loss)

  def get_predictions(self, X, n_frames=50, start_frame=0):
    '''Return the model's predictions of observations from X in shape of X'''  
    predictions = []
    for i in range(start_frame, start_frame+n_frames, 1):
      x = np.expand_dims(X[:,i:i+1,:].squeeze(), axis=0)
      predictions.append( self.model.predict(x) )
    return np.swapaxes(np.vstack(predictions), 0, 1)