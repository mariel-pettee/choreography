from functions.functions import *
from functions.mdn import *

### Initializing & loading data:
parser = argparse.ArgumentParser()
parser.add_argument("name", type=str, help="Name to identify this training iteration")
parser.add_argument("--cells", type=int, nargs='+', help="Number of nodes for each of the 3 LSTM layers and the final dense layer", default=(32,32,32,32))
args = parser.parse_args()
setup()
data = load_data('data/mariel_*')
X = data.selected.X  # only 15 joints! If you want all the joints, do data.all.X
print("Preparing to train...")

# Average frame-by-frame in (x,y):
X = X.swapaxes(0, 1)
X[:,:,0] = X[:,:,0] - np.mean(X[:,:,0], axis=0) + 0.5*np.ones(15)
X[:,:,1] = X[:,:,1] - np.mean(X[:,:,1], axis=0) + 0.5*np.ones(15)
X = X.reshape(X.shape[0],X.shape[1]*X.shape[2])

# PCA time:
from sklearn.decomposition import PCA
pca = PCA(.95)
pca_reduced_data = pca.fit_transform(X)
print('PCA reduction to a {}-dimensional latent space.'.format(pca_reduced_data.shape[1]))
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
np.save("weights/pca_means-"+args.name+".npy", pca.mean_)
np.save("weights/pca_components-"+args.name+".npy", pca.components_)

### Declare your model parameters:
cells = args.cells
n_time, n_dims, n_verts  = pca_reduced_data.shape[0], 1, pca_reduced_data.shape[1]
n_mixes = 3
look_back = 10

train_X = []
train_Y = []
for i in range(look_back, n_time, 1):
    train_X.append( pca_reduced_data[i-look_back:i,:] ) # look_back, verts * dims
    train_Y.append( pca_reduced_data[i,:] ) # verts * dims
train_X = np.array(train_X) # n_samples, lookback, verts * dims
train_Y = np.array(train_Y) # n_samples, verts * dims

print("train_X shape: {}".format(train_X.shape))
print("train_Y shape: {}".format(train_Y.shape))

### Define the model:
lstm_mdn = LSTM_MDN(cells = cells, n_verts=n_verts, n_dims=n_dims, n_mixes=n_mixes, look_back=look_back)
lstm_mdn.model.summary()

### Save the model architecture as a .json file:
from keras.models import model_from_json
model_json = lstm_mdn.model.to_json()
model_path = "weights/model-"+args.name+".json"
with open(model_path, "w") as json_file:
    json_file.write(model_json)
print("Model saved as {}!".format(model_path))

### Declare your training parameters:
n_epochs = 10
batch_size = 128

### Train the model:
checkpoint_filepath="weights/weights-"+args.name+".json"
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')

history = lstm_mdn.model.fit(train_X, train_Y, epochs=n_epochs, batch_size=batch_size, shuffle=False, verbose=2, callbacks=[checkpoint, TerminateOnNaN()])

# ### Save the weights:
# lstm_mdn.model.save_weights(args.weight_path)
# print("Weights saved as {}!".format(args.weight_path))
