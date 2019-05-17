from functions.functions import *
from functions.mdn import *

### Initializing & loading data:
parser = argparse.ArgumentParser()
parser.add_argument("name", type=str, help="Name to identify this training iteration")
parser.add_argument("--cells", type=int, nargs='+', help="Number of nodes for each of the 3 LSTM layers and the final dense layer", default=(32,32,32,32))
parser.add_argument("--look_back", type=int, help="Number of frames to prompt the next frame", default=10)
parser.add_argument("--n_mixes", type=int, help="Number of Gaussians to use in the MDN", default=3)
parser.add_argument("--n_epochs", type=int, help="Number of epochs to train", default=10)
parser.add_argument("--batch_size", type=int, help="Batch size", default=128)

### Set up & load data
args = parser.parse_args()
setup_gpus()
data = load_data('data/mariel_*')
X = data.selected.X  # only 15 joints! If you want all the joints, do data.all.X

### Average frame-by-frame in (x,y):
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
print('Reduced data has the shape: {}'.format(pca_reduced_data.shape))
n_time, n_dims, n_verts  = pca_reduced_data.shape[0], 1, pca_reduced_data.shape[1]

print('PCA means: {}'.format(pca.mean_))
np.save('pca/'+args.name+'-means.npy', pca.mean_)
print('PCA components: {}'.format(pca.components_))
np.save('pca/'+args.name+'-components.npy', pca.components_)

### Build the model:
look_back = args.look_back
n_mixes = args.n_mixes
print("look_back = {}".format(look_back))
print("n_mixes = {}".format(n_mixes))

lstm_mdn = LSTM_MDN(cells = [32,32,32,32], n_verts=n_verts, n_dims=n_dims, n_mixes=n_mixes, look_back=look_back)

train_X = []
train_Y = []
for i in range(look_back, n_time, 1):
    train_X.append( pca_reduced_data[i-look_back:i,:] ) # look_back, verts * dims
    train_Y.append( pca_reduced_data[i,:] ) # verts * dims
train_X = np.array(train_X) # n_samples, lookback, verts * dims
train_Y = np.array(train_Y) # n_samples, verts * dims

print(train_X.shape, train_Y.shape)
lstm_mdn.model.summary()

### Save the model as a .json file:
from keras.models import model_from_json
model_json = lstm_mdn.model.to_json()
model_path = "models/model-"+args.name+".json"
with open(model_path, "w") as json_file:
    json_file.write(model_json)
print("Model saved as {}!".format(model_path))

### Declare your training parameters:
n_epochs = args.n_epochs
batch_size = args.batch_size

### Train the model:
checkpoint_filepath = "weights/weights-pca-"+args.name+".h5"
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')

history = lstm_mdn.model.fit(train_X, train_Y, epochs=n_epochs, batch_size=batch_size, shuffle=False, verbose=2, callbacks=[checkpoint, TerminateOnNaN()])
