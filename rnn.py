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
parser.add_argument("--lr", type=float, help="Learning rate", default=1e-3)
parser.add_argument("--use_pca", action='store_true', help="Use PCA compression of each frame")
parser.add_argument("--load_weights", action='store_true', help="Load in a pre-trained set of weights to continue training")
parser.add_argument("--weights", type=str, help="Weights file to load", default="")

args = parser.parse_args()
setup_gpus()
data = load_data('rnn_data/mariel_*')
X = data.selected.X

print(args.use_pca)
print("Preparing to train...")

if args.use_pca:
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

### Declare your model parameters:
cells = args.cells
look_back = args.look_back
n_mixes = args.n_mixes
if args.use_pca:
	n_time, n_dims, n_verts  = pca_reduced_data.shape[0], 1, pca_reduced_data.shape[1]
else:
	n_verts, n_time, n_dims = X.shape
lr = args.lr
print("learning rate = {}".format(lr))
print("look_back = {}".format(look_back))
print("n_mixes = {}".format(n_mixes))

### Build the model:
lstm_mdn = LSTM_MDN(cells=cells, n_verts=n_verts, n_dims=n_dims, n_mixes=n_mixes, look_back=look_back, lr=lr)
print(lstm_mdn.model.summary())

if args.use_pca:
	train_X = []
	train_Y = []
	for i in range(look_back, n_time, 1):
	    train_X.append(pca_reduced_data[i-look_back:i,:]) # look_back, verts * dims
	    train_Y.append(pca_reduced_data[i,:] ) # verts * dims
	train_X = np.array(train_X) # n_samples, lookback, verts * dims
	train_Y = np.array(train_Y) # n_samples, verts * dims
else:
	train_X, train_Y = lstm_mdn.prepare_inputs(X, look_back=look_back)

print("total X shape: {}".format(train_X.shape))
print("total Y shape: {}".format(train_Y.shape))

### Save the model as a .json file:
from keras.models import model_from_json
model_json = lstm_mdn.model.to_json()
model_path = "models/model-"+args.name+".json"
with open(model_path, "w") as json_file:
    json_file.write(model_json)
print("Model saved as {}!".format(model_path))

### Optional: Load some pre-trained weights
if args.load_weights:
	lstm_mdn.model.load_weights(args.weights)

### Declare your training parameters:
n_epochs = args.n_epochs
batch_size = args.batch_size

### Train the model:
checkpoint_filepath = "weights/weights-"+args.name+".h5"
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history = lstm_mdn.model.fit(train_X, train_Y, validation_split=0.2, epochs=n_epochs, batch_size=batch_size, shuffle=False, verbose=2, callbacks=[checkpoint, TerminateOnNaN()])
