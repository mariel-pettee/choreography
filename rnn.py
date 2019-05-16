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

args = parser.parse_args()
setup()
data = load_data('data/mariel_*')
X = data.selected.X  # only 15 joints! If you want all the joints, do data.all.X
print("Preparing to train...")

### Declare your model parameters:
cells = args.cells
look_back = args.look_back
n_mixes = args.n_mixes
n_verts, n_time, n_dims = X.shape
print("look_back = {}".format(look_back))
print("n_mixes = {}".format(n_mixes))

### Build the model:
lstm_mdn = LSTM_MDN(cells=cells, n_verts=n_verts, n_dims=n_dims, n_mixes=n_mixes, look_back=look_back)
print(lstm_mdn.model.summary())
train_X, train_Y = lstm_mdn.prepare_inputs(X, look_back=look_back)
print("train_X shape: {}".format(train_X.shape))
print("train_Y shape: {}".format(train_Y.shape))

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
checkpoint_filepath = "weights/weights-"+args.name+".h5"
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')

history = lstm_mdn.model.fit(train_X, train_Y, epochs=n_epochs, batch_size=batch_size, shuffle=False, verbose=2, callbacks=[checkpoint, TerminateOnNaN()])

# ### Save the weights:
# lstm_mdn.model.save_weights(args.weight_path)
# print("Weights saved as {}!".format(args.weight_path))
