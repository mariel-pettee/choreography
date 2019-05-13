from functions.functions import *
from functions.mdn import *

### Initializing & loading data:
parser = argparse.ArgumentParser()
parser.add_argument("model_path", type=str, help="Filename for saving the model .json file")
parser.add_argument("weight_path", type=str, help="Filename for saving the trained weights")
parser.add_argument("--cells", type=int, nargs='+', help="Number of nodes for each of the 3 LSTM layers and the final dense layer", default=(32,32,32,32))
args = parser.parse_args()
setup()
data = load_data('data/mariel_*')
X = data.selected.X  # only 15 joints! If you want all the joints, do data.all.X
print("Preparing to train...")

### Declare your model parameters:
cells = args.cells
look_back = 10
n_mixes = 3
n_verts, n_time, n_dims = X.shape

### Build the model:
lstm_mdn = LSTM_MDN(cells=cells, n_verts=n_verts, n_dims=n_dims, n_mixes=n_mixes, look_back=look_back)
print(lstm_mdn.model.summary())
train_X, train_Y = lstm_mdn.prepare_inputs(X, look_back=look_back)
print("train_X shape: {}".format(train_X.shape))
print("train_Y shape: {}".format(train_Y.shape))

### Save the model as a .json file:
from keras.models import model_from_json
model_json = lstm_mdn.model.to_json()
with open(args.model_path, "w") as json_file:
    json_file.write(model_json)
print("Model saved as {}!".format(args.model_path))

### Declare your training parameters:
n_epochs = 10000
batch_size = 128

### Train the model:
checkpoint_filepath=args.weight_path
checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')

history = lstm_mdn.model.fit(train_X, train_Y, epochs=n_epochs, batch_size=batch_size, shuffle=False, verbose=2, callbacks=[checkpoint, TerminateOnNaN()])

# ### Save the weights:
# lstm_mdn.model.save_weights(args.weight_path)
# print("Weights saved as {}!".format(args.weight_path))
