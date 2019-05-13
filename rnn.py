from functions.functions import *
from functions.mdn import *

### Initializing & loading data:
parser = argparse.ArgumentParser()
parser.add_argument("weight_path", type=str, help="Filename for saving the trained weights")
parser.add_argument("model_path", type=str, help="Filename for saving the model .json file")
args = parser.parse_args()
setup()
data = load_data('data/mariel_*')
X_train = data.selected.X[:,:,:]
X = data.selected.X  # only 15 joints! If you want all the joints, do data.all.X
n_verts, n_time, n_dims = X.shape
print("Training dataset shape: {}".format(X_train.shape))
print("Preparing to train...")

### Declare your training parameters:
n_epochs = 2
cells = [32,32,32,32]
batch_size = 128
look_back = 10
n_mixes = 3

lstm_mdn = LSTM_MDN(n_verts=n_verts, n_dims=n_dims, n_mixes=n_mixes, look_back=look_back)
print(lstm_mdn.model.summary())
train_X, train_Y = lstm_mdn.prepare_inputs(X, look_back=look_back)

### Save the model as a .json file:
from keras.models import model_from_json
model_json = lstm_mdn.model.to_json()
with open(args.model_path, "w") as json_file:
    json_file.write(model_json)
print("Model saved as {}!".format(args.model_path))

### Train the model:
history = lstm_mdn.model.fit(train_X, train_Y, epochs=n_epochs, batch_size=batch_size, shuffle=False, verbose=2, callbacks=[TerminateOnNaN()])

### Save the weights:
lstm_mdn.model.save_weights(args.weight_path)
print("Weights saved as {}!".format(args.weight_path))
