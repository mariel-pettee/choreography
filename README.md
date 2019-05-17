# Beyond Imitation
'I didn’t want to imitate anybody. Any movement I knew, I didn’t want to use.' – Pina Bausch

### Create the Conda env from the .yaml file: 
```sh
conda env create --file choreo_env.yaml
conda activate choreo
python -m ipykernel install --user --name choreo --display-name "choreo" # installs the Conda kernel for use in Jupyter notebooks
```
If you need to recreate the .yaml file after adding packages, you can run: 
```sh
conda env export -p /path/to/conda_envs/choreo > choreo_env.yaml
```
### Play with the RNN model
This model, inspired by chor-rnn (https://arxiv.org/abs/1605.06921), uses 3 LSTM layers to predict new poses given a prompt of a sequence of poses. The length of the prompt is called `look_back`. We use a Mixture Density Network (https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf) to create multiple Gaussian distributions of potential poses given a prompt sequence. The number of Gaussian distributions is determined by `n_mixes`. 

You can experiment with this model interactively in a Jupyter notebook using `rnn.ipynb` or via the command line with commands such as: 
```
conda activate choreo
python rnn.py rnn_test --cells 64 64 64 64 --n_mixes 25 --look_back 128 --batch_size 128 --n_epochs 10 --lr 1e-4 --use_pca True
```
