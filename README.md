# Beyond Imitation
'I didn’t want to imitate anybody. Any movement I knew, I didn’t want to use.' – Pina Bausch

Paper: https://arxiv.org/abs/1907.05297
Mariel Pettee, Chase Shimmin, Douglas Duhaime, and Ilya Vidrin. Beyond Imitation: Generative and Variational Choreography via Machine Learning. Proceedings of the 10th International Conference on Computational Creativity. (2019).

### Getting started: 
I like to work within Conda environments to manage package dependencies. To download Conda (Miniconda is sufficient) for your particular system, and for Python 3, check out: https://docs.conda.io/en/latest/miniconda.html

Once that's installed, clone the repository and set up the Conda environment:
```sh
git clone https://github.com/mariel-pettee/choreography.git
cd choreography
conda env create -n choreo -f env_tf1.yml
```
Then, activate the environment and install a kernel for use in JupyterLab: 

```
conda activate choreo
python -m ipykernel install --user --name choreo --display-name "choreo"
```

Note that when opening a Jupyter notebook, to use the same packages as you've installed here, you need to select "choreo" from the list of kernels within your notebook.

To display animations live in the Jupyter notebook environment, we recommend installing `ffmpeg` (https://ffmpeg.org/download.html) in your main repository directory. If you'd prefer not to do this, you can also change the `to_html5_video()` commands to `.to_jshtml()`.

### Play with the RNN model
This model, inspired by chor-rnn (https://arxiv.org/abs/1605.06921), uses 3 LSTM layers to predict new poses given a prompt of a sequence of poses. The length of the prompt is called `look_back`. We use a Mixture Density Network (https://publications.aston.ac.uk/id/eprint/373/1/NCRG_94_004.pdf) to create multiple Gaussian distributions of potential poses given a prompt sequence. The number of Gaussian distributions is determined by `n_mixes`. 

You can experiment with this model interactively in a Jupyter notebook using `rnn.ipynb` or via the command line with commands such as: 
```
conda activate choreo
python rnn.py rnn_test --cells 64 64 64 64 --n_mixes 25 --look_back 128 --batch_size 128 --n_epochs 10 --lr 1e-4 --use_pca True
```

### Play with the autoencoder for poses
This model uses an autoencoder structure to compress each pose into a lower-dimensional latent space and then back into its original dimension. After sufficient training, the latent space will group similar poses together, and sequences of poses can be visualized as paths throughout the latent space. Users can also construct their own movement sequences by drawing paths throughout the latent space and decoding them into their original dimensions. The interactive Jupyter notebook is `pose_autoencoder.ipynb`.

### Play with the autoencoder for sequences
This model also uses an autoencoder structure, but for fixed-length sequences of movements, or 'phrases'. This can be then used in two primary ways: 
1. Sample randomly from within a given standard deviation in the latent space (which, when well-trained, should resemble an _n_-dimensional Gaussian distribution) to generate a new fixed-length movement sequence
2. Look at the location of a given sequence in data in the latent space, then add a small deviation to this location and observe its motion. Small deviations (~0.5 sigma or less) will usually closely resemble the original sequence with subtle differences in timing or expressiveness. Larger deviations (~1 sigma or larger) will often capture a similar choreographic idea to the original phrase, but will become increasingly inventive.

Users can experiment with the interactive Jupyter notebook `sequence_autoencoder.ipynb`.
