#!/bin/bash
#SBATCH --partition day
#SBATCH --time 1-
#SBATCH --job-name 32pca
#SBATCH --output logs/32pca-%J.log
#SBATCH --mem 40GB

source ~/.bashrc
conda activate choreo
# python rnn.py 256chor-rnn-noweightsloaded --cells 256 256 256 --n_mixes 6 --look_back 128 --batch_size 128 --n_epochs 1000


python rnn_with_pca.py rnn_pca_32 --cells 32 32 32 --n_mixes 6 --look_back 128 --batch_size 128 --n_epochs 1000
