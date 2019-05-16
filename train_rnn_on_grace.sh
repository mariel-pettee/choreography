#!/bin/bash
#SBATCH --partition bigmem
#SBATCH --time 12:00:00
#SBATCH --job-name quarterchor-rnn2
#SBATCH --output logs/quarterchor-rnn2-%J.log
#SBATCH --mem 100GB

source ~/.bashrc
conda activate choreo
python rnn.py quarter-chor-rnn2 --cells 256 256 256 --n_mixes 6 --look_back 256 --batch_size 128 --n_epochs 1000


# python rnn_with_pca.py rnn_pca_32 
