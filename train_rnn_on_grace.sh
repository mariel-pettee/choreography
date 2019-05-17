#!/bin/bash
#SBATCH --partition day
#SBATCH --time 3:00:00
#SBATCH --job-name rnn32
#SBATCH --output logs/chor_rnn32_rnn_lr1e-4-%J.log
#SBATCH --mem 10GB

source ~/.bashrc
conda activate choreo
python rnn.py chor_rnn32_lr1e-4 --cells 32 32 32 32 --n_mixes 25 --look_back 128 --batch_size 128 --n_epochs 1000 --lr 1e-4 --use_pca False

