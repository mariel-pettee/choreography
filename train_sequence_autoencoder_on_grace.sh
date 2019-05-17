#!/bin/bash
#SBATCH --partition day
#SBATCH --time 1-
#SBATCH --cpus-per-task 26
#SBATCH --job-name vae3
#SBATCH --output logs/vae-1e-3-ctd2-%J.log

source ~/.bashrc
conda activate choreo
python sequence_autoencoder.py --lr 1e-3

