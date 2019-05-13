#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres=gpu:p100:1
#SBATCH --time 12:00:00
#SBATCH --job-name gpup100
#SBATCH --output logs/rnn_nopca_gpup100-%J.log

source ~/.bashrc
conda activate choreo
python rnn.py weights/test.h5


