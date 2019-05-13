#!/bin/bash
#SBATCH --partition day
#SBATCH --time 12:00:00
#SBATCH --job-name 64
#SBATCH --output logs/rnn_64-%J.log

source ~/.bashrc
conda activate choreo
python rnn.py weights/model_rnn_64.json weights/weights_rnn_64.h5 --cells 64 64 64 64


