#!/bin/bash
#SBATCH --partition day
#SBATCH --time 6:00:00
#SBATCH --job-name 256
#SBATCH --output logs/rnn_256-%J.log

source ~/.bashrc
conda activate choreo
python rnn.py weights/model_rnn_256.json weights/weights_rnn_256.h5 --cells 256 256 256 256


