#!/bin/bash
#SBATCH --partition day
#SBATCH --time 6:00:00
#SBATCH --job-name rnn
#SBATCH --output logs/rnn-%J.log

source ~/.bashrc
conda activate choreo
python rnn.py weights/weights_rnn_128.h5 weights/model_rnn_128.h5


