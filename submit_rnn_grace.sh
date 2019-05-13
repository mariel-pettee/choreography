#!/bin/bash
#SBATCH --partition day
#SBATCH --time 00:01:00
#SBATCH --job-name test
#SBATCH --output logs/test-%J.log

source ~/.bashrc
conda activate choreo
python rnn.py weights/weights_test.h5 weights/model_test.h5


