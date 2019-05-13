#!/bin/bash
#SBATCH --partition gpu
#SBATCH --time 12:00:00
#SBATCH --job-name gpu
#SBATCH --output logs/rnn_nopca_gpu-%J.log

source ~/.bashrc
conda activate choreo
python main.py


