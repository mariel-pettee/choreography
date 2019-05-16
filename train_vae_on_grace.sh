#!/bin/bash
#SBATCH --partition day
#SBATCH --time 12:00:00
#SBATCH --cpus-per-task 26
#SBATCH --job-name vae3
#SBATCH --output vae-1e-3-ctd-%J.log

source ~/.bashrc
conda activate choreo
python chase_vae_notebook.py --lr 1e-3

