#!/bin/bash
#SBATCH --partition day
#SBATCH --time 10:00:00
#SBATCH --cpus-per-task 26
#SBATCH --job-name vae
#SBATCH --output vae-%J.log

source ~/.bashrc
conda activate choreo
python chase_vae_notebook.py

