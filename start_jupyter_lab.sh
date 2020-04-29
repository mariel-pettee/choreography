#!/bin/bash
#SBATCH --partition day
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem 20G
#SBATCH --time 3:00:00
#SBATCH --job-name jupyter-lab
#SBATCH --output logs/jupyter-lab_%j.log
#SBATCH -c 1

# get tunneling info
XDG_RUNTIME_DIR=""
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)
cluster=$(hostname -f | awk -F"." '{print $2}')

# print tunneling instructions jupyter-log
echo -e "
Open up a new terminal window on your local laptop and run:
ssh -N -L ${port}:${node}:${port} ${user}@${cluster}.hpc.yale.edu
   
Use a Browser on your local machine to go to:
localhost:${port}

To get the token, open this log file again and copy and paste the thing after 'token=' in the URL.
"
source ~/.bashrc
conda activate choreo
jupyter lab --no-browser --port=${port} --ip=${node}
