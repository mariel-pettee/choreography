# Beyond Imitation
'I didn’t want to imitate anybody. Any movement I knew, I didn’t want to use.' – Pina Bausch

### Create the Conda env from the .yaml file: 
```sh
conda env create --file choreo_env.yaml
conda activate choreo
python -m ipykernel install --user --name choreo --display-name "choreo" # installs the Conda kernel for use in Jupyter notebooks
```
If you need to recreate the .yaml file after adding packages, you can run: 
`conda env export -p /gpfs/loomis/project/hep/demers/mnp3/conda_envs/choreo > choreo_env.yaml`
