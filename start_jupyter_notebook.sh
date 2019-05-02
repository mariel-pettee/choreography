echo -e "
Open up a new Terminal window and enter:
ssh -N -f -L localhost:8888:localhost:8889 cuda@cuda.library.yale.edu

Then open a new browser window and go to the URL: localhost:8888
Copy-paste the token from the original window to log in.
"
# jupyter notebook --no-browser --port=8889
jupyter-lab --no-browser --port=8889