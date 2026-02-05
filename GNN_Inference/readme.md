# Instructions for GNN inference

Once the dependencies have been installed, the GNN_notebook_masking.ipynp should work for inferring the csv datafile in this folder. At the end we plot the true forces against the inferred forces. Note that the simulation this is inferring has a fluctuating particle number, with beads interacting via a soft lennard-jones pair potential and in a 2D asymmetric well with kappa_x = 0.4 and kappa_y = 0.2. Other parameters are specified in the notebook.

Dependencies:

tensorflow
sympy
pandas
numpy
matplotlib
pathlib

A more simple and well-commented version of the main GNN_inference.py code is in GNN_inference_Simple.py which can be used for the purposes of learning but it will not work for the current datafile in this directory. 

