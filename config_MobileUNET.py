import torch
import os
from os.path import dirname, join as pjoin
import numpy as np

# Case

Case = 'Test'    # Name of the case/folder in the Data folder
Vmat = 7.3            # Version of the .mat file. 

# GPU/CPU detection

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Training parameters

learning_rate = [200]   # *1e-6
num_epochs =250         # max number of epoch
batch_size = 2 #16          # batch size
val_per = 25            # Percentage of the database used for validation
patience = 100000          # Number of epoch without improvement of the validation loss for the early stopping