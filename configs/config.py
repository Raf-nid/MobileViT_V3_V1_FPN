import torch
import os
from os.path import dirname, join as pjoin
import numpy as np

# Case

Case = 'WedgeFull'
Vmat = 7.3

# GPU/CPU detection

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Training parameters

learning_rate = [1e-6]
num_epochs = 50000
batch_size = 4
val_per = 25
val_size = 25
patience = 500

# Data parameters

dat_factor = 1
SNR = 0.05
Fs = 100e6
order = 1
Lowcut = 3e6
Highcut = 7e6

# Saving Parameters

plt_inc = patience#np.round(num_epochs / 200).astype(int)
plt_step = 1000