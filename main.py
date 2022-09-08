import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import time
import datetime


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

# Dimensions definition
INPUT_DIM     = (256,256,3)
OUTPUT_CHANNELS = 1 #INPUT_DIM[-1]
BATCH_SIZE    = 10
R_LOSS_FACTOR = 10000
EPOCHS        = 100
INITIAL_EPOCH = 0

# Define local dir for images
data_dir = '/home/est_posgrado_manuel.suarez/data/sintel'
# Define expresions to load images
xl_files = glob(os.path.join(data_dir, 'final_left', '**/*.png'))
xr_files = glob(os.path.join(data_dir, 'final_right', '**/*.png'))
y_files  = glob(os.path.join(data_dir, 'disparities_viz', '**/*.png'))
# Sort
xl_files.sort()
xr_files.sort()
y_files.sort()
# Convert to np arrays
xl_files = np.array(xl_files)
xr_files = np.array(xr_files)
y_files  = np.array(y_files)
# Check first 5 files
for xl, xr, y in zip(xl_files[:5], xr_files[:5], y_files[:5]):
  print(xl, xr, y)