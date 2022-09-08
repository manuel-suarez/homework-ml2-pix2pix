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

# Parameters definition
print(len(xl_files), len(xr_files), len(y_files))
BUFFER_SIZE      = len(y_files)
steps_per_epoch  = BUFFER_SIZE // BATCH_SIZE
print('num image files : ', BUFFER_SIZE)
print('steps per epoch : ', steps_per_epoch )

# Definition of functions to load and decode images
def read_and_decode(file):
  '''
  Lee, decodifica y redimensiona la imagen.
  Aplica aumentación
  '''
  # Lectura y decodificación
  img = tf.io.read_file(file)
  img = tf.image.decode_png(img)
  img = tf.cast(img, tf.float32)
  # Normalización
  img = img / 127.5 - 1
  # Redimensionamiento
  img = tf.image.resize(img, INPUT_DIM[:2],
                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  return img


def load_images(xl_file, xr_file, y_file, flip=True):
  '''
  Lee el conjunto de imágenes de entrada y las redimensiona al tamaño especificado

  Aumentación: Flip horizontal aleatorio, sincronizado
  '''
  xl_img = read_and_decode(xl_file)
  xr_img = read_and_decode(xr_file)
  y_img = read_and_decode(y_file)
  # Aumentación (el flip debe aplicarse simultáneamente a las 3 imagenes)
  if flip and tf.random.uniform(()) > 0.5:
    xl_img = tf.image.flip_left_right(xl_img)
    xr_img = tf.image.flip_left_right(xr_img)
    y_img = tf.image.flip_left_right(y_img)

  return xl_img, xr_img, y_img


def display_images(fname, xl_imgs=None, xr_imgs=None, y_imgs=None, rows=3, offset=0):
  '''
  Despliega conjunto de imágenes izquierda y derecha junto a la disparidad
  '''
  # plt.figure(figsize=(20,rows*2.5))
  fig, ax = plt.subplots(rows, 3, figsize=(8, rows * 2.5))
  for i in range(rows):
    ax[i, 0].imshow((xl_imgs[i + offset] + 1) / 2)
    ax[i, 0].set_title('Left')
    ax[i, 1].imshow((xr_imgs[i + offset] + 1) / 2)
    ax[i, 1].set_title('Right')
    # Debido a que es de un solo canal requerimos obtener únicamente las dos
    # primeras dimensiones
    ax[i, 2].imshow((y_imgs[i + offset][:, :, 0] + 1) / 2)
    ax[i, 2].set_title('Disparities')

  plt.tight_layout()
  plt.savefig(fname)

xl_imgs = []
xr_imgs = []
y_imgs  = []
# Cargamos 3 imagenes
for i in range(3):
    xl_img, xr_img, y_img = load_images(xl_files[i], xr_files[i], y_files[i])
    xl_imgs.append(xl_img)
    xr_imgs.append(xr_img)
    y_imgs.append(y_img)
# Verificamos la forma de las imagenes cargadas
print(xl_imgs[0].shape, xr_imgs[0].shape, y_imgs[0].shape)

display_images('figure_1.png', xl_imgs, xr_imgs, y_imgs, rows=3)

# Datasets definition
idx = int(BUFFER_SIZE*.8)

train_xl = tf.data.Dataset.list_files(xl_files[:idx], shuffle=False)
train_xr = tf.data.Dataset.list_files(xr_files[:idx], shuffle=False)
train_y = tf.data.Dataset.list_files(y_files[:idx], shuffle=False)

test_xl = tf.data.Dataset.list_files(xl_files[idx:], shuffle=False)
test_xr = tf.data.Dataset.list_files(xr_files[idx:], shuffle=False)
test_y = tf.data.Dataset.list_files(y_files[idx:], shuffle=False)

train_xy = tf.data.Dataset.zip((train_xl, train_xr, train_y))
train_xy = train_xy.shuffle(buffer_size=idx, reshuffle_each_iteration=True)
train_xy = train_xy.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
train_xy = train_xy.batch(BATCH_SIZE)

test_xy = tf.data.Dataset.zip((test_xl, test_xr, test_y))
test_xy = test_xy.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
test_xy = test_xy.batch(BATCH_SIZE)

for xl, xr, y in train_xy.take(3):
    display_images('figure_2.png', xl, xr, y, rows=3)
    break