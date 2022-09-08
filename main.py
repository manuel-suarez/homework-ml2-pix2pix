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

# Model architecture definition
def downsample(filters, size, apply_batchnorm=True):
    '''
    Bloque de codificación (down-sampling)
    '''
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters           = filters,
                                      kernel_size       = size,
                                      strides           = 2,
                                      padding           = 'same',
                                      kernel_initializer= initializer,
                                      use_bias          = False))

    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result

down_model  = downsample(3, 4)
down_result = down_model(tf.expand_dims(xl_img, 0))
print("Downsample shape result: ", down_result.shape)

def upsample(filters, size, apply_dropout=False):
    '''
    Bloque de decodicación (up-sampling)
    '''
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters           = filters,
                                               kernel_size       = size,
                                               strides           = 2,
                                               padding           = 'same',
                                               kernel_initializer= initializer,
                                               use_bias          = False))

    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

up_model = upsample(3, 4)
up_result = up_model(down_result)
print("Upsample shape result: ", up_result.shape)

# Generator architecture definition
def Generator():
    '''
    UNet
    '''

    # Capas que la componen
    xl_input = tf.keras.layers.Input(shape=INPUT_DIM)
    xr_input = tf.keras.layers.Input(shape=INPUT_DIM)
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),  # (batch_size, 128, 128, 64)
        downsample(128, 4),  # (batch_size, 64,  64,  128)
        downsample(256, 4),  # (batch_size, 32,  32,  256)
        downsample(512, 4),  # (batch_size, 16,  16,  512)
        downsample(512, 4),  # (batch_size, 8,   8,   512)
        downsample(512, 4),  # (batch_size, 4,   4,   512)
        downsample(512, 4),  # (batch_size, 2,   2,   512)
        downsample(512, 4),  # (batch_size, 1,   1,   512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True),  # (batch_size, 2,    2,  1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 4,    4,  1024)
        upsample(512, 4, apply_dropout=True),  # (batch_size, 8,    8,  1024)
        upsample(512, 4),  # (batch_size, 16,   16, 1024)
        upsample(256, 4),  # (batch_size, 32,   32, 512)
        upsample(128, 4),  # (batch_size, 64,   64, 256)
        upsample(64, 4),  # (batch_size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(OUTPUT_CHANNELS, 4,
                                           strides=2,
                                           padding='same',
                                           kernel_initializer=initializer,
                                           activation='tanh')  # (batch_size, 256, 256, 3)

    # pipeline de procesamiento
    x = tf.keras.layers.Concatenate(axis=3)([xl_input, xr_input])
    # Codificador
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)  # se agrega a una lista la salida cada vez que se desciende en el generador
    skips = reversed(skips[:-1])
    # Decodificador
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=[xl_input, xr_input], outputs=x)
generator = Generator()

# Discriminator
def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    inp_l = tf.keras.layers.Input(shape=[256, 256, 3], name='input_left_image')
    inp_r = tf.keras.layers.Input(shape=[256, 256, 3], name='input_right_image')
    tar = tf.keras.layers.Input(shape=[256, 256, 1], name='target_image')
    x = tf.keras.layers.concatenate(
        [inp_l, inp_r, tar])  # (batch_size, 256, 256, 3 channels left + 3 channels right + disparity channel)

    down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)
    down2 = downsample(128, 4)(down1)  # (batch_size, 64, 64, 128)
    down3 = downsample(256, 4)(down2)  # (batch_size, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

    zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)
    last = tf.keras.layers.Conv2D(filters=1,
                                  kernel_size=4,
                                  strides=1,
                                  kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

    return tf.keras.Model(inputs=[inp_l, inp_r, tar], outputs=last)
discriminator = Discriminator()

# Discriminator evaluation result
disc_out = discriminator([xl_img[tf.newaxis, ...], xr_img[tf.newaxis, ...], gen_output], training=False)

fig, ax = plt.subplots(1, 3, figsize=(8, 4))
ax[0].imshow((xl_img+1)/2)
ax[1].imshow((xr_img+1)/2)
ax[2].imshow(disc_out[0, ..., -1]*200, vmin=-20, vmax=20, cmap='RdBu_r')  #*100
plt.tight_layout()
plt.savefig('figure_3.png')