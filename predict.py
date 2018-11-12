import os
import functools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.grid'] = False
mpl.rcParams['figure.figsize'] = (12,12)
import matplotlib.image as mpimg
import pandas as pd
import tensorflow as tf
import tensorflow.contrib as tfcontrib
from tensorflow.python.keras import losses
from tensorflow.python.keras import models
from tensorflow.python.keras import backend as K
from PIL import Image
#import scipy.misc
import imageio

project_name = 'temporal-bone'

img_dir = os.path.join(project_name, "test")
label_dir = os.path.join(project_name, "test_masks")

df_test = pd.read_csv(os.path.join(project_name, 'test_masks.csv'))
ids_test = df_test['img'].map(lambda s: s.split('.')[0])

x_test_filenames = []
y_test_filenames = []
for img_id in ids_test:
    x_test_filenames.append(os.path.join(img_dir, "{}.jpg".format(img_id)))
    y_test_filenames.append(os.path.join(label_dir, "{}_mask.gif".format(img_id)))

num_test_examples = len(x_test_filenames)
print("Number of test examples: {}".format(num_test_examples))
print(x_test_filenames[:10])

img_shape = (256, 256, 1)
batch_size = 1 

def _process_pathnames(fname, label_path):
  # We map this function onto each pathname pair  
  img_str = tf.read_file(fname)
  img = tf.image.decode_jpeg(img_str, channels=1)

  label_img_str = tf.read_file(label_path)
  # These are gif images so they return as (num_frames, h, w, c)
  label_img = tf.image.decode_gif(label_img_str)[0]
  # The label image should only have values of 1 or 0, indicating pixel wise
  # object (car) or not (background). We take the first channel only. 
  label_img = label_img[:, :, 0]
  label_img = tf.expand_dims(label_img, axis=-1)
  return img, label_img

def _augment(img,
             label_img,
             resize=None,  # Resize the image to some size e.g. [256, 256]
             scale=1,  # Scale image e.g. 1 / 255.
             hue_delta=0,  # Adjust the hue of an RGB image by random factor
             horizontal_flip=False,  # Random left right flip,
             width_shift_range=0,  # Randomly translate the image horizontally
             height_shift_range=0):  # Randomly translate the image vertically 
  if resize is not None:
    # Resize both images
    label_img = tf.image.resize_images(label_img, resize)
    img = tf.image.resize_images(img, resize)
  
  #img, label_img = flip_img(horizontal_flip, img, label_img)
  #img, label_img = shift_img(img, label_img, width_shift_range, height_shift_range)
  label_img = tf.to_float(label_img) * scale
  img = tf.to_float(img) * scale 
  return img, label_img

def get_baseline_dataset(filenames, 
                         labels,
                         preproc_fn=functools.partial(_augment),
                         threads=5, 
                         batch_size=batch_size,
                         shuffle=False):           
  num_x = len(filenames)
  # Create a dataset from the filenames and labels
  dataset = tf.data.Dataset.from_tensor_slices((filenames, labels))
  # Map our preprocessing function to every element in our dataset, taking
  # advantage of multithreading
  dataset = dataset.map(_process_pathnames, num_parallel_calls=threads)
  if preproc_fn.keywords is not None and 'resize' not in preproc_fn.keywords:
    assert batch_size == 1, "Batching images must be of the same size"

  dataset = dataset.map(preproc_fn, num_parallel_calls=threads)
  
  if shuffle:
    dataset = dataset.shuffle(num_x)
  
  
  # It's necessary to repeat our data for all epochs 
  dataset = dataset.repeat().batch(batch_size)
  return dataset

test_cfg = {
    'resize': [img_shape[0], img_shape[1]],
    'scale': 1 / 255.,
}
test_preprocessing_fn = functools.partial(_augment, **test_cfg)

test_ds = get_baseline_dataset(x_test_filenames, y_test_filenames, preproc_fn=test_preprocessing_fn, batch_size=batch_size, shuffle=False)

def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def bce_dice_loss(y_true, y_pred):
    loss = losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

save_model_path = './tmp/weights.hdf5'
model = models.load_model(save_model_path, custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_loss': dice_loss})

data_aug_iter = test_ds.make_one_shot_iterator()
next_element = data_aug_iter.get_next()

#plt.figure(figsize=(10, 20))
for i in range(num_test_examples):
  batch_of_imgs, label = tf.keras.backend.get_session().run(next_element)
  img = batch_of_imgs[0]
  predicted_label = model.predict(batch_of_imgs)[0]

  #plt.subplot(5, 3, 3 * i + 1)
  #plt.imshow(img)
  #plt.title("Input image")
  
  #plt.subplot(5, 3, 3 * i + 2)
  #plt.imshow(label[0, :, :, 0])
  #plt.title("Actual Mask")
  #plt.subplot(5, 3, 3 * i + 3)
  #plt.imshow(predicted_label[:, :, 0])
  #im = Image.fromarray(np.uint8(cm.gist_earth(predicted_label[:, :, 0])*255))
  imgoutfile = "outputimg/img{}.png".format(i)
  #scipy.misc.imsave(imgoutfile, img)
  imageio.imwrite(imgoutfile, img)
  labeloutfile = "outputlabel/img{}.png".format(i)
  #scipy.misc.imsave(labeloutfile, label[0, :, :, 0])
  imageio.imwrite(labeloutfile, label[0, :, :, 0]*255)
  predoutfile = "predlabel/img{}.png".format(i)
  #scipy.misc.imsave(predoutfile, predicted_label[:, :, 0])
  imageio.imwrite(predoutfile, predicted_label[:, :, 0]*255)
  #plt.title("Predicted Mask")
#plt.suptitle("Examples of Input Image, Label, and Prediction")
#plt.show()
