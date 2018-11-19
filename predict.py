import nibabel as nib
from importingdata import import_data
import numpy as np
import tensorflow as tf
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import losses
from tensorflow.contrib.keras import layers

# ---------------------------
# prediciting the label for one image and export it to a nii.gz file
# !! just one image can be in test directory!!
# ---------------------------

# global variables
directoryOfFiles = "./data/test/"
directoryToSave = "./data/prediction"


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



# get data
input, _ = import_data(directoryOfFiles)
input = np.reshape(input, [input.shape[0], input.shape[1], 1, input.shape[2]])
print(input.shape)
input = np.moveaxis(input, -1, 0)

# load trained model
print("Loading model")
save_model_path = './temp/newweights.h5'
model = models.load_model(save_model_path, custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_loss': dice_loss})

# predict label for one image
print("Predict")
predict = model.predict(input)
predict = np.moveaxis(predict, 0, -1)
predict = np.squeeze(predict)
print(np.amax(predict))
print(predict.shape)

# export it to a .nii.gz
print("Export to file")
img = nib.Nifti1Image(predict, affine=np.eye(4))  # np.eye(3)
print(img.shape)
filename = directoryToSave + "/prediction-label.nii.gz"
nib.save(img, filename)
