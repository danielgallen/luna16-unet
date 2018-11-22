import os
import nibabel as nib
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

# imports data from directory and returns numpy arrays input and labels with size [512, 512, num_slices]

files = os.listdir(directoryOfFiles)
input_size = len(files)
input_shape = [512, 512]

input_matrix = np.zeros((input_shape[0], input_shape[1], 0))
labels_matrix = np.zeros((input_shape[0], input_shape[1], 0))

print("Start importing data, Progress:")
counter = 0
directory = directoryOfFiles + "/" + files[0]
if directory.find("normalized") >= 0:
    current_image = nib.load(directory).get_fdata()
    input_matrix = np.concatenate((input_matrix, current_image), axis=2)
#if directory.find("label") >= 0:
    #current_file = nib.load(directory).get_fdata()
    #labels_matrix = np.concatenate((labels_matrix, current_file), axis=2)
#counter = counter + 1
#print(counter / input_size)

print("Import done, Input Size:")
print(input_matrix.shape)
targetaffine = nib.load(directory).affine
#print(labels_matrix.shape)

# get data
#input, _ = import_data(directoryOfFiles)
input = input_matrix
input = np.reshape(input, [input.shape[0], input.shape[1], 1, input.shape[2]])
print(input.shape)
input = np.moveaxis(input, -1, 0)
#input = input[:2,:,:,:]

# load trained model
print("Loading model")
save_model_path = './temp/finalweights2.hdf5'
model = models.load_model(save_model_path, custom_objects={'bce_dice_loss': bce_dice_loss, 'dice_loss': dice_loss})

# predict label for one image
print("Predict")
predict = model.predict(input)
predict = np.moveaxis(predict, 0, -1)
predict = np.squeeze(predict)
print(np.amax(predict))
print(predict.shape)
#predictlabel = predict
#predictlabel[predictlabel > 0.7] = 1
#predictlabel[predictlabel < 0.9] = 0

# export it to a .nii.gz
print("Export to file")
img = nib.Nifti1Image(predict, affine=targetaffine)  # np.eye(3)
#labelimg = nib.Nifti1Image(predictlabel, affine=np.eye(4))
print(img.shape)
filename = directoryToSave + "/prediction.nii.gz"
#labelfilename = directoryToSave + "/prediction-label.nii.gz"
nib.save(img, filename)
#nib.save(labelimg, labelfilename)
