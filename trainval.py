from importingdata import import_data
import numpy as np
import tensorflow as tf
import nibabel as nib
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import losses
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# global variables
# Change to relative path
directoryOfFiles = "./data/train/"
img_shape = (512, 512, 1)
epochs = 50 
steps_per_epoch = 893 


# functions for creating a unet model
def conv_block(input_tensor, num_filters):
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(input_tensor)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    encoder = layers.Conv2D(num_filters, (3, 3), padding='same')(encoder)
    encoder = layers.BatchNormalization()(encoder)
    encoder = layers.Activation('relu')(encoder)
    return encoder


def encoder_block(input_tensor, num_filters):
    encoder = conv_block(input_tensor, num_filters)
    encoder_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(encoder)
    return encoder_pool, encoder


def decoder_block(input_tensor, concat_tensor, num_filters):
    decoder = layers.Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(input_tensor)
    decoder = layers.concatenate([concat_tensor, decoder], axis=-1)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    decoder = layers.Conv2D(num_filters, (3, 3), padding='same')(decoder)
    decoder = layers.BatchNormalization()(decoder)
    decoder = layers.Activation('relu')(decoder)
    return decoder


# create the unet
def create_model(img_shape):
    inputs = layers.Input(shape=img_shape)
    encoder0_pool, encoder0 = encoder_block(inputs, 32)
    encoder1_pool, encoder1 = encoder_block(encoder0_pool, 64)
    encoder2_pool, encoder2 = encoder_block(encoder1_pool, 128)
    encoder3_pool, encoder3 = encoder_block(encoder2_pool, 256)
    encoder4_pool, encoder4 = encoder_block(encoder3_pool, 512)
    center = conv_block(encoder4_pool, 1024)
    decoder4 = decoder_block(center, encoder4, 512)
    decoder3 = decoder_block(decoder4, encoder3, 256)
    decoder2 = decoder_block(decoder3, encoder2, 128)
    decoder1 = decoder_block(decoder2, encoder1, 64)
    decoder0 = decoder_block(decoder1, encoder0, 32)
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0) # change to perceptron?
    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model


# define loss function
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

# Feeds the model fitting function with data
def generator(features, labels):
    batch_size = 3 
    batch_features = np.zeros((batch_size, 512, 512, 1))
    batch_labels = np.zeros((batch_size, 512, 512, 1))
    while True:
        for i in range(batch_size):
            index = np.random.choice(len(features),1)
            batch_features[i] = features[index]
            batch_labels[i] = labels[index]
        yield batch_features, batch_labels


# get data
input, labels = import_data(directoryOfFiles)
input = np.reshape(input, [input.shape[0], input.shape[1], 1, input.shape[2]])
labels = np.reshape(labels, [labels.shape[0], labels.shape[1], 1, labels.shape[2]])

# split data in training and validation
print("Prepare data for training")
input = np.moveaxis(input, -1, 0)
labels = np.moveaxis(labels, -1, 0)
# input = input[:10,:,:,:]
# labels = labels[:10,:,:,:]
x_train, x_val, y_train, y_val = train_test_split(input, labels, test_size=0.1, random_state=42)

# create the model
print("Creating the model")
num_train_examples = x_train.shape[0]
num_val_examples = x_val.shape[0]
print("Number of training examples")
print(num_train_examples)
print("Number of validation examples")
print(num_val_examples)

model = create_model(img_shape)

# Train the model
print("Start training")
save_model_path = "temp/weights.hdf5"
model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_loss])
model.summary()
cp = tf.contrib.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_dice_loss', save_best_only=True,
                                                verbose=1)

#history = model.fit_generator(generator(x_train, y_train), epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data=(x_val, y_val), callbacks=[cp])

history = model.fit(x_train, y_train, epochs=epochs, batch_size=3, validation_data=(x_val, y_val), callbacks=[cp])

dice = history.history['dice_loss']
val_dice = history.history['val_dice_loss']

loss = history.history['loss']
val_loss = history.history['val_loss']
model.save("temp/finalweights.h5")
models.save_model(model, "temp/finalweights2.hdf5")
epochs_range = range(epochs)
plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, dice, label='Training Dice Loss')
plt.plot(epochs_range, val_dice, label='Validation Dice Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Dice Loss')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

plt.show()

predicted = model.predict(x_val, verbose = 1)
print("Prediction Shape: ")
print(predicted.shape)
predicted = np.moveaxis(predicted, 0, -1)
predicted = np.squeeze(predicted)
print("New Shape: ")
print(predicted.shape)

x_val = np.moveaxis(x_val, 0, -1)
x_val = np.squeeze(x_val)
y_val = np.moveaxis(y_val, 0, -1)
y_val = np.squeeze(y_val)

print("Export to file")
saveimg = nib.Nifti1Image(predicted, affine=np.eye(4))
nib.save(saveimg, "./prediction.nii.gz")
saveval = nib.Nifti1Image(x_val, affine=np.eye(4))
nib.save(saveval, "./imageval.nii.gz")
savelab = nib.Nifti1Image(y_val, affine=np.eye(4))
nib.save(savelab, "./val-label.nii.gz")
