from importingdata import import_data
import numpy as np
import tensorflow as tf
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import losses
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# global variables
# Change to relative path
directoryOfFiles = "./data/train/"
img_shape = (512, 512, 1)
batch_size = 2
epochs = 2

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
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(decoder0)
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

# get data
input, labels = import_data(directoryOfFiles)
input = np.reshape(input, [input.shape[0], input.shape[1], 1, input.shape[2]])
labels = np.reshape(labels, [labels.shape[0], labels.shape[1], 1, labels.shape[2]])

# split data in training and validation
print("Prepare data for training")
input = np.moveaxis(input, -1, 0)
labels = np.moveaxis(labels, -1, 0)
x_train, x_val, y_train, y_val = train_test_split(input, labels, test_size=0.1)
# x_train = np.swapaxes(x_train, 0, 3)
# x_val = np.swapaxes(x_val, 0, 3)
# y_train = np.swapaxes(y_train, 0, 3)
# y_val = np.swapaxes(y_val, 0, 3)

train_ds = tf.contrib.data.Dataset.from_tensor_slices((x_train, y_train))
val_ds = tf.contrib.data.Dataset.from_tensor_slices((x_val, y_val))

# create the model
print("Creating the model")
num_train_examples = x_train.shape[0]
num_val_examples = x_val.shape[0]

model = create_model(img_shape)

# Train the model
print("Start training")
save_model_path = "../temp/weights.hdf5"
model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_loss])
model.summary()
cp = tf.contrib.keras.callbacks.ModelCheckpoint(filepath=save_model_path, monitor='val_dice_loss', save_best_only=True, verbose=1)

history = model.fit(x_train, y_train,
                   epochs=epochs,
                   callbacks=[cp])

dice = history.history['dice_loss']
val_dice = history.history['val_dice_loss']

loss = history.history['loss']
val_loss = history.history['val_loss']

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
