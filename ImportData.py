import os
import numpy as np
import nibabel as nib

def import_data(data_directory):
    # imports data from directory and returns numpy arrays input and labels with size [512, 512, num_slices]

    files = os.listdir(data_directory)
    inputSize = len(files)
    inputShape = [512, 512]

    input = np.zeros((inputShape[0], inputShape[1], 0))
    labels = np.zeros((inputShape[0], inputShape[1], 0))

    print("Start importing data, Progress:")
    counter = 0
    for filename in files:
        directory = data_directory + filename
        currentFile = nib.load(directory).get_fdata()
        if directory.find("label") < 0:
            input = np.concatenate((input, currentFile), axis=2)
        else:
            labels = np.concatenate((labels, currentFile), axis=2)
        counter = counter+1
        print(counter/inputSize)

    print("Import done, Input Size:")
    print(input.shape)

    return [input, labels]


data_directory = "C:/Users/steff/Desktop/LunaProject/data-cropped/"
input, labels = import_data(data_directory)
