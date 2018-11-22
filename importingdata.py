import os
import numpy as np
import nibabel as nib

# ---------------------------
# importing data from directory to a numpy array
# ---------------------------

def import_data(data_directory):
    # imports data from directory and returns numpy arrays input and labels with size [512, 512, num_slices]

    files = os.listdir(data_directory)
    input_size = len(files)
    input_shape = [512, 512]

    input_matrix = np.zeros((input_shape[0], input_shape[1], 0))
    labels_matrix = np.zeros((input_shape[0], input_shape[1], 0))

    print("Start importing data, Progress:")
    counter = 0
    for filename in files:
        directory = data_directory + "/" + filename
        if directory.find("normalized") >= 0:
            current_file = nib.load(directory).get_fdata()
            input_matrix = np.concatenate((input_matrix, current_file), axis=2)
        if directory.find("label") >= 0:
            current_file = nib.load(directory).get_fdata()
            labels_matrix = np.concatenate((labels_matrix, current_file), axis=2)
        counter = counter + 1
        print(counter / input_size)

    print("Import done, Input Size:")
    print(input_matrix.shape)
    print(labels_matrix.shape)
    #labels_matrix[labels_matrix == 4] = 3
    #labels_matrix[labels_matrix > 4] = 0
    #labels_matrix[labels_matrix == 3] = 1
    return [input_matrix, labels_matrix]
