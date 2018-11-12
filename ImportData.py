import os
# import numpy as np
# import nibabel as nib

data_directory = "C:/Users/steff/Desktop/LunaProject/data"

counter = 0
files = os.listdir(data_directory)

inputSize = len(files)
inputShape = [512, 512]

for filename in files:
    directory = data_directory + filename
    #currentFile = nib.load(directory)
    directory = directory[:-7]
    directory = directory + "-label.nii.gz"
    # currentLabel = nib.load(directory)
    counter = counter +1

print(input)