import csv
import cv2
import numpy as np
import sys # required for floyd argument

# import the driving log as csv
lines = []
floyd = 0
data_path = '../CarND-P3-Data/'

# enable floyd mode for different datapath
if "floyd" in sys.argv:
    # floyd = 1
    data_path = '/input/'
    print("floydmode activated!")


with open(data_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)      # skip the first line
    for line in reader: 
        lines.append(line)


# import all images from the available data 
# and append them to arrays for easier use
images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = data_path + 'IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)


# convert the image and measurements array
# to serve as training data
X_train = np.array(images)
Y_train = np.array(measurements)

# todo: split data


from keras.models import Seqential
from keras.layers import Flatten, Dense
