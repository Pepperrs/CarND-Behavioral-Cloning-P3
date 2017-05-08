import random
import os
import csv
import cv2
import numpy as np
import sys # required for floyd argument
import sklearn


print("Started Behavioural Cloning Model!")

lines = []
floyd = 0
data_path = '../CarND-P3-Data/'

# enable floyd mode for different datapath
if "floyd" in sys.argv:
    # floyd = 1
    data_path = '/input/'
    print("floydmode activated!")

# import the driving log as csv
with open(data_path + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)      # skip the first line
    for line in reader: 
        lines.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(lines, test_size=0.2)


# Generator for memory efficient data loading
correction = 0.2
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            # add all images of the sample to the set
            for batch_sample in batch_samples:
                center_image = cv2.imread(data_path + 'IMG/'+batch_sample[0].split('/')[-1])
                left_image = cv2.imread(data_path + 'IMG/'+batch_sample[1].split('/')[-1])
                right_image = cv2.imread(data_path + 'IMG/'+batch_sample[2].split('/')[-1])
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                images.append(left_image)
                angles.append(center_angle+correction)
                images.append(right_image)
                angles.append(center_angle-correction)


                images.append(np.fliplr(center_image))
                angles.append(-center_angle)
                images.append(np.fliplr(left_image))
                angles.append(-(center_angle+correction))
                images.append(np.fliplr(right_image))
                angles.append(-(center_angle-correction))



            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)




print("Loaded Data!")

# todo: split data


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Conv2D, Lambda, Cropping2D

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)



model = Sequential()

# input shape for normalized images
#model.add(Flatten(input_shape=(160,320,3)))

# normalization
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))

#model.add(Flatten())
model.add(Cropping2D(cropping=((50,20),(0,0))))

# Convolution 24@31x98
model.add(Conv2D(24, 5, 5, subsample=(2,2), activation = "relu"))

# Convolution 36@14x47
model.add(Conv2D(32, 5, 5, subsample=(2,2), activation = "relu"))


# Convolutioin 48@5x22
model.add(Conv2D(48, 3, 3, subsample=(2,2), activation = "relu"))


# Convolution 64@3x20
model.add(Conv2D(60, 3, 3, subsample=(2,2), activation = "relu"))


# Convolution 64@1x18
model.add(Conv2D(60, 3, 3, subsample=(2,2), activation = "relu"))


# Flatten
model.add(Flatten())

# Fully connected 1164 neurons
model.add(Dense(1164))


# Fully connected 100 neurons
model.add(Dense(100))


# Fully connected 50 neurons
model.add(Dense(50))


# Fully connected 10 neurons (?)
model.add(Dense(10))


# output vehicle control
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator, samples_per_epoch=len(train_samples*6), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=3)


model.save('model.h5')
