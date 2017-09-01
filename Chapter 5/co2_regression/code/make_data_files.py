import numpy as np
import h5py
import random


# Importing the data 

train_data = h5py.File('../data/train.h5')
test_data = h5py.File('../data/test.h5')

# Inputs
x_train = train_data['data'] 
x_test = test_data['data']
x = np.concatenate([x_train, x_test], 0)

N = np.shape(x)[0]

np.random.seed(123)
# Labels/outputs 
y_train = train_data['label']
y_test = test_data['label']
y = np.concatenate([y_train, y_test], 0)

# Shuffle the data
shf = random.sample(range(N), N)
x = x[shf ,:]
y = y[shf, :]

# Split data into 2/3 train 1/3 test
x_test = x[0:int(N/3), :]
x_train = x[int(N/3):, :]
y_test = y[0:int(N/3), :]
y_train = y[int(N/3):, :]

np.save('../data/x_train.npy', x_train)
np.save('../data/x_test.npy', x_test)
np.save('../data/y_train.npy', y_train)
np.save('../data/y_test.npy', y_test)