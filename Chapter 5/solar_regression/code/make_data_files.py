import numpy as np
import h5py


# Importing the data 

train_data = h5py.File('../data/train.h5')
test_data = h5py.File('../data/test.h5')

# Inputs
x_train = train_data['data'] 
x_test = test_data['data']
# Labels/outputs 
y_train = train_data['label']
y_test = test_data['label']

np.save('../data/x_train.npy', x_train)
np.save('../data/x_test.npy', x_test)
np.save('../data/y_train.npy', y_train)
np.save('../data/y_test.npy', y_test)