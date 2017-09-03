import numpy as np

x_train = np.load('../data/x_train.npy')
x_test = np.load('../data/x_test.npy')

print(np.shape(x_train))
print(np.shape(x_test))

print('[ ', np.min(x_train), ' , ', np.max(x_train), ' ]')
print('[ ', np.min(x_test), ' , ', np.max(x_test), ' ]')