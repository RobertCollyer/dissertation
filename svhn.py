
import numpy as np
import pickle
import os
import scipy.io as sio
import matplotlib.pyplot as plt


root = "C:/Users/Rob/data/"
train_data = sio.loadmat(os.path.join(root,'train_32x32.mat'))
test_data = sio.loadmat(os.path.join(root,'test_32x32.mat'))
extra_data = sio.loadmat(os.path.join(root,'extra_32x32.mat'))

# access to the dict
x_train = train_data['X'].transpose(3,0,1,2).astype(np.float32); y_train = train_data['y'].squeeze()#.astype(np.float32)
x_test = test_data['X'].transpose(3,0,1,2).astype(np.float32); y_test = test_data['y'].squeeze()
x_extra = extra_data['X'].transpose(3,0,1,2).astype(np.float32); y_extra = extra_data['y'].squeeze()
y_train[y_train == 10] = 0; y_train_ohe = np.eye(10)[y_train]
y_test[y_test == 10] = 0; y_test_ohe = np.eye(10)[y_test]
y_extra[y_extra == 10] = 0; y_extra_ohe = np.eye(10)[y_extra]
print(x_train.shape)
print(x_train.dtype)


def rgb2gray(rgb):
    return np.dot(rgb[:,:,:,:3], [0.299, 0.587, 0.114])

x_train = rgb2gray(x_train)/255.0
x_test = rgb2gray(x_test)/255.0
x_extra = rgb2gray(x_extra)/255.0


import pickle
# pickle the data
pickle_file = '../../../data/svhn.pickle'

f = open(pickle_file, 'wb')
save = {
    'train_images': x_train,
    'train_labels': y_train,
    'train_labels_ohe': y_train_ohe,
    'test_images': x_test,
    'test_labels': y_test,
    'test_labels_ohe': y_test_ohe,
    'extra_images': x_extra,
    'extra_labels': y_extra,
    'extra_labels_ohe': y_extra_ohe
}
pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
f.close()
print('complete')


'''
print('abcccccccc',x_train.shape)


n = 5
for i in range(n**2):
    plt.subplot(n,n,i+1)
    plt.title(y_train[i])
    plt.imshow(x_train[i,:,:])

plt.show()

print(x_train.shape)
print(x_test.shape)
print(x_extra.shape)
print(y_train_ohe.shape)
print(y_test_ohe.shape)
print(y_extra_ohe.shape)
'''

