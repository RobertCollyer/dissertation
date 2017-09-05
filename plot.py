from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import pickle
import time
import os
import scipy.misc
import utils as xx
from sys import argv
start_begin = time.time()
start = time.time()
#np.set_printoptions(threshold=np.inf)

script, hid1, hid2, z_nodes, y_nodes, obj, z_noise, rlr, dglr, sslr, drop,\
 cfy ,cfz, dataset, decrease, n_train, n_test, epochs, ss_labels, seed, rep = argv

hid1 = int(hid1); print('hid1',hid2)
hid2 = int(hid2); print('hid2',hid2)
z_nodes = int(z_nodes); print('z_nodes',z_nodes)
y_nodes = int(y_nodes); print('y_nodes',y_nodes)
obj = obj; print('obj',obj)
z_noise = z_noise; print('z_noise',z_noise)
rlr = float(rlr); print('rlr',rlr)
dglr = float(dglr); print('dglr',dglr)
sslr = float(sslr); print('sslr',sslr)
drop = float(drop); print('drop',drop)
cfy = float(cfy); print('cfy',cfy)
cfz = float(cfz); print('cfz',cfz)
dataset = dataset; print('dataset',dataset)
decrease = xx.str_to_bool(decrease); print('decrease',decrease)
n_train = int(n_train); print('n_train',n_train)
n_test = int(n_test); print('n_test',n_test)
epochs = int(epochs); print('epochs',epochs)
ss_labels = int(ss_labels); print('ss_labels',ss_labels)
seed = int(seed); print('seed',seed)
rep = int(rep); print('rep',rep)

name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.\
format(script, hid1, hid2, z_nodes, y_nodes, obj, z_noise, rlr, dglr, sslr, \
	drop, cfy, cfz, dataset, decrease, n_train, n_test, epochs, ss_labels, seed, rep)
print(name)

train_images, train_labels, test_images, test_labels, idxs, pixels = \
xx.get_dataset(dataset, decrease, n_train, n_test, ss_labels, seed)
#print(idxs)


yz_concat = z_nodes + y_nodes
batches = n_train // xx.BATCH_SIZE

# autoencoder units
e_units = [pixels,hid1,hid2]
y_units = [hid2,y_nodes]; z_units = [hid2,z_nodes]; s_units = z_units
d_units = [yz_concat,hid2,hid1,pixels]


### DEFINE INPUTS AND WEIGHTS ##################################################

tf.reset_default_graph()

# inputs
X = tf.placeholder(tf.float32, shape=[None, pixels],name='X')
Y = tf.placeholder(tf.float32, shape=[None, y_nodes],name ='Y')

# x dropout
x_drop = tf.placeholder(tf.float32,name='x_drop') 

# autoencoder weights
e_weights, e_list = xx.get_weights(e_units,'e',xx.linear_weights)
y_weights, y_list = xx.get_weights(y_units,'y',xx.linear_weights)
z_weights, z_list = xx.get_weights(z_units,'z',xx.linear_weights)
#if script != 'aae.py'
	#s_weights, s_list = xx.get_weights(s_units,'s', xx.linear_weights)
	#d_weights, d_list = xx.get_weights(d_units,'d',xx.linear_weights)


### DEFINE MODEL ###############################################################

# encoder
x_dropout = tf.nn.dropout(X, x_drop)
representation = xx.encoder_full(x_dropout, e_weights)

# y reparamterisation trick
y_x_logits = xx.y_latent(representation,y_weights)
#y_x = tf.nn.softmax(y_x_logits)
#y_gum = xx.y_reparameterize(y_x_logits,cfy)

# z reparamterisation trick
z_x_logits = xx.z_latent(representation,z_weights)
#s_x_logits = xx.s_latent(representation,s_weights)
#zs_x_logits = xx.z_reparameterize(z_x_logits,s_x_logits,cfz)


# evaluation
SS_score = xx.evaluation(y_x_logits, Y)

### INITIALIZATIONS ############################################################
MODEL = 'saae.py_1000_1000_10_10_MSE_GAUSS_0.001_0.001_0.003_1.0_0.3_0.1_svhn_True_50000_10000_100_2000_127_13_53.ckpt'
checkpoint_file = os.path.join('../models', MODEL)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

### TRAINING ###################################################################

with tf.Session() as sess:

	sess.run(init)
	saver.restore(sess, checkpoint_file)

	acc, y_logs, z_logs = \
	 sess.run([SS_score, y_x_logits, z_x_logits], \
		feed_dict={X: test_images, Y:test_labels, x_drop:1.0})

	print('Acc Train: {}'.format(acc))

y_logs = np.argmax(y_logs,1).reshape((-1,1))
yz_logs = np.concatenate((y_logs,z_logs),1)
print('yz_logs',yz_logs.shape)

n=10000

from sklearn.manifold import TSNE
#X_embedded = TSNE(n_components=2).fit_transform(yz_logs[:n,:])
X_embedded = TSNE(n_components=2).fit_transform(z_logs[:n,:])
print(X_embedded.shape)
labels = np.argmax(test_labels,1)

print(labels.shape)
import pylab
colors = [int(i % 10) for i in labels[:n]]
import matplotlib.pyplot as plt
plt.scatter(X_embedded[:,0], X_embedded[:,1], c=colors, marker=".")
plt.colorbar()
plt.title('GSVAE - z only')
plt.xlabel('Embedded axis 1')
plt.ylabel('Embedded axis 2')
plt.show()

duration_end = time.time() - start_begin
print(duration_end)





