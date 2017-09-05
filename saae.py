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
print(idxs)
#df = np.array(idxs).astype(int).reshape((-1,10))
#np.savetxt('idx.csv',df, delimiter=',',fmt='%i')

yz_concat = z_nodes + y_nodes
batches = n_train // xx.BATCH_SIZE

# autoencoder units
e_units = [pixels,hid1,hid2]
y_units = [hid2,y_nodes]; z_units = [hid2,z_nodes]; s_units = z_units
d_units = [yz_concat,hid2,hid1,pixels]

#discriminator units
dy_units = [y_nodes,hid2,hid1,1]
dz_units = [z_nodes,hid2,hid1,1]

### DEFINE INPUTS AND WEIGHTS ##################################################

tf.reset_default_graph()

# inputs
X = tf.placeholder(tf.float32, shape=[None, pixels],name='X')
Y = tf.placeholder(tf.float32, shape=[None, y_nodes],name ='Y')

# x dropout
x_drop = tf.placeholder(tf.float32,name='x_drop') 

# learning rates
ae_lr = tf.placeholder(tf.float32,name='ae_lr')
dg_lr = tf.placeholder(tf.float32,name='dg_lr')
ss_lr = tf.placeholder(tf.float32,name='ss_lr')

# autoencoder weights
e_weights, e_list = xx.get_weights(e_units,'e',xx.linear_weights)
y_weights, y_list = xx.get_weights(y_units,'y',xx.linear_weights)
z_weights, z_list = xx.get_weights(z_units,'z',xx.linear_weights)
s_weights, s_list = xx.get_weights(s_units,'s', xx.linear_weights)
d_weights, d_list = xx.get_weights(d_units,'d',xx.linear_weights)

# discriminator weights
dy_weights, dy_list = xx.get_weights(dy_units,'dy',xx.linear_weights)
dz_weights, dz_list = xx.get_weights(dz_units,'dz',xx.linear_weights)

# variable lists
enc_y_list = e_list + y_list; enc_z_list = e_list + z_list + s_list
enc_list = e_list + y_list + z_list + s_list; 
dec_list = d_list 
ae_list = enc_list + dec_list
dis_list = dz_list + dy_list


### DEFINE MODEL ###############################################################

# encoder
x_dropout = tf.nn.dropout(X, x_drop)
representation = xx.encoder_full(x_dropout, e_weights)

# y reparamterisation trick
y_x_logits = xx.y_latent(representation,y_weights)
y_x = tf.nn.softmax(y_x_logits)
y_gum = xx.y_reparameterize(y_x_logits,cfy)

# z reparamterisation trick
z_x_logits = xx.z_latent(representation,z_weights)
s_x_logits = xx.s_latent(representation,s_weights)
zs_x_logits = xx.z_reparameterize(z_x_logits,s_x_logits,cfz)

### decoder
yz_x = tf.concat([y_gum,zs_x_logits],1)
reconstruction, reconstruction_logits = xx.decoder_full(yz_x,d_weights)

# y discriminator
y_real = xx.random_categorical(y_nodes)
dy_real, dy_real_logits = xx.D_y(y_real, dy_weights)
dy_fake, dy_fake_logits = xx.D_y(y_gum, dy_weights) 

# z discriminator
z_real = xx.random_gaussian_laplace(z_noise,zs_x_logits)
dz_real, dz_real_logits = xx.D_z(z_real, dz_weights)
dz_fake, dz_fake_logits = xx.D_z(zs_x_logits, dz_weights) 


### TEST #######################################################################

# encoder ## test vs normal X
X_test = tf.placeholder(tf.float32, shape=[None, pixels],name='X_test')
representation_t = xx.encoder_full(X_test, e_weights)

y_x_logits_t = xx.y_latent(representation_t,y_weights)
y_x_t = tf.nn.softmax(y_x_logits_t)
z_x_logits_t= xx.z_latent(representation_t,z_weights)
yz_x_t= tf.concat([y_x_t,z_x_logits_t],1)

# decoder 
reconstruction_t, reconstruction_logits_t = xx.decoder_full(yz_x_t,d_weights)

# evaluation
SS_score = xx.evaluation(y_x_logits, Y)
SS_score_t = xx.evaluation(y_x_logits_t, Y)


### DEFINE LOSSES ##############################################################

#reconstruction loss
if obj == 'MSE':
	R_loss = 0.5*tf.reduce_mean(tf.reduce_sum(tf.pow(X - reconstruction, 2),1))
else:
	R_loss = 0.5*tf.reduce_mean(tf.reduce_sum(tf.abs(X - reconstruction),1))

#discriminator loss
DZ_loss = xx.discriminator_loss(dz_real_logits,dz_fake_logits)
DY_loss = xx.discriminator_loss(dy_real_logits,dy_fake_logits)
D_loss = DZ_loss + DY_loss

#generator loss
GZ_loss = xx.generator_loss(dz_fake_logits)
GY_loss = xx.generator_loss(dy_fake_logits)
G_loss = GZ_loss + GY_loss

# semi-supervised loss
C_loss_ = tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=y_x_logits)
C_loss = tf.reduce_mean(C_loss_)


### DEFINE OPTIMIZATIONS ####################################################### 

R_solver = tf.train.MomentumOptimizer(ae_lr,0.9).minimize(R_loss, \
	var_list=ae_list)
D_solver = tf.train.MomentumOptimizer(dg_lr,0.1).minimize(D_loss, \
	var_list=dis_list)
G_solver = tf.train.MomentumOptimizer(dg_lr,0.1).minimize(G_loss, \
	var_list=enc_list)
C_solver = tf.train.MomentumOptimizer(ss_lr,0.9).minimize(C_loss, \
	var_list=enc_y_list)


### INITIALIZATIONS ############################################################

# labeled batch
batch_x_l, batch_y_l = train_images[idxs], train_labels[idxs]

#results
results_train = []; results_test = []; recon_train = []; recon_test = []
best_accuracy = 0.0

init = tf.global_variables_initializer()
saver = tf.train.Saver()


### TRAINING ###################################################################

with tf.Session() as sess:

	sess.run(init)

	for epoch in range(epochs):

		if epoch == 50: 
			rlr = rlr/10		
			dglr = dglr/10
			sslr = sslr/10

		for i in range(batches):

			batch_x_u, _ = xx.get_data(i,xx.BATCH_SIZE,train_images,train_labels)

			_, r = sess.run([R_solver, R_loss], feed_dict=\
				{X: batch_x_u, ae_lr:rlr, x_drop:drop})

			_, d = sess.run([D_solver, D_loss], feed_dict=\
				{X: batch_x_u, dg_lr:dglr, x_drop:drop})

			_, g = sess.run([G_solver, G_loss], feed_dict=\
				{X: batch_x_u, dg_lr:dglr, x_drop:drop})

			_, c = sess.run([C_solver, C_loss], feed_dict=\
				{X: batch_x_l, Y:batch_y_l,ss_lr:sslr, x_drop:drop})

		aex_test, acc, acc_t, yxlogt, zxlogt, dyr, dyf, dzr, dzf = \
		 sess.run([R_loss,SS_score,SS_score_t,y_x,zs_x_logits,dy_real,dy_fake,dz_real,dz_fake], \
			feed_dict={X: test_images, X_test: test_images, Y:test_labels, x_drop:1.0})

		print('Y mean: {} / std: {}'.format(np.mean(yxlogt),np.std(yxlogt)))
		print('Z mean: {} / std: {}'.format(np.mean(zxlogt),np.std(zxlogt)))
		print('dyr:',dyr.mean(),'dyf:',dyf.mean(),'dzr:',dzr.mean(), 'dzf:',dzf.mean())
		if acc_t > best_accuracy and epoch > 50:

			xx.save_model(saver,sess,script,name,epoch)
			best_accuracy = acc_t

		print('Train = Epoch:{:04d}, Recon: {}, D: {}, G: {}, class: {}'.format(epoch+1,r,d,g,c))
		print('Recon: {} / Acc Train: {} / Acc Test: {}'.format(aex_test,acc,acc_t))

		results_train.append(acc); results_test.append(acc_t); recon_train.append(r); recon_test.append(aex_test); 

		duration = time.time() - start; print('time',duration); start = time.time()

		#xx.save_examples(sess, reconstruction_logits, yz_x, script, name, epoch, z_nodes)
		#xx.save_reconstructions(sess, reconstruction_logits, X, x_drop, test_images, pixels, script, name, epoch)


import datetime			
print([name,datetime.datetime.now()])			
print(results_train)
print(results_test)
print(recon_train)
print(recon_test)
duration = time.time() - start
print(duration)

duration_end = time.time() - start_begin
print(duration_end)

import csv
with open("{}.csv".format(script),'a',newline='') as wr:
	writer = csv.writer(wr, delimiter = ',')
	writer.writerows([['#####',name,datetime.datetime.now()]]) 
	writer.writerows([results_train]) 
	writer.writerows([results_test]) 
	writer.writerows([recon_train]) 
	writer.writerows([recon_test]) 

with open("all.csv",'a',newline='') as wr:
	writer = csv.writer(wr, delimiter = ',')
	writer.writerows([[max(results_test),results_test[-1],np.array(results_test[-25:]).std(),\
		max(results_train),results_train[-1],recon_test[-1],np.mean(yxlogt),np.std(yxlogt),\
		np.mean(zxlogt),np.std(zxlogt),dyr.mean(),dyf.mean(),dzr.mean(),dzf.mean(),name,datetime.datetime.now()]]) 

import csv
with open("{}_{}_{}.csv".format(dataset,script,n_train),'a',newline='') as wr:
	writer = csv.writer(wr, delimiter = ',')
	writer.writerows([[max(results_test),results_test[-1],np.array(results_test[-25:]).std(),\
		max(results_train),results_train[-1],recon_test[-1],np.mean(yxlogt),np.std(yxlogt),\
		np.mean(zxlogt),np.std(zxlogt),dyr.mean(),dyf.mean(),dzr.mean(),dzf.mean(),name,datetime.datetime.now()]]) 



import matplotlib.pyplot as plt
#print(4444444444444444,batch_y_l.sum(0))
'''
for i in range(100):
	plt.subplot(10,10,i+1)
	plt.axis('off')
	plt.imshow(batch_x_l[i,:].reshape((28,28)))
	print(batch_y_l[i])
model_path = 'D:/randoms/{}'.format(script)
if not os.path.exists(model_path):
	os.makedirs(model_path)
plt.savefig('{}/{}_{}_gen.jpg'.format(model_path, i, batch_y_l[i]))
'''
