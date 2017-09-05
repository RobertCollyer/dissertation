import tensorflow as tf
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pickle
import os
import indexes as index

### CONSTANTS AND HYPER-PARAMETERS ###
###############################################################################


BATCH_SIZE = 100
LEARNING_RATE = 0.001
CLASSES = 10
TINY = 1e-20
EXAMPLES = 10
DISPLAY = 1
#C_PATCH = 3; C_STRIDE = 2; C_PAD = 'SAME'
#STRIDES=[1, C_STRIDE, C_STRIDE, 1]


def sample_gumbel(shape, eps=1e-20): 

  U = tf.random_uniform(shape,minval=0,maxval=1)

  return -tf.log(-tf.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature): 

  y = logits + sample_gumbel(tf.shape(logits))

  return tf.nn.softmax( y / temperature)

def gumbel_softmax(logits, temperature, hard=False):

  y = gumbel_softmax_sample(logits, temperature)

  if hard:
    k = tf.shape(logits)[-1]
    y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,1,keep_dims=True)),y.dtype)
    y = tf.stop_gradient(y_hard - y) + y

  return y

def random_categorical(y_nodes):

	indexes = tf.squeeze(tf.random_uniform((1,BATCH_SIZE), minval=0, maxval=y_nodes, dtype=tf.int32))
	return tf.one_hot(indexes,y_nodes)

def random_gaussian_laplace(z_noise,zs_x_logits):

	if z_noise == 'GAUSS':
		z_real = tf.random_normal(tf.shape(zs_x_logits))
	else:
		z_real = tf.contrib.distributions.Laplace(0.0,1.0).sample(tf.shape(zs_x_logits))

	return z_real


def linear_weights(w,u1,u2):

	return tf.get_variable(w, shape=[u1, u2],
		initializer=tf.contrib.layers.xavier_initializer())


def bias(b,u):
	return tf.get_variable(name=b, shape=[u], 
			initializer=tf.zeros_initializer())


def get_weights(units,prefix,full_or_conv,deconv=False):

	with tf.variable_scope(prefix):
		weights = {}

		for i in range(len(units)-1):

			w = '{}w{}'.format(prefix,i+1); b = '{}b{}'.format(prefix,i+1)

			weights[w] = full_or_conv(w,units[i],units[i+1])
			
			if deconv:
				weights[b] = bias(b,units[i])
			else:
				weights[b] = bias(b,units[i+1])

			
	weight_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=prefix)

	return weights, weight_list


### AUTOENCODER ################################################################
################################################################################


def encoder_full(X,enc_weights):

	e1 = tf.nn.relu(tf.matmul(X, enc_weights['ew1']) + enc_weights['eb1'])

	e2 = tf.nn.relu(tf.matmul(e1, enc_weights['ew2']) + enc_weights['eb2'])

	return e2


def y_latent(e2,y_weights):

	y_la = tf.matmul(e2, y_weights['yw1']) + y_weights['yb1']

	return y_la


def y_reparameterize(y_x_logits,CFY):

	if CFY == 0.0:
		y_gum = tf.nn.softmax(y_x_logits); 
	else:
		y_gum = gumbel_softmax(y_x_logits,CFY,hard=False); 

	return y_gum


def z_latent(e2,z_weights):

	z_la = tf.matmul(e2, z_weights['zw1']) + z_weights['zb1']

	return z_la


def s_latent(e2,s_weights):

	s_la = tf.matmul(e2, s_weights['sw1']) + s_weights['sb1']

	return s_la


def z_reparameterize(z_mean, z_log_sigma_sq, e):

    eps = tf.random_normal(tf.shape(z_log_sigma_sq), 0, e, dtype = tf.float32)

    z = z_mean + tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps)
    #z = z_mean + tf.exp(0.5*z_log_sigma_sq) * eps * e
    return z



def decoder_full(mu2,dec_weights):

	d1 = tf.nn.relu(tf.matmul(mu2, dec_weights['dw1']) + dec_weights['db1'])

	d2 = tf.nn.relu(tf.matmul(d1, dec_weights['dw2']) + dec_weights['db2'])

	logits = tf.matmul(d2, dec_weights['dw3']) + dec_weights['db3']
	
	prob = tf.nn.sigmoid(logits)

	return prob, logits

epsilon = 1e-3

def batch_norm_wrapper(inputs, is_training, decay = 0.999):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)





### ADVERSARIAL ################################################################
################################################################################


def D_z(z, z_dis_weights):#,DROP=1):

	h1 = tf.nn.relu(tf.matmul(z, z_dis_weights['dzw1']) + z_dis_weights['dzb1'])

	h2 = tf.nn.relu(tf.matmul(h1, z_dis_weights['dzw2']) + z_dis_weights['dzb2'])

	logits = tf.matmul(h2, z_dis_weights['dzw3']) + z_dis_weights['dzb3']

	prob = tf.nn.sigmoid(logits)

	return prob, logits

def D_y(y,y_dis_weights):#,DROP=1):

	h1 = tf.nn.relu(tf.matmul(y, y_dis_weights['dyw1']) + y_dis_weights['dyb1'])

	h2 = tf.nn.relu(tf.matmul(h1, y_dis_weights['dyw2']) + y_dis_weights['dyb2'])

	logits = tf.matmul(h2, y_dis_weights['dyw3']) + y_dis_weights['dyb3']

	prob = tf.nn.sigmoid(logits)

	return prob, logits


def discriminator_loss(real_logits,fake_logits):

	loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
		labels=tf.ones_like(real_logits), logits=real_logits))

	loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
		labels=tf.zeros_like(fake_logits), logits=fake_logits))
	
	return loss_real + loss_fake

def generator_loss(fake_logits):
	
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
		labels=tf.ones_like(fake_logits), logits=fake_logits))



### EVALUATION #################################################################
################################################################################


def evaluation(logits, labels):
  length = logits.shape[0]
  y_h = tf.argmax(logits,1)
  y = tf.argmax(labels,1)
  correct_prediction = tf.equal(y_h, y)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  return accuracy



def save_reconstructions(sess,reconstruction,X,x_drop,test_images, pix,script, name, epoch):
	# Applying encode and decode over test set
	IMAGE_HEIGHT = int(np.sqrt(pix)); IMAGE_WIDTH = IMAGE_HEIGHT; DEPTH = 1; PIXELS = pix
	test_images = test_images.reshape(-1,IMAGE_HEIGHT,IMAGE_WIDTH,DEPTH).squeeze()
	#test_tmp = test_images.transpose(0,3,1,2)
	test_tmp = test_images.reshape(-1,PIXELS)
	reconstructions = sess.run(
			reconstruction, feed_dict={X: test_tmp[:EXAMPLES],x_drop:1})

	recon = reconstructions.reshape(-1,IMAGE_HEIGHT,IMAGE_WIDTH,DEPTH).squeeze()

	for i in range(EXAMPLES):
		plt.subplot(2,10,i+1)
		plt.axis('off')
		plt.imshow(test_images[i])
		plt.subplot(2,10,11+i)
		plt.axis('off')
		plt.imshow(recon[i])
	model_path = '../images/{}'.format(script)
	if not os.path.exists(model_path):
		os.makedirs(model_path)
	plt.savefig('{}/{}_{}_recon.jpg'.format(model_path, name, epoch))


def save_examples(sess, reconstruction, yz_x, script, name, epoch,Z_NODES):
	
	test_y = np.eye(CLASSES)
	test_z = np.random.randn(CLASSES,Z_NODES)
	test_yz = np.concatenate((test_y, test_z),axis = 1)

	test_recon = sess.run(reconstruction, feed_dict={yz_x: test_yz})#y_e: test_y, z_e:test_z})
	#print(test_recon)
	test_recon = test_recon.reshape((-1,IMAGE_HEIGHT,IMAGE_WIDTH,DEPTH)).squeeze()
	for i in range(CLASSES):
		plt.subplot(1,10,i+1)
		plt.axis('off')
		plt.imshow(test_recon[i,:,:])
	model_path = '../images/{}'.format(script)
	if not os.path.exists(model_path):
		os.makedirs(model_path)
	plt.savefig('{}/{}_{}_gen.jpg'.format(model_path, name, epoch))

	#scipy.misc.imsave('D:/images/{}/.jpg'.format(script,name), test_recon)


def save_model(saver,sess,script,name,epoch):

	model_path = "../models/{}".format(script)

	if not os.path.exists(model_path):
		os.makedirs(model_path)

	model_name = '{}/{}_{}.ckpt'.format(model_path,name,epoch)
	save_path = saver.save(sess, model_name)
	print("Model saved in file: {}".format(save_path))


def get_representation(sess,latent,X,train_images,test_images):

		train_images = np.reshape(train_images,[-1,PIXELS])
		test_images = np.reshape(test_images,[-1,PIXELS])

		train_rep = sess.run(latent, feed_dict={X:train_images})
		test_rep = sess.run(latent, feed_dict={X:test_images})
		return train_rep, test_rep


def save_results(pickle_file, train_rep, test_rep, train_labels,test_labels):

	f = open(pickle_file,'wb')
	save = {'train_rep': train_rep,
					'train_labels': train_labels,
					'test_rep': test_rep,
					'test_labels': test_labels}
	pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
	f.close()


### LOAD DATA ##################################################################
################################################################################


def load_mnist():

	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("../data/MNIST_data", one_hot=True)

	train_images = mnist.train.images
	valid_images = mnist.validation.images
	test_images = mnist.test.images

	train_labels = mnist.train.labels
	valid_labels = mnist.validation.labels
	test_labels = mnist.test.labels

	all_images = np.concatenate((train_images,valid_images,test_images),0)
	all_labels = np.concatenate((train_labels,valid_labels,test_labels),0)

	return all_images, all_labels


def load_svhn():

	pickle_file = '../data/svhn.pickle'

	with open(pickle_file, 'rb') as f:
		save = pickle.load(f)
		train_images = save['train_images']
		train_labels = save['train_labels_ohe']
		train_labels_ohe = save['train_labels_ohe']
		test_images = save['test_images']
		test_labels = save['test_labels_ohe']
		test_labels_ohe = save['test_labels_ohe']
		extra_images = save['extra_images']
		extra_labels = save['extra_labels_ohe']
		extra_labels_ohe = save['extra_labels_ohe']
		del save

		all_images = np.concatenate((train_images,extra_images),0)
		all_labels = np.concatenate((train_labels,extra_labels),0)

	return all_images,all_labels,test_images,test_labels


def get_dataset(dataset, decrease, n_train, n_test, ss_labels, seed):
	if dataset == 'mnist':
		pixels = 28 * 28
		all_images, all_labels= load_mnist()
		all_images = all_images.reshape((-1,pixels))
		train_images = all_images[:n_train,:]; train_labels = all_labels[:n_train,:]
		test_images = all_images[n_train:(n_train + n_test),:] 
		test_labels = all_labels[n_train:(n_train + n_test),:]
		
		if decrease:
			idxs = index.MNIST_IDXS
		else:
			idxs = get_strat_idx(train_labels,ss_labels,seed)

	elif dataset == 'svhn':
		pixels = 32 * 32
		train_images,train_labels,test_images,test_labels = load_svhn()
		train_images = train_images.reshape((-1,pixels))
		train_images = train_images[:n_train,:]; train_labels = train_labels[:n_train,:]
		test_images = test_images.reshape((-1,pixels))
		test_images = test_images[:n_test,:]; test_labels = test_labels[:n_test,:]
		if decrease:
			idxs = index.SVHN_IDXS
		else:
			idxs = get_strat_idx(train_labels,ss_labels,seed)
	else:
		raise Exception('I dont recongnise the dataset')

	return train_images, train_labels, test_images, test_labels, idxs, pixels


def get_strat_idx(train_labels,samples, s):
	
	np.random.seed(s)
	num = samples // CLASSES

	labels = np.argmax(train_labels,1)
	random_stratified_idx = []
	
	for i in range(CLASSES):

		idx = np.random.permutation(np.where(labels==i)[0])[:num]
		random_stratified_idx.append(idx)

	return np.random.permutation(np.array(random_stratified_idx).reshape(-1))


def get_data(step, batch_size, images, labels):

	if labels.shape[0] % batch_size == 0:
		offset = (step * batch_size) % (labels.shape[0])
	else:  
		offset = (step * batch_size) % (labels.shape[0] - batch_size)

	batch_data = images[offset:(offset + batch_size), :]
	batch_labels = labels[offset:(offset + batch_size), :]
	
	return batch_data, batch_labels


def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    else:
         raise ValueError
