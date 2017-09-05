# Stochastic Adversarial Autoencoder

## RobertCollyer

### Whats in it?

##### Files

- 3 models, saae.py, aae.py and cvae.py 
- utils.py contains all the functions used to build and train models
- index.py contains index information for labels
- svhn.py creates pickle file for svhn data.


### How to run

- create a data directory in the parent directory of the code folder
- The mnist dataset should download automatically to ../data when saae.py, aae.py or cvae.py are run 
- for svhn down load svhn files to ../data and run svhn.py file which creates a pickle of the data (~5gb)
- run any of the 3 models saae.py, aae.py or cvae.py

### Examples

#### SAAE example
python saae.py 1000 1000 10 10 MSE GAUSS 0.01 0.1 0.1 1.0 0.3 0.1 mnist True 60000 10000 100 100 127 0

#### SAAE params
units in hidden layer 1 - any +ve int
units in hidden layer 2 - any +ve int
number of y units - any +ve int
number of z units - any +ve int
likelihood / reconstruction - MSE or ABS
posterior / latent - GAUSS or LAP
reconstruction learning rate - float 0-1
regularisation learning rate - float 0-1
semi-supervised learning rate - float 0-1
dropout rate - float 0-1
gumbel-softmax temperature - float 0-1
gaussian noise - float 0-1
dataset - mnist or svhn
use predefine labels - True
number training points - bweteen 5000 and 60000
number validation / test points 10000
epochs - any +ve int
number of labels any +ve int < number training points
random seed - any +ve int
counter - any +ve int

#### AAE example	
python aae.py 1000 1000 10 10 MSE GAUSS 0.01 0.1 0.1 1.0 mnist True 600000 10000 100 100 127 0

#### AAE params	
units in hidden layer 1 - any +ve int
units in hidden layer 2 - any +ve int
number of y units - any +ve int
number of z units - any +ve int
likelihood / reconstruction - MSE or ABS
posterior / latent - GAUSS or LAP
reconstruction learning rate - float 0-1
regularisation learning rate - float 0-1
semi-supervised learning rate - float 0-1
dropout rate - float 0-1
dataset - mnist or svhn
use predefine labels - True
number training points - bweteen 5000 and 60000
number validation / test points 10000
epochs - any +ve int
number of labels any +ve int < number training points
random seed - any +ve int
counter - any +ve int

	
#### GSVAE example	
python cvae.py 1000 1000 10 10 MSE GAUSS 0.01 0.1 1.0 0.3 1.0 mnist True 60000 10000 100 100 127 0
		
#### GSVAE params		
units in hidden layer 1 - any +ve int
units in hidden layer 2 - any +ve int
number of y units - any +ve int
number of z units - any +ve int
likelihood / reconstruction - MSE or ABS
posterior / latent - GAUSS or LAP
reconstruction + regularization learning rate - float 0-1
semi-supervised learning rate - float 0-1
dropout rate - float 0-1
gumbel-softmax temperature - float 0-1
gaussian noise - float 0-1
dataset - mnist or svhn
use predefine labels - True
number training points - bweteen 5000 and 60000
number validation / test points 10000
epochs - any +ve int
number of labels any +ve int < number training points
random seed - any +ve int
counter - any +ve int


Python 3.5.0 and Tensorflow 1.0.1 was used in this report.

The agents were trained on a GPU. I tried to save the agents using tf.device("/cpu:0) however Im not sure if this worked.

To replicate the results in report please run 

- results.py which calls and prints the results of all the agents in the assignments. Although this may take a while due to 100 episodes for part B. This can be reduced in funcs_b.py

