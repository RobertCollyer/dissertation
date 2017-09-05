import os 
import time

start = time.time()

'''
for i in range(3):
	os.system('python saae.py 1000 1000 10 10 MSE GAUSS 0.001 0.001 0.003 1.0 0.3 0.1 svhn False 5000 10000 100 1000 {} {}'.format(i,i))
	
for i in range(3):
	os.system('python saae.py 1000 1000 10 10 MSE GAUSS 0.001 0.001 0.003 1.0 0.5 0.1 svhn False 5000 10000 100 1000 {} {}'.format(i,i))
	

for i in range(500):
	os.system('python saae.py 1000 1000 10 10 MSE GAUSS 0.001 0.001 0.003 1.0 0.3 0.1 svhn False 5000 10000 250 1000 {} {}'.format(i,i))
	os.system('python saae.py 1000 1000 10 10 MSE GAUSS 0.001 0.001 0.003 1.0 0.3 0.1 svhn False 5000 10000 250 2000 {} {}'.format(i,i))

	os.system('python saae.py 1000 1000 10 10 MSE GAUSS 0.001 0.001 0.003 1.0 0.3 0.1 svhn False 10000 10000 250 1000 {} {}'.format(i,i))
	os.system('python saae.py 1000 1000 10 10 MSE GAUSS 0.001 0.001 0.003 1.0 0.3 0.1 svhn False 10000 10000 250 2000 {} {}'.format(i,i))
'''


#os.system('python saae.py 1000 1000 10 10 MSE GAUSS 0.01 0.1 0.1 1.0 0.3 0.1 mnist True 60000 10000 10 100 127 0')



#os.system('python cvae.py 1000 1000 10 10 MSE GAUSS 0.01 0.1 1.0 0.3 1.0 mnist True 60000 10000 50 100 127 0')

### LAST
#for i in [0.01]:
	#for j in [0.1]:
		#os.system('python cvae.py 1000 1000 10 10 MSE GAUSS {} {} 1.0 0.3 1.0 mnist False 60000 10000 100 100 127 0'.format(i,j))


###NEXT

### saae svhn 1epoch = 20s
#for i in range(10000,5000-1,-5000):
#	for j in range(1):
#		os.system('python saae.py 1000 1000 10 10 MSE GAUSS 0.001 0.001 0.003 1.0 0.3 0.1 svhn True {} 10000 100 2000 127 {}'.format(i,j))
#		os.system('python saae.py 1000 1000 10 10 MSE GAUSS 0.001 0.001 0.003 1.0 0.3 0.1 svhn True {} 10000 200 2000 127 {}'.format(i,j))





#[60000,50000,4000,3000,20000,10000,5000]:

### BIG ONE ###
'''

for j in range(10):
	for i in [60000,50000,40000,30000,20000,10000,5000]:
		os.system('python saae.py 1000 1000 10 10 MSE GAUSS 0.01 0.1 0.1 1.0 0.3 0.1 mnist True {} 10000 100 100 127 {}'.format(i,j))
		os.system('python aae.py 1000 1000 10 10 MSE GAUSS 0.01 0.1 0.1 1.0 mnist True {} 10000 100 100 127 {}'.format(i,j))
		os.system('python cvae.py 1000 1000 10 10 MSE GAUSS 0.01 0.1 1.0 0.3 1.0 mnist True {} 10000 100 100 127 {}'.format(i,j))
		os.system('python saae.py 1000 1000 10 10 MSE GAUSS 0.001 0.001 0.003 1.0 0.3 0.1 svhn True {} 10000 100 2000 127 {}'.format(i,j))
		os.system('python aae.py 1000 1000 10 10 MSE GAUSS 0.001 0.001 0.003 1.0 svhn True {} 10000 100 2000 127 {}'.format(i,j))
		os.system('python cvae.py 1000 1000 10 10 MSE GAUSS 0.001 0.003 1.0 0.3 1.0 svhn True {} 10000 100 2000 127 {}'.format(i,j))
'''

os.system('python plot.py 1000 1000 10 10 MSE GAUSS 0.01 0.1 0.1 1.0 0.3 0.1 svhn True 60000 10000 100 100 127 167')
		


#for i in [60000,50000,40000,30000,20000,10000,5000]:
	#os.system('python cvae.py 1000 1000 10 10 MSE GAUSS 0.001 0.003 1.0 0.3 1.0 svhn True {} 10000 100 2000 127 15'.format(i))



'''

for i in range(10):
	os.system('python saae.py 1000 1000 10 10 MSE GAUSS 0.01 0.1 0.1 1.0 0.3 0.1 mnist True 60000 10000 100 100 127 {}'.format(i))
	os.system('python saae.py 1000 1000 10 10 MSE GAUSS 0.01 0.1 0.1 1.0 0.3 0.1 mnist True 60000 10000 101 100 127 {}'.format(i))
	os.system('python saae.py 1000 1000 10 10 MSE GAUSS 0.01 0.1 0.1 1.0 0.3 0.1 mnist True 5000 10000 100 100 127 {}'.format(i))
	os.system('python saae.py 1000 1000 10 10 MSE GAUSS 0.01 0.1 0.1 1.0 0.3 0.1 mnist True 5000 10000 101 100 127 {}'.format(i))




for i in [60000,50000,40000,30000,20000,10000,5000]:
	for j in range(20):
		os.system('python saae.py 1000 1000 10 10 MSE GAUSS 0.01 0.1 0.1 1.0 0.3 0.1 mnist True {} 10000 100 100 127 {}'.format(i,j))

### aae mnist saae mnist 1epoch = 12s
for i in [60000,50000,40000,30000,20000,10000,5000]:
	for j in range(20):
		os.system('python aae.py 1000 1000 10 10 MSE GAUSS 0.01 0.1 0.1 1.0 mnist True {} 10000 100 100 127 {}'.format(i,j))

### cvae mnist saae mnist 1epoch = 5s 
for i in [60000,50000,40000,30000,20000,10000,5000]:
	for j in range(20):
		os.system('python cvae.py 1000 1000 10 10 MSE GAUSS 0.01 0.1 1.0 0.3 1.0 mnist True {} 10000 100 100 127 {}'.format(i,j))

### saae svhn 1epoch = 20s
for i in [60000,50000,40000,30000,20000,10000,5000]:
	for j in range(20):
		os.system('python saae.py 1000 1000 10 10 MSE GAUSS 0.001 0.001 0.003 1.0 0.3 0.1 svhn True {} 10000 100 2000 127 {}'.format(i,j))

### aae svhn saae mnist 1epoch = 20s
for i in [60000,50000,40000,30000,20000,10000,5000]:
	for j in range(20):
		os.system('python aae.py 1000 1000 10 10 MSE GAUSS 0.001 0.001 0.003 1.0 svhn True {} 10000 100 2000 127 {}'.format(i,j))

### cvae svhn saae mnist 1epoch = 15s 
for i in [60000,50000,40000,30000,20000,10000,5000]:
	for j in range(20):
		os.system('python cvae.py 1000 1000 10 10 MSE GAUSS 0.001 0.003 1.0 0.3 1.0 svhn True {} 10000 100 2000 127 {}'.format(i,j))



'''




#os.system('python aae.py 1000 1000 10 10 MSE GAUSS 0.001 0.001 0.003 1.0 svhn False 5000 10000 100 2000 {} {}'.format(i,i))
	


#for i in range(1):
	#os.system('python cvae.py 1000 1000 10 10 MSE GAUSS 0.03 {} 1.0 0.3 1.0 mnist True 60000 10000 50 100 127 0'.format(i,j))





'''
for j in range(5):
	os.system('python aae.py 1000 1000 10 10 MSE GAUSS 0.001 0.001 0.003 1.0 svhn True 60000 10000 100 2000 11 {}'.format(j))

for j in range(5):
	os.system('python aae.py 1000 1000 10 10 MSE GAUSS 0.001 0.001 0.003 1.0 svhn True 10000 10000 100 2000 11 {}'.format(j))

for j in range(5):
	os.system('python aae.py 1000 1000 10 10 MSE GAUSS 0.001 0.001 0.003 1.0 svhn True 5000 10000 100 2000 11 {}'.format(j))


for j in range(5):
	os.system('python saae.py 1000 1000 10 10 MSE GAUSS 0.001 0.001 0.003 1.0 0.3 0.1 svhn True 60000 10000 100 2000 11 {}'.format(j))

for j in range(5):
	os.system('python saae.py 1000 1000 10 10 MSE GAUSS 0.001 0.001 0.003 1.0 0.3 0.1 svhn True 10000 10000 100 2000 11 {}'.format(j))

for j in range(5):
	os.system('python saae.py 1000 1000 10 10 MSE GAUSS 0.001 0.001 0.003 1.0 0.3 0.1 svhn True 5000 10000 100 2000 11 {}'.format(j))
'''


#os.system('python aae.py 1000 1000 10 10 MSE GAUSS 0.001 0.001 0.003 1.0 svhn True 5000 10000 100 2000 11 0')
#os.system('python aae.py 1000 1000 10 10 MSE GAUSS 0.001 0.001 0.003 1.0 svhn True 10000 10000 100 2000 11 0')
#os.system('python aae.py 1000 1000 10 10 MSE GAUSS 0.001 0.001 0.003 1.0 svhn True 60000 10000 100 2000 11 0')
	


#os.system('python cvae.py 1000 1000 10 10 MSE GAUSS 0.001 0.003 1.0 0.5 1.0 svhn False 5000 10000 100 2000 0 0')



'''
for i in range(10):
	os.system('python saae.py 1000 1000 10 10 MSE GAUSS 0.001 0.001 0.003 1.0 0.5 0.1 svhn True 60000 10000 100 5000 127 {}'.format(i))

for i in range(10):
	os.system('python aae.py 1000 1000 10 10 MSE GAUSS 0.001 0.001 0.003 1.0 svhn True 60000 10000 100 5000 127 {}'.format(i))

for i in range(10):
	os.system('python cvae.py 1000 1000 10 10 MSE GAUSS 0.001 0.003 1.0 0.5 1.0 svhn True 60000 10000 100 5000 127 {}'.format(i))

for i in range(10):
	os.system('python saae.py 1000 1000 10 10 MSE GAUSS 0.001 0.001 0.003 1.0 0.5 0.1 svhn True 10000 10000 100 5000 127 {}'.format(i))

for i in range(10):
	os.system('python aae.py 1000 1000 10 10 MSE GAUSS 0.001 0.001 0.003 1.0 svhn True 10000 10000 100 5000 127 {}'.format(i))

for i in range(10):
	os.system('python cvae.py 1000 1000 10 10 MSE GAUSS 0.001 0.003 1.0 0.5 1.0 svhn True 10000 10000 100 5000 127 {}'.format(i))
'''


#stable
#####,bn.py_1000_1000_10_10_ABS_GAUSS_0.001_0.003_0.003_0.0_0.0_0.0_svhn_False_60000_10000_250_5000,2017-08-12 05:51:12.001982

#os.system('python bn.py 1000 1000 10 10 ABS GAUSS 0.0003 0.003 0.01 0.8 0 0 svhn False 60000 10000 250 5000 127')


#os.system('python bn.py 1000 1000 10 10 MSE GAUSS 0.001 0.003 0.003 0 0 0 svhn False 60000 10000 250 5000 127')
#os.system('python bn.py 1000 1000 10 10 ABS GAUSS 0.001 0.003 0.003 0 0 0 svhn False 60000 10000 250 5000 127')

#os.system('python bn.py 1000 1000 10 10 ABS GAUSS 0.0003 0.003 0.01 0.8 0 0 svhn False 60000 10000 250 5000 127')


#5k! labels

#reduce labels to 1k?
#reduce training to 10k
# find best labels in 10k to decrease for INDEXES
#find best params for saae
#better with without dropout?




#for i in [0.01,0.001,0.0001]:
#	for j in [0.1,0.01]:
#		for k in [0.1,0.01]:
#			os.system('python saae.py 1000 1000 10 10 MSE GAUSS {} {} {} 0 0 0 svhn False 50000 10000 5 100 127'.format(i,j,k))



'''
for i in [0.1,0.01,0.001]:
	for j in [0.1,0.01,0.001]:
		os.system('python _cvae.py {} {} MSE GAUSS 0 0.5 1 False 127 100 10 1000 1000 50 60000'.format(i,j))
'''

#for i in [0.1,0.3,0.5,1.0]:
#	os.system('python _cvae.py 0.01 0.1 MSE GAUSS 0 {} 1 False 127 100 10 1000 1000 100 60000'.format(i))
#	os.system('python _cvae.py 0.01 0.1 MSE GAUSS 0 {} 1 False 127 100 10 1000 1000 100 60000'.format(i))

#good cvae
#os.system('python _cvae.py 0.01 0.1 MSE GAUSS 0 0.3 1 False 127 100 10 1000 1000 500 60000')
#os.system('python _cvae.py 0.01 0.1 MSE GAUSS 0 0.5 1 False 127 100 10 1000 1000 500 60000')


#os.system('python _aae_re.py 0.0001 0.01 0.1 MSE GAUSS 1 0 0 0 0 0 0 False 11 1000 20 1000 1000 50 531131')

#os.system('python _aae_re.py 0.01 0.1 0.1 MSE GAUSS 1 0.1 1 0 False 127 100 10 1000 1000 101')
#os.system('python _aae_re.py 0.01 0.1 0.1 MSE GAUSS 1 0.1 1 0 False 127 100 10 1000 1000 101')

#discriminative network qφ(y|x), inference network qφ(z|x, y), and generative network pθ(x|y, z)

'''
print(2*2*3*3*3*3*10//60)

objs = ['MSE','ABS']
type_noise = ['GAUSS', 'LAP']
reg_noise = [0.5,1,2]
xyz_noise = [0,0.1,0.3]

for i in objs:
	for j in type_noise:
		for k in reg_noise:
			for l in xyz_noise:
				for m in xyz_noise:
					for n in xyz_noise:					
						os.system('python _aae_re.py 0.01 0.1 0.1 {} {} {} {} {} {} False 5 100 10 1000 1000 50'.format(i,j,k,l,m,n)) # 0 seed

print(2*2*3*3*3*3*10//60)

for q in range(250):
	os.system('python _aae.py 0.01 0.1 0.1 MSE GAUSS 1 0 0 0 False {} 100 10 1000 1000 20'.format(q))
'''



duration = time.time() - start
print (duration)

