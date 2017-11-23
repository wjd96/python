import sys
sys.path.append(r'/home/wjd/python/machinelearn/tp/')

#digits=load_digits()
#data=digits.data
#target=pandas.get_dummies(digits.target).values
#trainx=numpy.array(data[:1500])
#trainx = tf.reshape(trainx,[-1,8,8,1])
#trainy=numpy.array(target[:1500])
#testx=numpy.matrix(data[1500:])
#testy=numpy.matrix(target[1500:])
#sys.path.append(r'/home/wjd/python/machinelearn/tp/')
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
