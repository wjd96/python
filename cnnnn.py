#from tensorflow.examples.tutorials.mnist import input_data  
import tensorflow as tf 
from sklearn.datasets import load_digits
import pandas
import numpy
import sys
#sys.path.append(r'/home/wjd/python/machinelearn/tp/')

digits=load_digits()
data=digits.data
target=pandas.get_dummies(digits.target).values
trainx=numpy.matrix(data[:1500])
trainx=tf.constant(trainx,dtype=tf.float32)
#trainx = tf.reshape(trainx,[-1,8,8,1])
trainy=numpy.matrix(target[:1500])
#testx=numpy.matrix(data[1500:])
#testy=numpy.matrix(target[1500:])
#sys.path.append(r'/home/wjd/python/machinelearn/tp/')
#import input_data
#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
sess = tf.InteractiveSession()
def weight_variable(shape):  
	initial = tf.truncated_normal(shape, stddev=0.1)  
	return tf.Variable(initial)  
def bias_variable(shape):  
	initial = tf.constant(0.1, shape=shape)  
        return tf.Variable(initial)  
def conv2d(x, W):    
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')    
def max_pool_2x2(x):    
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')    
#xs = tf.placeholder(tf.float32, [None, 8*8])   
ys = tf.placeholder(tf.float32, [None, 10])   
keep_prob = tf.placeholder(tf.float32)  
trainx = tf.reshape(trainx, [-1,8,8, 1])  
 
#print x_image 
W_conv1 = weight_variable([5, 5, 1, 32])   
b_conv1 = bias_variable([32])    
h_conv1 = tf.nn.relu(conv2d(trainx, W_conv1) + b_conv1)    
h_pool1 = max_pool_2x2(h_conv1)   
  
#w_conv2 = weight_variable([5,5,32,64])   
#b_conv2  = bias_variable([64])   
#h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)    
#h_pool2 = max_pool_2x2(h_conv2)    
  
W_fc1 = weight_variable([4*4*32, 1024])   
b_fc1 = bias_variable([1024])   
h_pool2_flat = tf.reshape(h_pool1, [-1, 4*4*32])   
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)   
  
#keep_prob = tf.placeholder(tf.float32)   
#h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
    
W_fc2 = weight_variable([1024, 10])    
b_fc2 = bias_variable([10])    
y_conv=tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)   
  
cross_entropy = -tf.reduce_sum(ys * tf.log(y_conv))   
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)  
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(ys,1))  
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  
#tf.global_variables_initializer().run()  
sess.run(tf.initialize_all_variables())
for i in range(200):  
#	batch = mnist.train.next_batch(50)  
        if i%100 == 0:  
        	train_accuracy = accuracy.eval(feed_dict={ ys:trainy})  
                print("step %d, training accuracy %g"%(i, train_accuracy))  
        train_step.run(feed_dict={ ys:trainy})  
#print("test accuracy %g"%accuracy.eval(feed_dict={x: mnist.test.images, ys: mnist.test.labels, keep_prob: 1.0}))  
