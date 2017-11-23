#!/usr/bin/python2.7
import numpy
import tensorflow as tf
from sklearn import datasets
iris=datasets.load_iris()
trainx=numpy.array([[x[0],x[3]]for x in iris.data])
trainy=numpy.array([1 if y==0 else -1 for y in iris.target])
print trainx
print trainy
x,y=tf.constant(trainx,dtype=tf.float32),tf.constant(trainy,dtype=tf.float32)
#x=tf.placeholder("float",[None,2])
#y=tf.placeholder("float",[None,1])
w=tf.Variable(tf.random_normal(shape=[2,1]))
b=tf.Variable(tf.random_normal(shape=[1]))
model_output=tf.sub(tf.matmul(x,w),b)
l2_norm=tf.reduce_sum(tf.square(w))
alpha=tf.constant([0.01])
classifer=tf.reduce_mean(tf.maximum(0.,tf.sub(1.,tf.mul(model_output,y))))
loss=tf.add(classifer,tf.mul(alpha,l2_norm))
train_step=tf.train.GradientDescentOptimizer(0.01).minimize(loss)
sess=tf.Session()
sess.run(tf.initialize_all_variables())
for i in range(2000):
	sess.run(train_step)
	

