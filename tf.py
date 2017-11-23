#!/usr/bin/python
import numpy
import pandas
import tensorflow as tf
from sklearn.datasets import load_digits
digits=load_digits()
data=digits.data
target=pandas.get_dummies(digits.target).values
x=tf.placeholder("float",[None,64])
trainx=numpy.matrix(data[:1500])
trainy=numpy.matrix(target[:1500])
testx=numpy.matrix(data[1500:])
testy=numpy.matrix(target[1500:])
w=tf.Variable(tf.zeros([64,10]))
b=tf.Variable(tf.zeros([10]))
y=tf.nn.softmax(tf.matmul(x,w)+b)
y_=tf.placeholder("float",[None,10])
#cross_entropy=-tf.reduce_sum(y_*tf.log(y))
cross_entropy=tf.reduce_sum(tf.pow(y_-y,2))
train_step=tf.train.GradientDescentOptimizer(0.00001).minimize(cross_entropy)
init=tf.initialize_all_variables()
sess=tf.Session()
sess.run(init)
for i in range(100):
	sess.run(train_step,feed_dict={x:trainx,y_:trainy})
	correct=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
	accuracy=tf.reduce_mean(tf.cast(correct,"float"))
	print sess.run(accuracy,feed_dict={x:trainx,y_:trainy})
print "*****************************************************************"
#cross_entropy=tf.reduce_sum(tf.pow(y_-y,2))
#train_step1=tf.train.GradientDescentOptimizer(0.00001).minimize(cross_entropy)
#init=tf.initialize_all_variables()
#sess=tf.Session()
#sess.run(init)
saver=tf.train.Saver()
saver.save(sess,"/home/wjd/python/machinelearn/tf.ckpt")
sess.run(train_step,feed_dict={x:testx,y_:testy})
correct=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct,"float"))
print sess.run(accuracy,feed_dict={x:testx,y_:testy})





