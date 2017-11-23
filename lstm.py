#!/usr/bin/python2.7
import numpy
import tensorflow as tf
import sys
from sklearn.datasets import load_digits
import pandas
digits=load_digits()
data=digits.data
target=pandas.get_dummies(digits.target).values
trainx=data[:1500]
trainy=target[:1500]
testx=data[1500:]
testy=target[1500:]
train_iters=5000
trainx=numpy.reshape(trainx,[-1,8,8])
testx=numpy.reshape(testx,[-1,8,8])
lr=0.001
n_inputs=8
n_steps=8
n_hidden_units=8
n_classes=10
x=tf.placeholder(tf.float32,[None,n_steps,n_inputs])
y=tf.placeholder(tf.float32,[None,n_classes])
#with tf.variable_scope('forward'):
cell=tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
weights={'in':tf.Variable(tf.random_normal([n_inputs,n_hidden_units])),
'out':tf.Variable(tf.random_normal([n_hidden_units,n_classes]))}
biases={'in':tf.Variable(tf.constant(0.1,shape=[n_hidden_units,])),
'out':tf.Variable(tf.constant(0.1,shape=[n_classes,]))}
def Rnn(x,weigths,biases,num,cell):
#	tf.reset_default_graph()  
	x=tf.reshape(x,[-1,n_inputs])
	x_in=tf.matmul(x,weights['in'])+biases['in']
	x_in=tf.reshape(x_in,[-1,n_steps,n_hidden_units])
	#print x_in
	#cell=tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
	init_state=cell.zero_state(num,dtype=tf.float32)
	#print init_state
#	print init_state.eval()
	outputs,final_state=tf.nn.dynamic_rnn(cell,x_in,initial_state=init_state)
	print outputs
	print final_state
	#output=tf.unpack(tf.transpose(outs,[1,0,2]))
	results=tf.matmul(final_state[1],weights['out'])+biases['out']
	return results
pred=Rnn(x,weights,biases,1500,cell)
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
train_op=tf.train.AdamOptimizer(lr).minimize(cost)
correct_pred=tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_pred,tf.float32))
#print trainx
#print trainy
with tf.Session() as sess:
	init = tf.global_variables_initializer()
	sess.run(init)
	step=0
	for step in range(0,train_iters):
		sess.run(train_op,feed_dict={x:trainx,y:trainy})
#		sess.run(accuracy,feed_dict={x:trainx,y:trainy})
		print accuracy.eval(feed_dict={x:trainx,y:trainy}, session=sess)
	print '******************************************************************'
#pred1=Rnn(x,weights,biases,testx.size)
	#sess.run(train_op,feed_dict={x:testx,y:testy})
#	with tf.variable_scope('forward',resue=True):
      #  cell=tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
	pred1=Rnn(x,weights,biases,297,cell)
	#cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
	#train_op=tf.train.AdamOptimizer(lr).minimize(cost)
	correct_pred1=tf.equal(tf.argmax(pred1,1),tf.argmax(y,1))
	accuracy1=tf.reduce_mean(tf.cast(correct_pred1,tf.float32))
	sess.run(accuracy1,feed_dict={x:testx,y:testy})
	print accuracy1.eval(feed_dict={x:testx,y:testy},session=sess)
