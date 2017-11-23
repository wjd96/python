import numpy
import pandas
import tensorflow as tf
def weight_variable(shape):
        initial=tf.truncated_normal(shape,stddev=0.1)
        return tf.Variable(initial)
def bias_variable(shape):
        initial=tf.constant(0.1,shape=shape)
        return tf.Variable(initial)
def conv2d(x,w):
        return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
def max_pool_2x2(x):
        return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
w_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
from sklearn.datasets import fetch_lfw_people
from sklearn.datasets import load_digits
digits=load_digits()
data=digits.data
target=pandas.get_dummies(digits.target).values
trainx=numpy.matrix(data[:1500])
#trainx = tf.reshape(trainx,[-1,8,8,1])
trainy=numpy.matrix(target[:1500])
testx=numpy.matrix(data[1500:])
testy=numpy.matrix(target[1500:])
#x=tf.placeholder("float",[None,64])
#y_=tf.placeholder("float",[None,10])
#xtrain=tf.reshape(x,[-1,28,28,1])
trainx=tf.constant(trainx,dtype=tf.float32)
#trainy=tf.constant(trainy,dtype=tf.float32)
#x=tf.constant(x,dtype=tf.float32)
trainx=tf.reshape(trainx,[-1,8,8,1])
print trainx
h_conv1=tf.nn.relu(conv2d(trainx,w_conv1)+b_conv1)
print h_conv1
h_pool1=max_pool_2x2(h_conv1)
print h_pool1
#w_conv2=weight_variable([5,5,32,64])
#b_conv2=bias_variable([64])
#h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
#h_pool2=max_pool_2x2(h_conv2)
w_fc1=weight_variable([4*4*32,64])
b_fc1=bias_variable([64])
h_pool2_flat=tf.reshape(h_pool1,[-1,4*4*32])
print h_pool2_flat
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)
w_fc2=weight_variable([64,10])
b_fc2=bias_variable([10])
y_conv=tf.nn.softmax(tf.matmul(h_fc1,w_fc2)+b_fc2)
#x=tf.placeholder("float",[None,64])
#y_=tf.placeholder("float",[None,10])
#print len(y_conv)
print y_conv
#y_conv=y_conv[0:1500]
cross_entropy = -tf.reduce_sum(trainy*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(trainy,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
for i in range(1000):
     # batch = mnist.train.next_batch(50)
        if i%100 == 0:
                train_accuracy = accuracy.eval()
                print "step %d, training accuracy %f"%(i, train_accuracy)
        train_step.run()
print "???????????????????????????????????????????????????????????????????"
saver=tf.train.Saver()
saver.save(sess,"/home/wjd/python/machinelearn/model.ckpt")
testx=tf.constant(testx,dtype=tf.float32)
testx=tf.reshape(testx,[-1,8,8,1])
#yy=tf.placeholder("float",[None,10])
h_conv2=tf.nn.relu(conv2d(testx,w_conv1)+b_conv1)
h_pool2=max_pool_2x2(h_conv2)
print h_pool2
h_pool3_flat=tf.reshape(h_pool2,[-1,4*4*32])
print h_pool3_flat
h_fc2=tf.nn.relu(tf.matmul(h_pool3_flat,w_fc1)+b_fc1)
print h_fc2
y_conv1=tf.nn.softmax(tf.matmul(h_fc2,w_fc2)+b_fc2)
print y_conv1
#cross_entropy1 = -tf.reduce_sum(testy*tf.log(y_conv1))
#train_step1 = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy1)
correct_prediction1 = tf.equal(tf.argmax(y_conv1,1), tf.argmax(testy,1))
accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, "float"))
#train_step1.run()
#cross_entropy1.run()
print "***********************************************************************"
print "test accuracy %f"%accuracy1.eval()
