#!/usr/bin/python2.7
import numpy
import pandas
from sklearn.datasets import load_boston
ds=load_boston()
train_x=ds.data[0:400]
train_y=ds.target[0:400]
test_x=ds.data[400:]
test_y=ds.target[400:]
mat=numpy.dot(train_x.T,train_x)
mat=numpy.matrix(mat)
aph=numpy.matrix(numpy.eye(len(mat)))
t=1
leng=1000
result=[]
x=[]
for i in range(0,leng):
	x.append(t)
	mat=mat+t*aph
	mat=mat.I
	mr=numpy.dot(mat,train_x.T)
	w=numpy.dot(mr,train_y)
	test_theta=numpy.array(w)[0]
	#print t
	#print test_theta
	t=t+1
	result.append(test_theta.tolist())
result=numpy.array(result)
print result[:,0]
#y=[]
#for i in range(0,len(test_x)):
#        y.append(numpy.dot(test_x[i],test_theta))
        #print '%f      %s'% (y, test_y[i])
#print y
#print test_y
#ax=[]
#ax.append(y)
#ax.append(test_y)
#ax=numpy.array(ax).T
#print ax
#result=ax[numpy.lexsort(ax.T)]
import matplotlib.pyplot as plt

for i in range(0,len(result[0])):
	plt.subplot(4,4,1+i)
	plt.plot(x,result[:,i],'b-')
#	plt.figure(num=13, figsize=(8, 5))
plt.xticks(numpy.arange(0,1000,50))
plt.yticks(numpy.arange(0,30000,50))
plt.show()
