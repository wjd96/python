#!/usr/bin/python2.7
import numpy
import pandas
from sklearn.datasets import load_boston
ds=load_boston()
train_x=ds.data[0:400]
train_y=ds.target[0:400]
test_x=ds.data[400:]
test_y=ds.target[400:]
theta=numpy.ones(13)
count=len(train_x)
x=numpy.zeros(shape=(100,2))
y=numpy.zeros(shape=100)
for i in range(0,100):
	x[i][0]=1
	x[i][1]=i
	y[i]=(i+25)+numpy.random.uniform(0,1)*10

def z_score(x):
        x=numpy.array(x).astype(float)
        xr=numpy.rollaxis(x,axis=1)
        xr-=numpy.mean(x,axis=1)
        xr/=numpy.std(x,axis=1)
        return -xr.T
def num(x):
	x=numpy.array(x).astype(float)
	xr=numpy.rollaxis(x,axis=1)
	min=numpy.min(x,axis=1)
	max=numpy.max(x,axis=1)
	xr-=min
	xr/=(max-min)
	return xr.T
#test_theta=gradientd(train_x,train_y,theta,0.3,count,100)
#print "**********************************************************************************"
#for i in test_x:
#	y=numpy.dot(i,test_theta)
#	print"%f  %f" %  y,test_y[i]

def gradientd(x,y,theta,alpha,m,numiter):
	xTrans=numpy.array(x).transpose()
	for i in range(0,numiter):
#		print x,y
		hypothesis=numpy.dot(x,theta)
		loss=hypothesis-y
#		print loss
		cost=numpy.sum(loss**2)/(2*m)
		#print cost
		gradient=numpy.dot(xTrans,loss)/m
		theta=theta-alpha*gradient
		theta=theta*(1-(30*alpha)/m)-alpha*gradient
		print cost,theta
	return theta



def batch_gradient_descent(x,y,theta,alpha,m,max_iter):
	iter = 0
	while  iter < max_iter:
		deviation = 0
        	sigma1 = 0
        	sigma2 = 0
        	for i in range(m):
            		h = theta[0] * x[i][0] + theta[1] * x[i][1]
            		sigma1 = sigma1 +  (y[i] - h)*x[i][0] 
            		sigma2 = sigma2 +  (y[i] - h)*x[i][1] 
        	theta[0] = theta[0] + alpha * sigma1 /m
        	theta[1] = theta[1] + alpha * sigma2 /m
       		for i in range(m):
            		deviation = deviation + (y[i] - (theta[0] * x[i][0] + theta[1] * x[i][1])) ** 2
        	iter = iter + 1
		print theta
		print deviation
	return theta, iter
#train_x=z_score(train_x)
print train_x
#print numpy.min(train_x,axis=1)
#print numpy.max(train_x,axis=1)
#train_x=num(train_x)
#print train_x
test_theta=gradientd(train_x,train_y,theta,0.000001,400,100000)
#test_theta=batch_gradient_descent(train_x,train_y,theta,0.00005,100,500)
print "------------------------------------------------------------------------------------"
#print test_y
#test_x=num(test_x)
#test_x=z_score(test_x)
y=[]
for i in range(0,len(test_x)):
        y.append(numpy.dot(test_x[i],test_theta))
        #print '%f	%s'% (y, test_y[i])
#print y
#print test_y
ax=[]
ax.append(y)
ax.append(test_y)
ax=numpy.array(ax).T
print ax
result=ax[numpy.lexsort(ax.T)]
#print result[:,0]
#print result[:,1]
import matplotlib.pyplot as plt
plt.plot(result[:,1],result[:,0],'ob-')
plt.show()
