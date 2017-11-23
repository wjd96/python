#!/usr/bin/python2.7

import numpy
import random
import pandas
import math
def np_exp(hypo):
	result=[]
	for i in hypo:
		result.append(1/(1+math.exp(-i)))
	return numpy.array(result)
def gradientd(x,y,theta,alpha,m,numiter):
	xTrans=numpy.array(x).transpose()
	for i in range(0,numiter):
		hypothesis=numpy.dot(x,theta)
		hypo=np_exp(hypothesis)
		loss=hypo-y
		print hypo
		print y
		cost=numpy.sum(loss**2)/(2*m)
		gradient=numpy.dot(xTrans,loss)/m
		theta=theta-alpha*gradient
		print theta
	return theta
def train():
	pf=pandas.read_csv('/home/wjd/python/machinelearn/12.txt',header=None)
	pf1=pf[pf[4]=='Iris-virginica']
	pf2=pf[pf[4]=='Iris-setosa'] 
	pf1[4]=0
	pf2[4]=1
	pf=pandas.concat([pf1,pf2])
	pf.index=range(0,len(pf))
	rag=range(0,85)
#	import random
	random.shuffle(rag)
	x=pf.ix[:,0:3].values
	y=pf.ix[:,4].values
	return x,y

def test():
        pf=pandas.read_csv('/home/wjd/python/machinelearn/13.txt',header=None)
        pf1=pf[pf[4]=='Iris-virginica']
        pf2=pf[pf[4]=='Iris-setosa']
        pf1[4]=0
        pf2[4]=1
        pf=pandas.concat([pf1,pf2])
        pf.index=range(0,len(pf))
       # rag=range(0,85)
       # import random
       # random.shuffle(rag)
        x=pf.ix[:,0:3].values
        y=pf.ix[:,4].values
        return x,y
trainx,trainy=train()
testx,testy=test()
theta=numpy.random.randn(4)
test_theta=gradientd(trainx,trainy,theta,0.05,85,100)
print '*****************************************************************************'
gradientd(testx,testy,test_theta,0.005,len(testx),1)






