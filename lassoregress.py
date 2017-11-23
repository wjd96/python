#!/usr/bin/python2.7

import numpy
import math
import pandas
from sklearn.datasets import load_boston
def gradientd(x,y,theta,alpha,m,numiter):
	xTrans=numpy.array(x).transpose()
	for i in range(0,numiter):
		hypothesis=numpy.dot(x,theta)
		loss=hypothesis-y
		cost=numpy.sum(loss**2)/(2*m)
		gradient=numpy.dot(xTrans,loss)/m
		theta=theta-alpha*gradient
	return theta
def gradient(x,y,theta,alpha,numiter):
	#xtrans=numpy.array(x).transpose()
#	newtheta=[]
	for i in range(0,numiter):
		for i in range(0,len(x[0])):
			delx=numpy.delete(x,i,axis=1)
			deltheta=numpy.delete(theta,i)
			hypo=numpy.dot(delx,deltheta)
			cost=train_y-hypo
			xt=x[:,i]
			p=numpy.dot(xt,cost)
			q=pow(xt,2).sum()
			if p<-(alpha/2):
				theta[i]=(p+alpha/2)/q
			elif p>alpha/2:
				theta[i]=(p-alpha/2)/q
			else:
				theta[i]=0
		print "theta:"
		print theta
		ty=numpy.dot(x,theta)
		print "cost:%f"%math.sqrt(pow(ty-train_y,2).sum())
ds=load_boston()
train_x=ds.data[0:40]
train_y=ds.target[0:40]
test_x=ds.data[400:]
test_y=ds.target[400:]
theta=numpy.ones(13)
#print train_x
#print train_y
#print theta
gradient(train_x,train_y,theta,35,200000)
		
