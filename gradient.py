#!/usr/bin/python2.7

import numpy
def gradientd(x,y,theta,alpha,m,numiter):
	xTrans=numpy.array(x).transpose()
	for i in range(0,numiter):
		hypothesis=numpy.dot(x,theta)
		loss=hypothesis-y
		cost=numpy.sum(loss**2)/(2*m)
		gradient=numpy.dot(xTrans,loss)/m
		theta=theta-alpha*gradient
	return theta
