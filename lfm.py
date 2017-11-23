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
import pandas
data=pandas.read_table('/home/wjd/python/machinelearn/movies.base',header=None)
data=data.ix[:,[0,1,2]]
data=data.pivot(0,1,2)
data=data.fillna(0)
data=data.values
ui=numpy.matrix(data[0:30,0:30])
u,i=ui.shape
d=3
uk=numpy.matrix(numpy.random.rand(u*d).reshape(u,d))
ik=numpy.matrix(numpy.random.rand(i*d).reshape(d,i))
#lfm(ui,uk,ik,0.02,0.01,10)
def lfm(ui,uk,ik,alpha,k,numiter):
	for i in range (0,numiter):
		tui=numpy.dot(uk,ik)
		de=ui-tui
		mp1=numpy.dot(de,ik.T)
		mp1-=k*uk
		uk+=alpha*mp1
		mp2=numpy.dot(uk.T,de)
		mp2-=k*ik
		ik+=alpha*mp2
		result=numpy.dot(uk,ik)
		print result
	
lfm(ui,uk,ik,0.002,0.01,500)	
print '*****************************************************************************'
print ui
