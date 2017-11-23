#!/usr/bin/python2.7
import numpy
import pandas
pf=pandas.read_csv('/home/wjd/python/machinelearn/12.txt',header=None)
x1=pf[pf[4]=='Iris-setosa']
x2=pf[pf[4]=='Iris-virginica']
x3=pf[pf[4]=='Iris-versicolor']
x1=x1.ix[:,0:3]
x2=x2.ix[:,0:3]
x3=x3.ix[:,0:3]
x1_cov=x1.cov()
x2_cov=x2.cov()
sw=x1_cov+x2_cov
temp=x1.mean(axis=0)-x2.mean(axis=0)
matr=numpy.matrix(temp.values)
sb=numpy.dot(matr.T,matr)
sw=numpy.matrix(sw)
sb=numpy.matrix(sb)
e,v=numpy.linalg.eig(numpy.dot(sw.I,sb))
print e
print v
w=v[:,1]
print w
x3_cov=x3.cov()
sw1=x1_cov+x2_cov+x3_cov
mean=pf.ix[:,0:3].mean(axis=0)
t1=numpy.matrix(x1.mean(axis=0)-mean)
t2=numpy.matrix(x2.mean(axis=0)-mean)
t3=numpy.matrix(x3.mean(axis=0)-mean)
sb1=len(x1)*numpy.dot(t1.T,t1)+len(x2)*numpy.dot(t2.T,t2)+len(x3)*numpy.dot(t3.T,t3)
sw1=numpy.matrix(sw1)
sb1=numpy.matrix(sb1)
e1,v1=numpy.linalg.eig(numpy.dot(sw1.I,sb1))
print e1
print v1
