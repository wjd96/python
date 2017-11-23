#!/usr/bin/python2.7
import numpy
import pandas
from sklearn.datasets import load_digits
digits=load_digits()  
xe=digits.data  
def pca(x):
#pf=pandas.DataFrame(x)
	mn=x.mean(axis=1)
	tp=[]
	for i in range(0,len(x[0])):
		tp.append(list(x[:,i]-mn))
	result=numpy.array(tp).transpose()
	c=numpy.dot(result,result.T)/len(result[0])
	mat=numpy.matrix(result)
	e,v=numpy.linalg.eig(c)
	return numpy.dot(v[:20],x).T
print pca(xe.T)
print pca(xe.T).shape




