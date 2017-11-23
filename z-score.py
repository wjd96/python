#!/usr/bin/python2.7
import numpy
import pandas
def z_score(x):
	x=numpy.array(x).astype(float)
	xr=numpy.rollaxis(x,axis=1)
	xr-=numpy.mean(x,axis=1)
	xr/=numpy.std(x,axis=1)
	return xr.T
x=range(0,15)
ar=numpy.array(x)
print z_score(ar.reshape(3,5))
