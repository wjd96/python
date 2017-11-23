#!/usr/bin/python2.7
import numpy
import pandas
from sklearn import svm
import pylab
clf=svm.SVC(kernel='linear')
x=[numpy.random.randn(20,2)-[2,2],numpy.random.randn(20,2)+[2,2]]
xt=[i.tolist() for i in x]
x=xt[0]+xt[1]
y=[0]*20+[1]*20
clf.fit(x,y)
#print (clf)
#print (clf.support_vectors_)
#print(clf.predict([2,0]))
w=clf.coef_[0]
a=-w[0]/w[1]
xx=numpy.linspace(-5,5)
yy=a*xx-(clf.intercept_[0])/w[1] 
#pylab.plot(xx,yy,'k--')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(1,1)
axes.plot(xx,yy,'k--')
axes.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1])
axes.scatter(numpy.array(x)[:,0],numpy.array(x)[:,1])
plt.show()
