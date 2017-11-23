#!/usr/bin/python2.7
import numpy
import pandas
def forward(x,dw,do,y):
	value=x
	y[0]=value
	for i in range(0,len(dw)):
		value=com(value,dw[i],do[i+1])
		y[i+1]=value

def com(xz,dwz,doz):   
	values=[] 
	for i in dwz.columns[:]:
		a=(xz*dwz[i]).sum()+doz.values[i]
		e=1/(1+numpy.exp(-a))
		values.append(e.tolist())
	temp=numpy.reshape(values,[1,len(doz)])
	value=temp[0].tolist()
	return value   
def error(dw,y,err):
	E=[]
	for i in dw.index:
		a=(dw.ix[i,:]*err).sum()
		E.append(a)
	e1=numpy.reshape(E,[1,len(E)])
	e2=e1[0].tolist()
	y1=[y[i]*(1-y[i]) for i in range(0,len(y))]
	values=[y1[i]*e2[i] for i in range(0,len(y1))]
	return values
def ew(err,y,dw,l):
	for i in  dw.index:
		for j in dw.columns:
			dw.ix[i,j]=dw.ix[i,j]+l*err[j]*y[i]
def eo(err,do,l):
	errt=[i*l for i in err]
	do[0]=do[0]+errt
def backward(arr,dw,do,y,perr,l):
	rang=range(0,len(arr)-1)
	rang.reverse()
	err=perr
	for i in rang:
		err1=error(dw[i],y[i],err)
		ew(err,y[i],dw[i],l)
		eo(err,do[i+1],l)
		err=err1
	
#df=pandas.read_csv('/home/wjd/python/machinelearn/12.txt',names=[1,2,3,4,'color'])
#sample=df.values[:,0:4]
#y=pandas.get_dummies(df['color'])
#values=y.values
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
from sklearn.datasets import load_digits   
digits=load_digits()  
x=digits.data
y=digits.target
y=pandas.get_dummies(y)
trainx=pca(x[0:100].T)
testx=pca(x[1700:len(x)].T)
trainy=pca(y.values[0:100].T)
testy=pca(y.values[1700:len(y)].T)
print len(trainx[0])
arr=[len(trainx[0]),32,len(trainy[0])]
dw={}
do={}
#f = open("/home/wjd/python/a.txt", "w")
for i in range(0,len(arr)-1):
	dw[i]=pandas.DataFrame(numpy.random.randn(arr[i],arr[i+1]))
	do[i+1]=pandas.DataFrame(numpy.random.randn(arr[i+1]))
def deeplearn(sample,values,arr,dw,do):
	count=0;
        for i in range(0,len(sample)):
                y={}
                forward(sample[i],dw,do,y)
                t=values[i]
                esty=y[len(y)-1]
                l=0.1
                perr=[esty[i]*(1-esty[i])*(t[i]-esty[i]) for i in range(0,len(esty))]
                backward(arr,dw,do,y,perr,l)
	#	value=[]
	#	f.write(str(t.tolist().index(t.max()))+'\n')
	#	for j in esty:
    	#		if j==numpy.array(esty).max():
        #			value.append(1)
    	#		else:
        #			value.append(0)
#		f.write(str(esty.index(numpy.array(esty).max()))+'\n')
		if esty.index(numpy.array(esty).max())==t.tolist().index(t.max()):
			count=count+1
#	f.write(str(count)+'\n')
	print count                
#print t
                #print esty
for i in range(0,250):
	print i
#	f.write('********************************************************************'+'\n')
	deeplearn(trainx,trainy,arr,dw,do)
#f.close()
print('*************************************************************************')
deeplearn(testx,testy,arr,dw,do)
