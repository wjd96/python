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
	
df=pandas.read_csv('/home/wjd/python/machinelearn/12.txt',names=[1,2,3,4,'color'])
sample=df.values[:,0:4]
y=pandas.get_dummies(df['color'])
values=y.values
sample=sample[0:20]
values=values[0:20]
df1=pandas.read_csv('/home/wjd/python/machinelearn/13.txt',names=[1,2,3,4,'color'])
tx=df1.values[:,0:4]
ty=pandas.get_dummies(df1['color'])
ty1=ty.values
#tx=tx[0:5]
#ty1=ty1[0:5]
arr=[len(sample[0]),4,len(values[0])]
dw={}
do={}
result=[]
#f = open("/home/wjd/python/a.txt", "w")
for i in range(0,len(arr)-1):
	dw[i]=pandas.DataFrame(numpy.random.randn(arr[i],arr[i+1]))
	do[i+1]=pandas.DataFrame(numpy.random.randn(arr[i+1]))
def deeplearn(sample,values,arr,dw,do,l):
	count=0.0
        for i in range(0,len(sample)):
                y={}
                forward(sample[i],dw,do,y)
                t=values[i]
                esty=y[len(y)-1]
		#print 't:',t
		#print 'esty:',esty
               # l=0.5
                perr=[esty[i]*(1-esty[i])*(t[i]-esty[i]) for i in range(0,len(esty))]
                backward(arr,dw,do,y,perr,l)
		if esty.index(numpy.array(esty).max())==t.tolist().index(t.max()):
			count=count+1
#		f.write(str(t)+'\n')
#		f.write(str(esty)+'\n')
                #print t
                #print esty
	print 'accuracy:',100*(count/len(sample)),'%   step:',l
	return count
l=0.1
for i in range(0,220):
	print 'compute time:', i
	count=deeplearn(sample,values,arr,dw,do,l)
#	if 100*(count/len(sample))>90:
#		l=0.01
#	if 100*(count/len(sample))>=95:
#		l=0.0001
#	result.append(count)
#	if count<numpy.array(result).max()&l>0:
#		l=l-0.01		
#  	if result[-1]==result[-2]&l>0:
#		l=l+0.02
#	print 'accuracy:',100*(count/len(sample)),'%   step:',l
#f.close()
print '******************************************************************************'
deeplearn(tx,ty1,arr,dw,do,l)
 #print 'accuracy:',100*(count/len(sample)),'%   step:',l
