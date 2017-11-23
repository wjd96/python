#!/usr/bin/python2.7
import numpy
import pandas
import os
from collections import Counter
import re
#count=[]
all=dict()
dir=os.listdir('/home/wjd/python/machinelearn/20_newsgroups')
for file in dir:
	#print file
	words=str()
#	if os.path.isdir('/home/wjd/python/machinelearn/20_enwsgroups/'+file):
#	print file
	files=os.listdir('/home/wjd/python/machinelearn/20_newsgroups/'+file)
	for f in files:
		ft=open('/home/wjd/python/machinelearn/20_newsgroups/'+file+'/'+f,'r')
		words=words+ft.read()
	all[file]=words
#print all
textm=dict()
for i in all.keys():
	count=dict()
	print i
	words=re.findall('[0-9a-zA-Z]+',all[i])
	count=Counter(words)
#	print count
	df=pandas.DataFrame(count.values(),index=count.keys(),columns=[i])
	textm[i]=df
#for i in textm.keys():
#	print textm[i]

#df1=df[df[0]>2]
filter=open('/home/wjd/python/machinelearn/filter','r')
filter=filter.read()
filter=re.findall('[0-9a-zA-Z]+',filter)
filter.append('The')
result=pandas.DataFrame()
for i in textm.keys():
#	print i
	df=textm[i]
	df1=df[df[i]>2]
	key=df1.index
	f=[x for x in key if x not in filter]
	dft=df1.ix[f]
#	print dft
	if len(result)==0:
		result=dft
	else:
		result=pandas.concat([result,dft],axis=1)
#print result
#key=df1.index
#f=[x for x in key if x not in filter]
#df.ix[f]
#print df.ix[f]





















print textm
