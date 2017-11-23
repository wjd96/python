#!/usr/bin/python2.7
import numpy
import pandas
import scipy
import sklearn
df=pandas.read_excel('/home/wjd/python/machinelearn/computerby.xls')
featurelist=[]
labelslist=[]
for row in df.values:
	labelslist.append(row[len(row)-1])
	rdict={}
	for i in range(0,len(row)-1):
		rdict[df.columns[i]]=row[i]
	featurelist.append(rdict)
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import tree
vec=DictVectorizer()
dummyx=vec.fit_transform(featurelist).toarray()
lb=preprocessing.LabelBinarizer()
dummyy=lb.fit_transform(labelslist)
clf=tree.DecisionTreeClassifier(criterion='entropy')
clf=clf.fit(dummyx,dummyy)
print(dummyx)
print(dummyy)

