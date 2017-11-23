#!/usr/bin/python2.7
import numpy
import pandas
from sklearn import neighbors
from sklearn import datasets
knn=neighbors.KNeighborsClassifier()
iris=datasets.load_iris()
knn.fit(iris.data,iris.target)

