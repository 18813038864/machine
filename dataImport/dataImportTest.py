#coding=utf-8
from _csv import reader

import numpy as np
from pandas import read_csv

filename = 'pima_data.csv'
with open(filename, 'rt') as raw_data:
    readers = reader(raw_data, delimiter=',')
    x = list(readers)
    data = np.array(x).astype('float')
    print "csv reader read csv:",data.shape

with open(filename, 'rt') as raw_data:
    data = np.loadtxt(raw_data, delimiter=',')
    print 'numpy loadtxt read csv:',data.shape

names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
print 'pandas read csv:',data.shape