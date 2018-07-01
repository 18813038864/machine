#coding=utf-8
from pandas import read_csv, set_option

filename = 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)
set_option('precision',2)
set_option('display.width',100)
print 'pandas read csv:',data.shape
print data.dtypes
print data.describe()
print data.groupby('class').size()
print data.corr(method='pearson')
print data.skew()
# peek = data.head(10)
# print peek