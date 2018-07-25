#coding=utf-8
import pandas as pd
from pandas import read_csv
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold, cross_val_score, train_test_split

# filename = 'pima_data.csv'
# names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# data = read_csv(filename, names=names)
# array = data.values
# x = array[:, 0:8]
# y = array[:, 8]
# num_folds = 10
# seed = 7
# kfold = KFold(n_splits=num_folds, random_state=seed)
# model = LogisticRegression()
# scoring = 'roc_auc'
# result = cross_val_score(model, x, y, cv=kfold, scoring=scoring)
# print ("算法评估结果准确度：%.3f (%.3f)" % (result.mean(), result.std()))
# test_size = 0.33
# seed = 4
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)
# model = LogisticRegression()
# model.fit(x_train, y_train)
# predcited = model.predict(x_test)
# matrix = confusion_matrix(y_test,predcited)
# classes = ['0','1']
# dataframe = pd.DataFrame(data=matrix, index=classes, columns=classes)
#
# print dataframe
# report = classification_report(y_test, predcited)
#
# print report

filename='housing.csv'
names=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PRTATIO', 'B', 'LSTAT', 'MEDV']
data = read_csv(filename, names=names, delim_whitespace=True)
array=data.values
x=array[:, 0:13]
y=array[:, 13]
n_splits = 10
seed = 7
kfold = KFold(n_splits=n_splits, random_state=seed)
model = LinearRegression()
scoring = 'neg_mean_absolute_error'
result = cross_val_score(model, x, y, cv=kfold, scoring=scoring)
print 'mae: %.3f (%.3f)' %(result.mean(), result.std())








