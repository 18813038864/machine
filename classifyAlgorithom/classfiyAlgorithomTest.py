#coding=utf-8
from pandas import read_csv
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

filename='pima_data.csv'
names=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data=read_csv(filename, names=names)
array = data.values
x=array[:, 0:8]
y=array[:, 8]
num_kfold = 10
seed = 7
kfold = KFold(n_splits=num_kfold, random_state=seed)
model = LogisticRegression()
model = LinearDiscriminantAnalysis()
model = KNeighborsClassifier()
model = GaussianNB()
model = DecisionTreeClassifier()
model = SVC()
result = cross_val_score(model, x, y, cv=kfold)
print result.mean()

