#coding=utf-8
from matplotlib import pyplot
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
model = {}
model['LR'] = LogisticRegression()
model['LD'] = LinearDiscriminantAnalysis()
model['KN'] = KNeighborsClassifier()
model['NB'] = GaussianNB()
model['DTC'] = DecisionTreeClassifier()
model['SVC'] = SVC()

results = []
for name in model:
    result = cross_val_score(model[name], x, y, cv=kfold)
    results.append(result)
    msg = '%s:%.3f(%.3f)' %(name, result.mean(), result.std())
    print msg

flg = pyplot.figure()
flg.suptitle('Algorithm cpmparision')
ax = flg.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(model.keys())
pyplot.show()

