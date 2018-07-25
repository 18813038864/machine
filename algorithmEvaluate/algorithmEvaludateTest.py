#coding=utf-8
from pandas import read_csv
from sklearn.model_selection import train_test_split, KFold, cross_val_score, LeaveOneOut
from sklearn.linear_model import LogisticRegression

filename = 'pima_data.csv'
names = ['preg', 'plas', 'press', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

arr = data.values
x = arr[:, 0:8]
y = arr[:, 8]
# test_size = 0.33
# seed = 4
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed)
# model = LogisticRegression()
# model.fit(x_train,y_train)
# result = model.score(x_test, y_test)
# print ("algorithm result: %.3f%%" % (result*100))
# num_folds = 10
# seed = 7
# kfold = KFold(n_splits=num_folds, random_state=seed)
# model = LogisticRegression()
# result = cross_val_score(model, x, y, cv=kfold)
# print ("算法评估结果：%.3f%% (%.3f%%)") % (result.mean() * 100, result.std()*100)
loocv = LeaveOneOut()
model = LogisticRegression()
result = cross_val_score(model, x, y, cv=loocv)
print ("算法评估结果：%.3f%% (%.3f%%)" % (result.mean() * 100, result.std()*100))
