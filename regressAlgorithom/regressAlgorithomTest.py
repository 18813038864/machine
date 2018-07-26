#coding=utf-8
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

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
model = Ridge()
model = Lasso()
model = ElasticNet()
model = KNeighborsRegressor()
model = DecisionTreeRegressor()
model = SVR()
scoring = 'neg_mean_absolute_error'
result = cross_val_score(model, x, y, cv=kfold, scoring=scoring)