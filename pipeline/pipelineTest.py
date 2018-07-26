#coding=utf-8
from pandas import read_csv
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import FeatureUnion, Pipeline

filename='pima_data.csv'
names=['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data=read_csv(filename, names=names)
array = data.values
x=array[:, 0:8]
y=array[:, 8]
num_kfold = 10
seed = 7
kfold = KFold(n_splits=num_kfold, random_state=seed)

features = []
features.append(('pca',PCA()))
features.append(('select_best',SelectKBest(k=6)))

steps = []
steps.append(('feature_union',FeatureUnion(features)))
steps.append(('logistic', LogisticRegression()))

model = Pipeline(steps)
result = cross_val_score(model, x, y, cv=kfold)
print result.mean()