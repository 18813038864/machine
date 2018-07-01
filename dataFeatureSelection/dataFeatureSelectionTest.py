#coding=utf-8
from numpy import set_printoptions
from pandas import read_csv
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.linear_model import LogisticRegression

filename= 'pima_data.csv'
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(filename, names=names)

arr = data.values
x = arr[:, 0:8]
y = arr[:, 8]

# 单变量特征选取：统计变量与结果的相关性，选择相关性大的特征
# test = SelectKBest(score_func=chi2, k=4)
# fit = test.fit(x, y)
# set_printoptions(precision=3, linewidth=150)
# print fit.scores_
# features = fit.transform(x)
# print features

# 递归特征消除：使用一个模型进行多轮训练，每轮训练消除若干权值系数的特征
#
# fit = rfe.fit(x, y)model = LogisticRegression()
# rfe = RFE(model, 3)
# print 'the feature num is :'
# print fit.n_features_
# print 'the feature for support is :'
# print fit.support_
# print 'the feature for ranking:'
# print fit.ranking_

# 主要成分分析：数据降维
# pca = PCA(n_components=3)
# fit = pca.fit(x)
# print 'explained std:%s' % fit.explained_variance_ratio_
# print fit.components_

# 特征重要性
model = ExtraTreesClassifier()
fit = model.fit(x, y)
set_printoptions(linewidth=150)
print fit.feature_importances_