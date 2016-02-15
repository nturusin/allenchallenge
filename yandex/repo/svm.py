# coding=utf-8

import pandas
import numpy as np
from sklearn.svm import SVC
from sklearn.cross_validation import KFold
from sklearn import svm, grid_search

#data = pandas.read_csv('./data/svm-data.csv', header=None, names=[1,2,3])

#print data

#X = data[[2,3]]
#y = data[1]

#model = SVC(C=100000, random_state=241)
#model.fit(X,y)
#print model.support_vectors_

from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer

newsgroups = datasets.fetch_20newsgroups(
                    subset='all', 
                    categories=['alt.atheism', 'sci.space']
             )

X = newsgroups.data
y = newsgroups.target

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

#grid = {'C': np.power(10.0, np.arange(-5, 6))}

#cv = KFold(y.size, n_folds=5, shuffle=True, random_state=241)
#clf = SVC(kernel='linear', random_state=241)
#gs = grid_search.GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
#gs.fit(X, y)

#print gs.best_score_
#print gs.best_estimator_
#print gs

model = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='linear',
  max_iter=-1, probability=False, random_state=241, shrinking=True,
  tol=0.001, verbose=False)
model.fit(X, y)

features = vectorizer.get_feature_names()
indexes = pandas.Series(model.coef_.toarray().reshape(-1)).abs().nlargest(10).index

print " ".join(sorted([features[i] for i in indexes]))


