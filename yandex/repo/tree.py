import numpy as np
import pandas
from sklearn.tree import DecisionTreeClassifier
#X = np.array([[1, 2], [3, 4], [5, 6]])
#y = np.array([0, 1, 0])
#clf = DecisionTreeClassifier()
#clf.fit(X, y)

data = pandas.read_csv('./data/titanic.csv', index_col='PassengerId')

d = {
    'male': 0,
    'female': 1
}


#data = data.dropna()
#data.where(data == 'male', 'm', inplace=True)
data['Bsex'] = data['Sex'].map(d) 
#data = pandas.merge(data[['Pclass', 'Fare', 'Age']], pandas.DataFrame(data['Sex']))

data = data[['Pclass', 'Fare', 'Age', 'Bsex', 'Survived']]
#data = data.dropna()
print data

X = data[['Pclass', 'Fare', 'Age', 'Bsex']]
print X.head()
y = data['Survived']
print y.head()

clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)

importances = clf.feature_importances_
print importances

print pandas.DataFrame(clf.feature_importances_, columns = ["Imp"], index = X.columns).sort_values(by=['Imp'], ascending = False)
