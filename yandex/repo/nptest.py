import numpy as np
import pandas


data = pandas.read_csv('./data/titanic.csv', index_col='PassengerId')

all_count = len(data)
print 'all %s' % all_count

alive_count = len(np.nonzero(data['Survived'])[0])

print 'alive %s' % alive_count

fclass = len(np.nonzero(data['Pclass'] == 1)[0])

print 'fclass %s' % fclass

ages = data['Age']

print 'age mean %s median %s' % (np.mean(ages), np.nanmedian(ages))

import scipy.stats as stats

print stats.pearsonr(data['SibSp'], data['Parch'])

male_names = []
for k,p in enumerate(data['Name']):
    if data['Sex'][k+1] == 'male':
        male_names.extend(p.translate(None, "(),?.!/;:\"").split(' '))
        
    #print k,p,data['Sex'][k+1]
male_names = list(set(male_names))

female_names = []
for k,p in enumerate(data['Name']):
    if data['Sex'][k+1] == 'female':
        names = p.translate(None, "(),?.!/;:\"").split(' ')
        for n in names:
            if n not in male_names and n not in female_names:
                female_names.append(n)

female_names_1 = []
for k,p in enumerate(data['Name']):
    if data['Sex'][k+1] == 'female':
        names = p.translate(None, "(),?.!/;:\"").split(' ')
        for n in names:
            if n in female_names:
                female_names_1.append(female_names.index(n))

print np.bincount(female_names_1)
print female_names[:100]
