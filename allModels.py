import pandas as pd

# Loading Data in Pandas Data Frames
print 'reading data'
trainDF=pd.read_csv('train.csv',index_col=0)
testDF=pd.read_csv('test.csv',index_col=0)
eventTypeDF=pd.read_csv('event_type.csv',index_col=0)
logFeatureDF=pd.read_csv('log_feature.csv',index_col=0)
resourceTypeDF=pd.read_csv('resource_type.csv',index_col=0)
severityTypeDF=pd.read_csv('severity_type.csv',index_col=0)
eventTypeVect=pd.get_dummies(eventTypeDF).groupby(eventTypeDF.index).sum()
logFeatureVect=pd.get_dummies(logFeatureDF).groupby(logFeatureDF.index).sum()
resourceTypeVect=pd.get_dummies(resourceTypeDF).groupby(resourceTypeDF.index).sum()
severityTypeVect=pd.get_dummies(severityTypeDF).groupby(severityTypeDF.index).sum()

#merge test and train
data=pd.concat([trainDF,testDF],axis=0)
data=data.join(eventTypeVect).join(logFeatureVect).join(resourceTypeVect).join(severityTypeVect)
data.location=data.location.apply(lambda x: int(x.split(' ')[1]))

#preprocess data
bins=range(0,1300,100)
names=range(0,12,1)
data['loc_cat']=pd.cut(data.location,bins,labels=names)

df=pd.get_dummies(data.loc_cat,prefix='loc_cat')
data=pd.concat([data,df], axis=1)
data.drop('loc_cat',axis=1,inplace=True)

testDF=data[data.fault_severity.isnull()]
trainDF=data[data.fault_severity.notnull()]

import numpy as np
X=trainDF
X=X.drop(['location','volume','fault_severity'],axis=1)
X=X.as_matrix()
Y=trainDF.fault_severity
Y=Y.as_matrix()
X_test=testDF
X_test=X_test.drop(['location','volume','fault_severity'],axis=1)
X_test=X_test.as_matrix()

from sklearn import metrics
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# prepare configuration for cross validation test with basic classifier
num_folds = 5
num_instances = len(X)
seed = 7
print 'preparing models'
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', MultinomialNB()))
models.append(('SVM', SVC()))

# evaluate each model
results = []
names = []
scoring = 'accuracy'

for name, model in models:
	print 'evaluating '+ name
	kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, random_state=seed)
	cv_results = cross_validation.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

from matplotlib import pyplot as plt
# Creating boxplot for Model comparison
fig = plt.figure()
fig.suptitle('Model Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()