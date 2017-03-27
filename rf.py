import pandas as pd
# Loading Data in Pandas Data Frames
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

# bins=range(0,1300,100)  #logloss 0.62
bins=range(0,1300,50)  #logloss 0.59
# names=range(0,12,1)
names=range(0,25,1)
data['loc_cat']=pd.cut(data.location,bins,labels=names)

df=pd.get_dummies(data.loc_cat,prefix='loc_cat')
data=pd.concat([data,df], axis=1)
data.drop('loc_cat',axis=1,inplace=True)

# data.volume.apply(np.log).hist()
# data['volumeTransformed']=data.volume.apply(np.log)
# bins=range(0,9,1)
# names=range(0,8,1)
# data['vol_trans']=pd.cut(data.volumeTransformed,bins,labels=names)
# df=pd.get_dummies(data.vol_trans,prefix='vol_trans')
# data=pd.concat([data,df], axis=1)
# data.drop(['vol_trans','volumeTransformed'],axis=1,inplace=True)

testDF=data[data.fault_severity.isnull()]
trainDF=data[data.fault_severity.notnull()]

from matplotlib import pyplot as plt
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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.naive_bayes import MultinomialNB,BernoulliNB,GaussianNB

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# clf = MLPClassifier(hidden_layer_sizes=(40,40,40,40,40), alpha=1e-4, solver='sgd', verbose=True, \
# 	tol=0.0001, learning_rate_init=.001)
# clf=DecisionTreeClassifier(criterion="entropy")
clf=RandomForestClassifier(n_estimators=1000,max_depth=5,min_samples_split=2)
print "training classifier ..."
clf.fit(X,Y)
print "performing cross fold validation ..."
predicted = cross_validation.cross_val_predict(clf, X, Y, cv=5)

print "accuracy score: ", metrics.accuracy_score(Y, predicted)
print "precision score: ", metrics.precision_score(Y, predicted,average='weighted')
print "recall score: ", metrics.recall_score(Y, predicted,average='weighted')
print "classification_report: \n ", metrics.classification_report(Y, predicted)
print "confusion_matrix:\n ", metrics.confusion_matrix(Y,predicted)
print "log loss :", metrics.log_loss(Y,clf.predict_proba(X))
Y_test=clf.predict_proba(X_test)
# create submission
submission = pd.DataFrame(Y_test,columns=['predict_0','predict_1','predict_2'])
submission.head()
submission['id']=testDF.index.values
cols=submission.columns.tolist()
cols=cols[-1:] + cols[:-1]
submission=submission[cols]
submission.to_csv(open('submission.csv','wt'),index=False)

