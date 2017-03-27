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


import numpy as np

#preprocess data
data['volumeTransformed']=data.volume.apply(np.log)  #0.613

# import ipdb; ipdb.set_trace()
data['index1']=data.index  #0.604
locCount=data.groupby('location').count()[['index1']]  #count of locations
data=pd.merge(data,locCount, how='inner', left_on='location', right_index=True)  

# normalization
# data['volumeTransformed']=data.volumeTransformed.apply(lambda x: (x-np.mean(data.volumeTransformed))/np.std(data.volumeTransformed))
# data['location']=data.location.apply(lambda x: (x-np.mean(data.location))/np.std(data.location))
# cols_to_norm = ['volumeTransformed','location']
# data[cols_to_norm] = data[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))


testDF=data[data.fault_severity.isnull()]
trainDF=data[data.fault_severity.notnull()]


X=trainDF
X=X.drop(['index1_x','volume','fault_severity'],axis=1)
X=X.as_matrix()
Y=trainDF.fault_severity
Y=Y.as_matrix()
X_test=testDF
X_test=X_test.drop(['index1_x','volume','fault_severity'],axis=1)
X_test=X_test.as_matrix()

#import scikit libraries
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn import metrics
from sklearn.grid_search import RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC,SVC
	
#Logistic regression classifier
clf=LogisticRegression(C=1e3, penalty='l2') #ll 0.608 c=1e3

# clf = OneVsRestClassifier(SVC(C=100,kernel = 'linear', verbose= False, probability=True))

clf.fit(X,Y)

#Performing CV with k=10
predicted = cross_validation.cross_val_predict(clf, X, Y, cv=10)

#print metrics
print "accuracy score: ", metrics.accuracy_score(Y, predicted)
print "precision score: ", metrics.precision_score(Y, predicted,average='weighted')
print "recall score: ", metrics.recall_score(Y, predicted,average='weighted')
print "log loss :", metrics.log_loss(Y,clf.predict_proba(X))
print "confusion_matrix:\n ", metrics.confusion_matrix(Y,predicted)
print "classification_report: \n ", metrics.classification_report(Y, predicted)
# fpr,tpr,threshold= metrics.roc_curve(Y,predicted)
# print metrics.auc(fpr,tpr)
#Create prediction for test
Y_test=clf.predict_proba(X_test)

# create submission file
submission = pd.DataFrame(Y_test,columns=['predict_0','predict_1','predict_2'])
submission.head()
submission['id']=testDF.index.values
cols=submission.columns.tolist()
cols=cols[-1:] + cols[:-1]
submission=submission[cols]

#save to csv
submission.to_csv(open('submission.csv','wt'),index=False)
print'Submission file created..'