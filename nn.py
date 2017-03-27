import pandas as pd
import numpy as np

# Loading Data in Pandas Data Frames
trainDF=pd.read_csv('train.csv',index_col=0)
testDF=pd.read_csv('test.csv',index_col=0)
eventTypeDF=pd.read_csv('event_type.csv',index_col=0)
logFeatureDF=pd.read_csv('log_feature.csv',index_col=0)
resourceTypeDF=pd.read_csv('resource_type.csv',index_col=0)
severityTypeDF=pd.read_csv('severity_type.csv',index_col=0)

#Vectorizing individual DF
eventTypeVect=pd.get_dummies(eventTypeDF).groupby(eventTypeDF.index).sum()
logFeatureVect=pd.get_dummies(logFeatureDF).groupby(logFeatureDF.index).sum()
resourceTypeVect=pd.get_dummies(resourceTypeDF).groupby(resourceTypeDF.index).sum()
severityTypeVect=pd.get_dummies(severityTypeDF).groupby(severityTypeDF.index).sum()

#merge test and train
data=pd.concat([trainDF,testDF],axis=0)
data=data.join(eventTypeVect).join(logFeatureVect).join(resourceTypeVect).join(severityTypeVect)

#convert location to int
data.location=data.location.apply(lambda x: int(x.split(' ')[1]))

#preprocess data

#binning location
bins=range(0,1300,50)  #logloss 0.59
names=range(0,25,1)
data['loc_cat']=pd.cut(data.location,bins,labels=names)
df=pd.get_dummies(data.loc_cat,prefix='loc_cat')
data=pd.concat([data,df], axis=1)
data.drop('loc_cat',axis=1,inplace=True)


#Transforming Volume and then binning
data['volumeTransformed']=data.volume.apply(np.log) #logloss 0.577
bins=range(0,9,1)
names=range(0,8,1)
data['vol_trans']=pd.cut(data.volumeTransformed,bins,labels=names)
df=pd.get_dummies(data.vol_trans,prefix='vol_trans')
data=pd.concat([data,df], axis=1)
data.drop(['vol_trans','volumeTransformed'],axis=1,inplace=True)

#seperating test and train data
testDF=data[data.fault_severity.isnull()]
trainDF=data[data.fault_severity.notnull()]
testDF.columns
import ipdb; ipdb.set_trace()
#preparing test and train data
X=trainDF
X=X.drop(['location','volume','fault_severity'],axis=1)
X=X.as_matrix()

Y=trainDF.fault_severity
Y=Y.as_matrix()
X_test=testDF
X_test=X_test.drop(['location','volume','fault_severity'],axis=1)
X_test=X_test.as_matrix()
#importing necessary libraries
from sklearn import metrics
from sklearn import cross_validation
from sklearn.neural_network import MLPClassifier

#training Neural Network
clf = MLPClassifier(hidden_layer_sizes=(600,300,40,10,10),max_iter=150, alpha=1e-4, solver='sgd',verbose=True, \
	tol=0.0001, learning_rate_init=0.0005)
print "training classifier ..."
clf.fit(X,Y)

#Performing CV with k=10
# print "performing cross fold validation ..."
# predicted = cross_validation.cross_val_predict(clf, X, Y, cv=10)

# print "accuracy score: ", metrics.accuracy_score(Y, predicted)
# print "precision score: ", metrics.precision_score(Y, predicted,average='weighted')
# print "recall score: ", metrics.recall_score(Y, predicted,average='weighted')
# print "classification_report: \n ", metrics.classification_report(Y, predicted)
# print "confusion_matrix:\n ", metrics.confusion_matrix(Y,predicted)
# print "log loss :", metrics.log_loss(Y,clf.predict_proba(X))

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
submission.to_csv(open('submission_nn.csv','wt'),index=False)
print 'submission file created...'