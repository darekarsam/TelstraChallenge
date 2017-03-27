import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

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


X=trainDF
X=X.drop(['location','volume','fault_severity'],axis=1)
X=X.as_matrix()
Y=trainDF.fault_severity
Y=Y.as_matrix()
X_test=testDF
X_test=X_test.drop(['location','volume','fault_severity'],axis=1)
X_test=X_test.as_matrix()

from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from mpl_toolkits.mplot3d import Axes3D

pca = PCA(n_components=3)

# import ipdb;ipdb.set_trace()
X_r = pca.fit(X).transform(X)


# lda = LinearDiscriminantAnalysis(n_components=2)
# X_r2 = lda.fit(X, Y).transform(X)
# Percentage of variance explained for each components
print('explained variance ratio (first two components): %s'
      % str(pca.explained_variance_ratio_))

fig=plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['navy', 'turquoise', 'darkorange']
lw = 2
target_names=['predict_0','predict_1','predict_2']

for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    ax.scatter(X_r[Y == i, 0], X_r[Y == i, 1], X_r[Y == i, 2], color=color, alpha=.8, lw=lw,
                label=target_name)
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.title('PCA of Data')

# plt.figure()
# for color, i, target_name in zip(colors, [0, 1, 2], target_names):
#     plt.scatter(X_r2[Y == i, 0], X_r2[Y == i, 1], alpha=.8, color=color,
#                 label=target_name)
# plt.legend(loc='best', shadow=False, scatterpoints=1)
# plt.title('LDA of Data')


plt.show()

import ipdb;ipdb.set_trace()
h=0.02 #step size in mesh

clf=LogisticRegression(C=1e5, penalty='l2')
clf.fit(X,Y)
predicted = cross_validation.cross_val_predict(clf, X, Y, cv=5)
print "accuracy score: ", metrics.accuracy_score(Y, predicted)
print "precision score: ", metrics.precision_score(Y, predicted,average='weighted')
print "recall score: ", metrics.recall_score(Y, predicted,average='weighted')

Y_test=clf.predict_proba(X_test)
# create submission
submission = pd.DataFrame(Y_test,columns=['predict_0','predict_1','predict_2'])
submission.head()
submission['id']=testDF.index.values
cols=submission.columns.tolist()
cols=cols[-1:] + cols[:-1]
submission=submission[cols]
submission.to_csv(open('submission.csv','wt'),index=False)

# import ipdb;ipdb.set_trace()
# x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
# y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# import ipdb;ipdb.set_trace()
# plt.figure(1, figsize=(4, 3))
# plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# # Plot also the training points
# plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')

# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.xticks(())
# plt.yticks(())

# plt.show()
