import pandas as pd
# Loading Data in Pandas Data Frames
trainDF=pd.read_csv('train.csv',index_col=0)
testDF=pd.read_csv('test.csv',index_col=0)
trainDF.location=trainDF.location.apply(lambda x: int(x.split(' ')[1]))
testDF.location=testDF.location.apply(lambda x: int(x.split(' ')[1]))

eventTypeDF=pd.read_csv('event_type.csv',index_col=0)
eventTypeDF.event_type=eventTypeDF.event_type.apply(lambda x: int(x.split(' ')[1]))

logFeatureDF=pd.read_csv('log_feature.csv',index_col=0)
logFeatureDF.log_feature=logFeatureDF.log_feature.apply(lambda x: int(x.split(' ')[1]))

resourceTypeDF=pd.read_csv('resource_type.csv',index_col=0)
resourceTypeDF.resource_type=resourceTypeDF.resource_type.apply(lambda x: int(x.split(' ')[1]))

severityTypeDF=pd.read_csv('severity_type.csv',index_col=0)
severityTypeDF.severity_type=severityTypeDF.severity_type.apply(lambda x: int(x.split(' ')[1]))

eventTypeVect=pd.get_dummies(eventTypeDF).groupby(eventTypeDF.index).sum()
logFeatureVect=pd.get_dummies(logFeatureDF).groupby(logFeatureDF.index).sum()
resourceTypeVect=pd.get_dummies(resourceTypeDF).groupby(resourceTypeDF.index).sum()
severityTypeVect=pd.get_dummies(severityTypeDF).groupby(severityTypeDF.index).sum()
trainDF=trainDF.join(eventTypeDF).join(logFeatureDF).join(resourceTypeDF).join(severityTypeDF)
print trainDF.head()