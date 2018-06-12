# -*- coding: utf-8 -*-
"""
Title: Kaggle Titanic (simple version using scikit-Learn)

@author: Vishnuvardhan Janapati
"""

#this is titanic data

import pandas as pd
#import time
import numpy as np
#import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from sklearn import svm, preprocessing, cross_validation

# reading training, testing data
train_df=pd.read_csv('train.csv')
test_df=pd.read_csv('test.csv')
PassengerId=test_df['PassengerId']

# data processing (train data)
#Avg_age=train_df['Age'].dropna().mean()
train_df['Age'].fillna(value=train_df.Age.median(),inplace=True)
train_df.Fare.replace(0,value=train_df.Fare.median(),inplace=True)
train_df=train_df.replace('male',1).replace('female',0)
train_df['Embarked'].fillna('S',inplace=True)
train_df['Embarked']=train_df['Embarked'].replace('C',1).replace('Q',2).replace('S',3)
train_df=train_df.drop(['Name','Ticket','Cabin'],1)
train_df.dropna(inplace=True)
y=np.array(train_df['Survived'])
train_df=train_df.drop(['Survived','SibSp','Parch'],1)
X=np.array(train_df)
X=preprocessing.scale(X)


# data processing (test data)
#Avg_age_test=test_df['Age'].dropna().mean()
#Avg_fare_test=test_df['Fare'].dropna().mean()

test_df['Age'].fillna(value=test_df['Age'].median(),inplace=True) # fill missing data with median
test_df.Fare.replace(0,value=test_df.Fare.median(),inplace=True) # fill missing data with median
test_df['Fare'].fillna(value=test_df.Fare.median(),inplace=True) # fill missing data with median
test_df=test_df.replace('male',1).replace('female',0)            # replace 'male' to 1 and female to 0
test_df['Embarked'].fillna('S',inplace=True)                    # fill missing feature with starting location
test_df['Embarked']=test_df['Embarked'].replace('C',1).replace('Q',2).replace('S',3)
test_df=test_df.drop(['Name','Ticket','Cabin'],1)   #drop features
test_df.count()  # to see how many data sets (excluding NA) are there for every feature
test_df.dropna(inplace=True)
test_df=test_df.drop(['SibSp','Parch'],1) #drop features
X_test=np.array(test_df)
X_test=preprocessing.scale(X_test)

## ------------ Test model parameters and optimize for better performance before testing with unlabel data 
#X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)
#clf=svm.SVC(kernel='linear')
#clf.fit(X_train,y_train)
#print("The accuracy of prediction with train set is " + str(round(clf.score(X_train,y_train)*100,2)) +" Percent")
#print("The accuracy of prediction with test set is " + str(round(clf.score(X_test,y_test)*100,2)) +" Percent")

## ------------ Train with given label data (X,y), and test model accuracy with unlabel data 
clf=svm.SVC(kernel='linear')
clf.fit(X,y)


# Accuracy of prediction with train set data. This should be as high as possible but don't overfit the model
print("The accuracy of prediction with train set is " + str(round(clf.score(X,y)*100,2)) +" Percent")

predictions=clf.predict(X_test)
submission=pd.DataFrame({'PassengerId':PassengerId,'Survived':predictions})
submission.to_csv('submission.csv',index=False)