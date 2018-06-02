#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

# load the data
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# transform to pandas dataframe
df = pd.DataFrame()
df = df.from_dict(data_dict, orient = 'index', dtype = float)

# set data types
df = df.astype(dtype = {'poi': bool})
df.replace(to_replace = 'NaN', value = np.nan, inplace = True)

# remove 'LOCKHART EUGENE E'
df = df.drop('LOCKHART EUGENE E')
# remove "THE TRAVEL AGENCY IN THE PARK"
df = df.drop('THE TRAVEL AGENCY IN THE PARK')
# remove outliers
df = df.drop('TOTAL')
# get row names
row_names = df.index

# create a new boolean feature indicating whether a person has an email address or not
df['email_account'] = (df.email_address.notnull()).astype(float)
# replace the str type email feature
df = df.drop('email_address', axis = 1)
# create 3 ratio features regarding email conversations
df['ratio_from_this_to_poi'] = df.from_this_person_to_poi / df.from_messages
df['ratio_from_poi_to_this'] = df.from_poi_to_this_person / df.to_messages
df['ratio_shared_receipt_with_poi'] = df.shared_receipt_with_poi / df.to_messages

# impute NaN values with 0
df.fillna(0, inplace = True)

# format the dataset
X = df.select_dtypes(exclude = ['bool']).values.reshape(143,23)
y = np.asarray(df.select_dtypes(include = ['bool']).astype(float)).flatten().reshape(143,)
# extract the features' names
feature_name = df.select_dtypes(exclude = ['bool']).columns.values.astype(str)

# scale features
X_scaler = StandardScaler()
X = X_scaler.fit_transform(X)

# select features, tune 2 algorithms with cross-validation
print "Have a cup of tea and relax..."
# I actually tried more K's than what shows below
# reduced numbers here to save a bit of time
# warning messages may pop up during cross-validation
for K in [17, 15, 14, 11]:

    
    selection = SelectKBest(k = K)
    selection.fit(X, y)
    X = selection.transform(X)
    print "With %d Best Features:" %K
    
    if K == 15:
        X1 = X
    
    cv = StratifiedShuffleSplit(n_splits = 100, random_state = 42)
    
    # Try and tune SVM
    parameters = {'kernel':('linear', 'poly', 'sigmoid'),\
                  'C':[0.001, 0.01, 0.1, 1, 10, 50, 100, 1000],\
                  'gamma': [0.001, 0.01, 0.05]}
    from sklearn.svm import SVC
    svc = SVC()
    clf1 = GridSearchCV(svc, parameters, 'precision', cv = cv)
    clf1.fit(X, y)
    print "Best Parameter for SVM:", clf1.best_params_, " Best Score:", clf1.best_score_
    
    # Try and tune logistic regression
    parameters = {'solver':('liblinear', 'sag'), 'C':[0.001, 0.01, 0.1, 1, 10, 100], 'max_iter':[100, 1000]}
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    clf2 = GridSearchCV(lr, parameters, 'precision', cv = cv)
    clf2.fit(X, y)
    print "Best Parameter for LR:", clf2.best_params_, " Best Score:", clf2.best_score_
    
# store features_list
feature_name = zip(feature_name, selection.scores_)
feature_rank = sorted(feature_name, key = lambda x: x[1], reverse = True)
feature_rank = np.asarray(feature_rank)
features_list = ['poi']
features_list.extend(feature_rank[:15,0])

# build classifier
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(C = 0.01, max_iter = 100, solver = 'liblinear')

# write my data to dictionary for output
mydf = pd.DataFrame()
mydf = mydf.from_records(X1)
mydf.columns = features_list[1:]
mydf.index = row_names
mydf['poi'] = y
my_dataset = mydf.to_dict(orient='index')

dump_classifier_and_data(clf, my_dataset, features_list)