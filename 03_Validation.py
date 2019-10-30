import os
import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
from sklearn.model_selection import GridSearchCV
from datetime import datetime

os.chdir(r'C:\Users\Ricardo\Desktop\PITAO\[PROJ] CartaoContinente')

with open('train_test_set.pickle', 'rb') as f:
    train, test = pickle.load(f)

def modelEvaluation(train, test, sampling = 'normal', n_jobs = None):

    for _ in np.asarray(train.select_dtypes(include = 'category').columns):
        encoder = LabelEncoder()
        train[_] = encoder.fit_transform(train[_])
        test[_] = encoder.fit_transform(test[_])

    x_train, y_train = [np.asarray(train.drop('churned', 1)), np.asarray(train['churned'])]
    x_test, y_test = [np.asarray(test.drop('churned', 1)), np.asarray(test['churned'])]

    grid_values = {
        'n_estimators' : [900, 1200, 1500],
        'criterion' : ['gini', 'entropy'],
        'max_depth' : [7, 8, 9],
        'max_features' : [5, 7, 9]
    }

    model = RandomForestClassifier()

    if sampling == 'smote':
        smote = SMOTE()
        x_train, y_train = smote.fit_sample(x_train, y_train)
    if sampling == 'nearmiss':
        nm = NearMiss()
        x_train, y_train = nm.fit_sample(x_train, y_train)

    grid = GridSearchCV(model, param_grid = grid_values, scoring = 'f1', cv = 3, n_jobs = n_jobs)
    grid.fit(x_train, y_train)
    best_params = grid.best_params_
    predictions = grid.predict(x_test)
    rf_accuraccy = accuracy_score(predictions, y_test)
    rf_f1_score = f1_score(predictions, y_test)

    with open('ValidationResults.txt', 'a+') as f:
        f.write('{}\n'.format(datetime.now()))
        f.write(name.upper()+':\n')
        f.write(sampling.upper()+':\n')
        f.write('Parameters found: {}\nAccuracy: {}\nF1_Score :{}\n\n\n'.format(best_params, rf_accuraccy, rf_f1_score))
        f.close()
    
    return best_params, rf_accuraccy, rf_f1_score

for mode in ['normal', 'SMOTE', 'nearmiss']:
    print(modelEvaluation(train, test, sampling = mode, n_jobs = -1))

"""
BEST F1 SCORE:

SMOTE:
Parameters found: {'criterion': 'gini', 'max_depth': 7, 'max_features': 5, 'n_estimators': 900}
Accuracy: 0.787793542074364
F1_Score :0.6259969821082131

"""
