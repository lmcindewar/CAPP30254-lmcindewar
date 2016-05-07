import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid, GridSearchCV
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import pylab as pl
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.pipeline import Pipeline
import time
import numpy as np
from evaluation import evaluation_dict, build_comparison, plot_metrics
import json

def create_samples(df, fraction_for_testing):
    '''Given a dataframe and fraction for testing data (given as decimal), splits
    the data into random samples of training and testing data. Fraction for testing
    input as decimals, eg. 0.1 for 10 pct.
    Returns the test sample and training samples'''

    sample_n = df.shape[0]
    testing_n = round(sample_n * fraction_for_testing)
    testing_sample = random.sample(range(sample_n), testing_n)
    testing_data = df.iloc[testing_sample]
    training_data = df.drop(df.index[testing_sample], axis = 0)
    return testing_data, training_data

def splitX_y(df, y_column):
    '''Splits a dataframe into the explanatory data and the outcome variable.
    Takes a dataframe and returns the dataframe of features and datframe of
    outcome variables.'''

    y = df[y_column].copy()
    X = df.drop(y_column, axis = 1)
    return X, y

def logistic_reg(xdf, y):
    '''Runs a logistic regression. Takes the feature dataframe and outcome
    dataframe. Returns a model object.'''

    model = LogisticRegression()
    y = y.ravel()
    model = model.fit(xdf, y)
    return model


def classifiers_and_params():
    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),
        'ET': ExtraTreesClassifier(n_estimators=10, n_jobs=-1, criterion='entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
        'LR': LogisticRegression(penalty='l1', C=1e5),
        'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
        'GB': GradientBoostingClassifier(learning_rate=0.05, subsample=0.5, max_depth=6, n_estimators=10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss="hinge", penalty="l2"),
        'KNN': KNeighborsClassifier(n_neighbors=3)
            }

    grid = {
    # Reduced parameters for training time
    #'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'RF':{'n_estimators': [1,10,100], 'max_depth': [1,5,10,20], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'LR': { 'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
    'SGD': { 'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
    'ET': { 'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
    # Reduced parameters for training time
    #'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
    'GB': {'n_estimators': [1,10,100], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20]},
    'NB': {},
    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
    'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
    'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
           }
    return clfs, grid

def model_loop(X_train, X_test, y_train, y_test, models_to_run):
    clfs, grid = classifiers_and_params()
    for index, clf in enumerate([clfs[x] for x in models_to_run]):
        print(models_to_run[index] + '#' * 20)
        parameter_values = grid[models_to_run[index]]
        for p in ParameterGrid(parameter_values):
            #start = time.time()
            try:
                clf.set_params(**p)
                print(clf)
                y_pred_probs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
                y_pred = clf.predict(X_test)
                #threshold = np.sort(y_pred_probs)[::-1][int(.05*len(y_pred_probs))]
                #print threshold
                print(precision_at_k(y_test, y_pred_probs, .05))
                #plot_precision_recall_n(y_test,y_pred_probs,clf)
                #end = time.time()
                #elapsed = end - start
            except IndexError as e:
                print('Error:', e)
                continue

def gridsearch_model(X_train, X_test, y_train, y_test, models_to_run, filename):
    '''Uses grid search to try combinations of parameters from the parameter grid.
    Uses the best combination to fit the model with the training data. The predictions
    of the model are passed to create and graph a series of evaluation metrics.'''
    clfs, grid = classifiers_and_params()
    comparison_metrics = {}
    for index, clf in enumerate([clfs[x] for x in models_to_run]):
        start = time.time()
        try:
            param_values = grid[models_to_run[index]]
            model = GridSearchCV(clf, param_grid = param_values)
            model.fit(X_train, y_train)
            end = time.time()
            y_predictions = model.predict(X_test)
            metrics = evaluation_dict(y_test, y_predictions, (end - start))
            #Prints the parameters and eval metrics of the current model to screen/file
            print(model, '\n')
            print(json.dumps(metrics, indent = 2), '\n')
            comparison_metrics = build_comparison(comparison_metrics, metrics, models_to_run[index])
        except:
            print('Unexpected error raised for: ', str(clf))
    print(json.dumps(comparison_metrics, indent = 2), '\n')
    plot_metrics(comparison_metrics, filename)
