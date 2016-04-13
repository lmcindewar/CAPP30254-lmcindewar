import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split



def create_samples(df, fraction_for_testing):
    '''Given a dataframe and fraction for testing data (given as decimal), splits
    the data into random samples of training and testing data. Fraction for testing
    input as decimals, eg. 0.1 for 10 pct.
    Returns the test sample and training samples'''

    sample_n = dataframe.shape[0]
    testing_n = round(sample_n * fraction_for_testing)
    testing_sample = random.sample(range(sample_n), testing_n)
    testing_data = df.iloc[testing_sample]
    training_data = df.drop(dataframe.index[testing_sample], axis = 0)
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
