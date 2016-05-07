import requests
import pandas as pd
import random
import numpy as np


def med_impute(df, columns, return_med = True):
    '''Fills in missing values with the median of the column. Takes the dataframe
    and a list of the columns with missing values.'''
    for col in columns:
        med = df.loc[:, col].median()
        df.loc[:, col].fillna(value = med, inplace = True)
    return med

def mean_impute(df, columns, return_mean = True):
    '''Fills in missing values with the mean of the column. Takes the dataframe
    and a list of the columns with missing values.'''
    for col in columns:
        mean = df.loc[:, col].mean()
        df.loc[:, col].fillna(value = mean, inplace = True)
    return mean

def cond_mean_impute(df, columns, classifier):
    '''Fills in missing values with the conditional mean based on a classification
    column(s). Take the dataframe, list of columns with missing values, and the
    column of the classifier.'''
    for i, row in df.iterrows():
        for col in columns:
            if pd.isnull(row[col]):
                class_ = row[classifier]
                cond_mean = df.groupby(classifier).get_group(class_).mean()[col]
                df.set_value(index = i, col = col, value = cond_mean)

def impute_to_value(df, column, val):
    '''Fills missing data with speficied value.'''
    df[column].fillna(value = val, inplace = True)


def impute_gender(df, name_col, gender_col):
    '''Fills in missing gender using Genderize.io API.
    Takes the dataframe, column with first name, and column with gender'''
    for i, row in df.iterrows():
        if pd.isnull(df.ix[i , gender_col]):
            name = df.ix[i , name_col]
            result = requests.get('https://api.genderize.io/?name=' + name)
            gender = result.json()['gender']
            #Capitalize gender to match the rest of the table
            df.set_value(index = i, col = gender_col, value = gender.title())


def drop_data(df, columns_to_drop = None, drop_if_col = None, drop_val = None):
    '''Takes a dataframe and lists of column names to drop entire column.
    Another list of column names to conditionally drop observations (rows)
    if value is an specified.'''

    df.drop(df[columns_to_drop], axis = 1, inplace = True)
    for column in drop_if_zero:
        df = df[df[column] != drop_val]

def bin_data(df, bin_dict):
    '''Takes a dataframe as an argument and a dictionary with keys as column names
    and value as a list of tuples with bin ranges.
    eg. {col_name : [(label1, lb1, ub1), (label2, lb2, ub2)]}
    Categorizes numeric data into bins and drops observations with values outside
    of bin ranges.'''

    FIRST_BIN = 0
    LOWER_BOUND = 1
    for category, bins in bin_dict.items():
        labels = []
        #Initialize bin boundaries with lowest bound, then only add upper bounds
        bin_list = [bins[FIRST_BIN][LOWER_BOUND]]
        for label, lb, ub in bins:
            labels.append(label)
            bin_list.append(ub)
        #Include lowest to categorize any data equal to lowest bin boundary
        df.loc[:,category] = pd.cut(df.loc[:,category], bin_list, labels = labels, \
                include_lowest = True)
        #Drops observations outside of the bins
        df = df.dropna(axis = 0, subset = [category])

def cat_from_cont(df, column, boundaries, labels, drop_col = True):
    '''Creates categorical, dummy variables from a continuous variable. One dummy
    variable column per category is created. If keep_col is not passed as True,
    the original category is dropped.'''
    df.loc[:,column] = pd.cut(df[column], boundaries, labels = labels, include_lowest = True)
    dummy_var = pd.get_dummies(df[column])
    df = pd.concat([df, dummy_var], axis = 1)
    if drop_col:
        df.drop(column, axis = 1, inplace = True)
    return df

def log_feature(df, column, offset_zero = 0, drop_orig = True):
    '''Applies natural log to a column and adds it to the dataframe. Offsets values
    of 0 by amount given to avoid errors. Default is to drop original column from
    dataframe; eg. log wages will be added and wages removed.'''
    offset = df[column] + offset_zero
    newcolumn = 'log_' + str(column)
    df.loc[:,newcolumn] = offset.apply(np.log)
    if drop_orig:
        df.drop(column, axis = 1, inplace = True)
