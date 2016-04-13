import requests
import pandas as pd
import random


def mean_impute(df, columns):
    '''Fills in missing values with the mean of the column. Takes the dataframe
    and a list of the columns with missing values.'''
    for col in columns:
        mean = df.loc[:, col].mean()
        df.loc[:, col].fillna(value = mean, inplace = True)


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
        df[category] = pd.cut(df.loc[:,category], bin_list, labels = labels, \
                include_lowest = True)
        #Drops observations outside of the bins
        df = df.dropna(axis = 0, subset = [category])

def cat_from_cont(df, column, boundaries, labels, keep_col = False):
    df[column] = pd.cut(df[column], boundaries, labels = labels, include_lowest = True)
    dummy_var = pd.get_dummies(df[column], drop_first = True)
    df = pd.concat([df, dummy_var], axis = 1)
    if not keep_col:
        df = df.drop(column, axis = 1)
    return df

