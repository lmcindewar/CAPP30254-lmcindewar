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


def drop_(df, columns_to_drop, drop_if_col, drop_val):
    '''Takes a dataframe and lists of column names to drop entire column.
    Another list of column names to conditionally drop observations (rows)
    if value is an specified.'''

    df.drop(df[columns_to_drop], axis = 1, inplace = True)
    for column in drop_if_zero:
        df = df[dataframe[column] != drop_val]


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
