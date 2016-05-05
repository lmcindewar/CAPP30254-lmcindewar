import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import numpy as np
from scipy import stats

def read(filename, index_col = False):
    '''Read csv file to pandas dataframe'''
    df = pd.DataFrame.from_csv(filename)
    return df


def preview(df, filename = None):
    '''Returns the data types and column names, descriptive statistics, and
    historgram for dataframe. If filename given, saves output to file as html
    for markdown.'''
    types = (df.dtypes).to_frame()
    stats = df.describe().round(3)
    modes = df.mode()
    missing_vals = (len(df) - df.count()).to_frame()

    explore = [types, stats, modes, missing_vals]

    if filename:
        html_str = '<br>'.join([x.to_html() for x in explore])
        with open(filename, 'w') as f:
            f.write(html_str)
    return explore


def outliers(df, column, stdev = 2):
    '''Return a dataframe of the outliers, defined as data point more than three
    standard deviation away from the mean of the column.'''
    outliers = df[column][(np.abs(stats.zscore(df[column])) > stdev)]
    return outliers

# def clip_outliers(df, column, outliers = , value):
#     '''Change outlier values to prevent distortion of statistics and models. Takes
#     the dataframe, column, index of outliers, and value to replace. If value is not
#     given, fills with NaN.'''
#     if outliers:
#         cutoff = df.column.mean() + stdev * df.column.std()

def gen_hist(df):
    '''Generate first histograms of for each column in the dataframe'''
    cols = df.columns.values
    for col in cols:
        df.hist(col)
        plt.savefig(col)


def hist(df, title, num_bins = 8, code_percentile = .99):
    '''Takes data, title, number of bins (max 10), and percentile.
    Outputs a histogram.'''

    title = title.title()
    data_list = df.dropna().tolist()
    top_code_val = df.quantile(code_percentile)
    distinct_vals = len(set(data_list))
    num_bins = min(distinct_vals, num_bins)

    plt.style.use('ggplot')
    df.hist(bins = np.linspace(0, top_code_val, num_bins + 1), normed=True)
    plt.xlabel(title)
    plt.title('Histogram of ' + title.replace("_", " "))
    plt.tight_layout()
    plt.savefig('Histogram_' + title + '.png', format='png')
    plt.close()


def make_hist(df):
    '''Loop over columns to create histograms.'''
    titles = df.columns.values
    for title in titles:
        column_data = df[title]
        column_title = title
        hist(column_data, column_title)
