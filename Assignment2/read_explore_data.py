import sys
import pandas as pd
import matplotlib.pyplot as plt

def read(filename):
    '''Read csv file to pandas dataframe'''
    dataframe = pd.read_csv(filename)
    return dataframe


def preview(df, filename = None):
    '''Returns the data types and columsn names, descriptive statistics, and
    historgram for dataframe. If filename given, saves output to file as html
    for markdown.'''
    types = (df.dtypes).to_frame()
    stats = df.describe()
    modes = df.mode()
    missing_vals = (len(df) - df.count()).to_frame()

    explore = [types, stats, modes, missing_vals]

    if filename:
        html_str = '<br>'.join([x.to_html() for x in explore])
        with open(filename, 'w') as f:
            f.write(html_str)
    else:
        return explore


# if __name__ == '__main__':
#     if len(sys.argv) != 2:
#         return 'Enter one filename'
#     else:
#         df = go(sys.argv[1])
#         preview(df)
