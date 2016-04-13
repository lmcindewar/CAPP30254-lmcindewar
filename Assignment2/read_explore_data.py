import sys
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

def read(filename, index_col = False):
    '''Read csv file to pandas dataframe'''
    df = pd.DataFrame.from_csv(filename)
    return df


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
    return explore

def gen_hist(df):
    '''Generate histograms of for each column in the dataframe'''
    cols = df.columns.values
    for col in cols:
        df.hist(col)
        plt.savefig(col)

# if __name__ == '__main__':
#     if len(sys.argv) != 2:
#         return 'Enter one filename'
#     else:
#         df = go(sys.argv[1])
#         preview(df)
