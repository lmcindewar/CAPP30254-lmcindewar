from read_explore_data import read, preview, gen_hist
from preprocess import mean_impute, impute_to_value, cat_from_cont, med_impute, log_feature
from models import logistic_reg, splitX_y, create_samples, model_loop
from sklearn.cross_validation import train_test_split
import numpy as np

def go():
    #################### Read and explore ####################

    df = read('./data/cs-training.csv')

    gen_hist(df)

    #################### Split, Preprocess, and Impute ####################

    dftrain, dftest = create_samples(df, 0.2)
    med = med_impute(dftrain, ['MonthlyIncome'])

    #Impute missing monthly income data from the test set to the median of the
    #training set and create log_income. 1 added to values to avoid log(0) errors.
    impute_to_value(dftest, 'MonthlyIncome', med)
    log_feature(dftrain, 'MonthlyIncome', offset_zero = 1)
    log_feature(dftest, 'MonthlyIncome', offset_zero = 1)
    #Impute missing numbers of dependents to zero.
    impute_to_value(dftrain, 'NumberOfDependents', 0)
    impute_to_value(dftest, 'NumberOfDependents', 0)

    #Bins and labels for debt ratio
    DebtRatioBins = [0, .2, .4, .6, .8, 1, 10, float("inf")]
    DebtRatioLabels = ['<.2', '.2-.4', '.4-.6', '.6-.8', '.8-1', '1-10', '10+']
    #Bins and labels for age
    AgeBins = [0, 20, 30, 40, 50, 60, 70, 80, 150]
    AgeLabels = ['<20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80+']

    #Create dummy variables for categories of debt ratio and age in training and testing set_value
    dftrain1 = cat_from_cont(dftrain, 'DebtRatio', DebtRatioBins, DebtRatioLabels)
    dftrain1 = cat_from_cont(dftrain1, 'age', AgeBins, AgeLabels)
    dftest1 = cat_from_cont(dftest, 'DebtRatio', DebtRatioBins, DebtRatioLabels)
    dftest1 = cat_from_cont(dftest1, 'age', AgeBins, AgeLabels)

    #Split training and test sets into x and y
    X_train, y_train = splitX_y(dftrain1, 'SeriousDlqin2yrs')
    X_test, y_test = splitX_y(dftest1, 'SeriousDlqin2yrs')


    #################### Generate and evaluate models #####################

    gridsearch_model(X_train, X_test, y_train, y_test, ['KNN', 'LR', 'DT', 'RF', 'SVM'])
