from read_explore_data import read, preview, gen_hist
from preprocess import mean_impute, impute_to_value, cat_from_cont
from models import logistic_reg, splitX_y
from sklearn.cross_validation import train_test_split

df = read('../cs-training.csv')

pre = preview(df, 'preview.html')
gen_hist(df)


mean_impute(df, ['MonthlyIncome'])
impute_to_value(df, 'NumberOfDependents', 0)

dfX, dfy = splitX_y(df, 'SeriousDlqin2yrs')
model1 = logistic_reg(dfX, dfy)




DebtRatioBins = [0, .2, .4, .6, .8, 1, 10, float("inf")]
DebtRatioLabels = ['<.2', '.2-.4', '.4-.6', '.6-.8', '.8-1', '1-10', '10+']

AgeBins = [0, 20, 30, 40, 50, 60, 70, 80, 150]
AgeLabels = ['<20', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80+']

df1 = cat_from_cont(df, 'DebtRatio', DebtRatioBins, DebtRatioLabels)
df1 = cat_from_cont(df1, 'age', AgeBins, AgeLabels)

df1X, df1y = splitX_y(df1, 'SeriousDlqin2yrs')
df1X_train, df1X_test, df1y_train, df1y_test = train_test_split(df1X, df1y)

model2 = logistic_reg(df1X_train, df1y_train)

score = model2.score(df1X_test, df1y_test)