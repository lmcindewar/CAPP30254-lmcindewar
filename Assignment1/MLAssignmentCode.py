import sys
import pandas as pd
import requests
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def go():
    student_data = pd.read_csv('../Downloads/mock_student_data_A2.csv')
    #
    numeric_stats = student_data.ix[:, ['Age', 'GPA', "Days_missed"]]

    descriptive_stats = numeric_stats.describe()

    modes = student_data.mode()

    hist = numeric_stats.hist()

    missing_vals = len(student_data) - student_data.count()

    with open("MLAssignment1.txt", 'w') as f:
        for output in [descriptive_stats, modes, missing_vals]:
            f.write(output.to_string())

    imputed_gender = student_data.loc[:,'Gender']
    for i, student in student_data.iterrows():
        if pd.isnull(student_data.ix[i , 'Gender']):
            name = student_data.ix[i , 'First_name']
            result = requests.get('https://api.genderize.io/?name=' + name)
            gender = result.json()['gender']
            #Capitalize gender to match the rest of the table
            imputed_gender.ix[i] = gender.title()
    student_data.loc[: , 'Gender'] = imputed_gender
    student_data.to_csv('mock_student_data_A2.csv')

    student_dataA = student_data.copy()
    for i in ['Age', 'GPA', 'Days_missed']:
        mean = student_dataA.loc[:, i].mean()
        student_dataA.loc[:, i].fillna(value = mean, inplace = True)
    student_dataA.to_csv('mock_student_dataA3A.csv')

    student_dataB = student_data.copy()
    for row, student in student_dataB.iterrows():
    for i in ['Age', 'GPA', 'Days_missed']:
        if pd.isnull(student[i]):
            class_ = student['Graduated']
            cond_mean = student_dataB.groupby('Graduated').get_group(class_).mean()[i]
            student_dataB.set_value(index = row, col = str(i), value = cond_mean)
    student_dataB.to_csv('mock_student_dataA3B.csv')

if __name__ == '__main__':
    go()
