# TODO Explore the titanic data set

# Mean, Median and Mode for Pclass, age, sibsp, parch
# PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

# Pclass is passenger class, either 1, 2 or 3
# SibSp is number of siblings and spouse
# Parch is number of parents and children

import pandas as pd
import numpy as np


def print_stats(array, name_array):
    mean = np.mean(array)
    median = np.median(array)
    var = np.var(array)
    std_deviation = np.std(array)

    print("**********************************************************************")
    print("{}".format(name_array))
    print("Mean for {} is {}".format(name_array, mean))
    print("Median for {} is {}".format(name_array, median))
    print("Var for {} is {}".format(name_array, var))
    print("Standard Deviation for {} is {}".format(name_array, std_deviation))
    print("**********************************************************************")

dataframe = pd.read_csv('../dataset/train.csv')

survival = np.array(dataframe['Survived'])
pclass = np.array(dataframe['Pclass'])
age = np.array(dataframe['Age'])
sibsp = np.array(dataframe['SibSp'])
parch = np.array(dataframe['Parch'])
fare = np.array(dataframe['Fare'])
embarked = np.array(dataframe['Embarked'])

print_stats(survival, 'Survival')
print_stats(pclass, 'Class')
print_stats(age, 'Age')
print_stats(sibsp, 'Siblings and spouse')
print_stats(parch, 'Parents and children')
print_stats(fare, 'Fare')
