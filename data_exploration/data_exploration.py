# TODO Explore the titanic data set

# Mean, Median and Mode for Pclass, age, sibsp, parch
# PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked

# Pclass is passenger class, either 1, 2 or 3
# SibSp is number of siblings and spouse
# Parch is number of parents and children

import pandas as pd
import numpy as np
from sklearn import tree
from sklearn import preprocessing


dataframe = pd.read_csv('../dataset/train.csv')

label_encoder = preprocessing.LabelEncoder()
dataframe['Sex'] = label_encoder.fit_transform(dataframe['Sex'])

dataframe = dataframe.drop('Cabin', axis=1)
dataframe = dataframe.drop('Embarked', axis=1)
dataframe = dataframe.drop('Ticket', axis=1)
dataframe = dataframe.drop('Name', axis=1)

for i in dataframe.columns:
    mean = np.mean(dataframe[i])
    dataframe[i] = dataframe[i].fillna(mean)

label = dataframe['Survived']
features = dataframe.drop('Survived', axis=1)

clf = tree.DecisionTreeClassifier(splitter='random', random_state=9, max_depth=4)
clf.fit(features, label)

test_df = pd.read_csv('../dataset/test.csv').drop('Cabin', axis=1)
test_df = test_df.drop('Embarked', axis=1)
test_df = test_df.drop('Ticket', axis=1)
test_df = test_df.drop('Name', axis=1)


label_encoder = preprocessing.LabelEncoder()
test_df['Sex'] = label_encoder.fit_transform(test_df['Sex'])

for i in test_df.columns:
    mean = np.mean(test_df[i])
    test_df[i] = test_df[i].fillna(mean)

d = clf.predict(test_df)

survived = pd.read_csv('../gender_submission.csv')['Survived']

count = 0

actuals = {
    "PassengerId": test_df['PassengerId'],
    "Survived":d
}

act = survived
print(act)
for i in range(len(d)):
    if d[i] == act[i]:
        count += 1
print(count)

predicted = pd.DataFrame(actuals)
predicted.to_csv('../predicted.csv', index=False)
