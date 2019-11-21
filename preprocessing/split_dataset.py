import pandas
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

file_name = "../datasets/bank_additional_full.csv"

dataset = pandas.read_csv(file_name, sep=";", na_values="unknown")

dataset.fillna(method='ffill', inplace=True)

ss = preprocessing.StandardScaler()
for column_name in dataset.columns:
    if dataset[column_name].dtype == object:
        dataset = pandas.get_dummies(dataset, columns=[column_name])
    else:
        dataset[column_name] = ss.fit_transform(dataset[[column_name]])

train, test = train_test_split(dataset, test_size=0.25, random_state=42)

train.to_csv('train.csv')
test.to_csv('test.csv')
