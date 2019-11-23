import pandas
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

file_name = "../datasets/bank_additional_full.csv"

dataset = pandas.read_csv(file_name, sep=";", na_values="unknown")

# place the class as the first column
columns = ['y'] + dataset.columns.tolist()[0:-1]
dataset = dataset[columns]

# replace missing values
dataset.fillna(method='ffill', inplace=True)

# standard scaler
ss = preprocessing.StandardScaler()
for column_name in dataset.columns[1:]: # excluding 'y'
    if dataset[column_name].dtype == object:
        dataset = pandas.get_dummies(dataset, columns=[column_name])
    else:
        dataset[column_name] = ss.fit_transform(dataset[[column_name]])

# binarize 'y' column
lb = preprocessing.LabelBinarizer()
dataset[['y']] = lb.fit_transform(dataset[['y']])

train, test = train_test_split(dataset, test_size=0.25, random_state=42)

train.to_csv('../datasets/train.csv', index=False)
test.to_csv('../datasets/test.csv', index=False)
