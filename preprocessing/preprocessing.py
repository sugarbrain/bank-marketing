# -*- coding: utf-8 -*-

import pandas
import matplotlib.pyplot as plt
from sklearn import preprocessing

file_name = "bank_additional_full.csv"

dataset = pandas.read_csv(file_name, sep=";", na_values="unknown")

# Tira limite de vizualição do dataframe quando printado 
pandas.set_option('display.max_columns', None) 
pandas.set_option('display.max_rows', None)

print("Apresentando o shape dos dados (dimensoes)")
print(dataset.shape)

print("Apresentando o tipo das colunas gerado pelo read_csv")
print(dataset.dtypes)

print("Visualizando o conjunto inicial (head) dos dados, ou mais claramente, "\
      "os 20 primeiros registros (head(20))")
print(dataset.head(20))

print("Conhecendo os dados estatisticos dos dados carregados (describe)")
print(dataset.describe())

print("Apresenta a contagem de valores NaN em cada coluna")
print(dataset.isnull().sum())

categorical_columns = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome','y']
for column in categorical_columns:
    dataset[column].value_counts().plot(kind='barh')
    plt.title(column)
    plt.show()

print("Criando graficos de caixa da distribuicao das classes")
dataset.plot(kind='box', subplots=True,figsize=(10,10),layout=(5,2), sharex=False, sharey=False)
plt.show()

# Preenchendo com as ocorrências anteriores
dataset.fillna(method='ffill', inplace=True)

print("Apresentando o shape dos dados (dimensoes) apos fillna")
print(dataset.shape)

print("Apresenta a contagem de valores NaN em cada coluna apos fillna")
print(dataset.isnull().sum())

le = preprocessing.LabelEncoder()
ss = preprocessing.StandardScaler()
for column_name in dataset.columns:
    if dataset[column_name].dtype == object:
        dataset = pandas.get_dummies(dataset, columns=[column_name])
    else:
        dataset[column_name] = ss.fit_transform(dataset[[column_name]])
        
print("Apresentando o shape dos dados (dimensoes) apos get_dummies e fit_transform")
print(dataset.shape)

print("Apresentando o tipo das colunas gerado pelo read_csv apos get_dummies e fit_transform")
print(dataset.dtypes)

print("Visualizando o conjunto inicial (head) dos dados, ou mais claramente, "\
      "os 20 primeiros registros (head(20))")
print(dataset.head(20))