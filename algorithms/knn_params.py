import pandas
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn import neighbors

file_name = "../bank_additional_full.csv"

dataset = pandas.read_csv(file_name, sep=";", na_values="unknown")

# Tira limite de vizualição do dataframe quando printado 
pandas.set_option('display.max_columns', None) 
pandas.set_option('display.max_rows', None)

categorical_columns = ['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome','y']

# Preenchendo com as ocorrências anteriores
dataset.fillna(method='ffill', inplace=True)

le = preprocessing.LabelEncoder()
ss = preprocessing.StandardScaler()
for column_name in dataset.columns:
    if dataset[column_name].dtype == object:
        dataset = pandas.get_dummies(dataset, columns=[column_name])
    else:
        dataset[column_name] = ss.fit_transform(dataset[[column_name]])

# KNN Params
params = [
    {
        "name": "A",
        "metric": "manhattan",
        "n_neighbors": 11
    },
    {
        "name": "B",
        "metric": "manhattan",
        "n_neighbors": 101
    },
    {
        "name": "C",
        "metric": "manhattan",
        "n_neighbors": 303
    },
    {
        "name": "D",
        "metric": "euclidean",
        "n_neighbors": 11
    },
    {
        "name": "E",
        "metric": "euclidean",
        "n_neighbors": 101
    },
    {
        "name": "F",
        "metric": "euclidean",
        "n_neighbors": 303
    },
    {
        "name": "G",
        "metric": "chebyshev",
        "n_neighbors": 11
    },
    {
        "name": "H",
        "metric": "chebyshev",
        "n_neighbors": 101
    },
    {
        "name": "I",
        "metric": "chebyshev",
        "n_neighbors": 303
    },
    {
        "name": "J",
        "metric": "minkowski",
        "n_neighbors": 11
    },
    {
        "name": "K",
        "metric": "minkowski",
        "n_neighbors": 101
    },
    {
        "name": "L",
        "metric": "minkowski",
        "n_neighbors": 303
    },
]

X = dataset.values[:, 0:-1]
Y = dataset['y_no']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 10)

print("========== KNN ==========")
results = {}

for p in params:
    clf = neighbors.KNeighborsClassifier(metric=p["metric"], n_neighbors=p["n_neighbors"])

    clf = clf.fit(X_train, y_train)

    print("Variacao " + p["name"])
    print("Acuracia de trainamento clf: %0.3f" %  clf.score(X_train, y_train))
    acc_test = clf.score(X_test, y_test)
    print("Acuracia de teste clf: %0.3f" %  acc_test)

    results[p["name"]] = acc_test

results_sorted = sorted(results, key=results.get)
print(results_sorted)
print("Melhores parametros: " + results_sorted[0])