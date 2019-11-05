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

### KFold
k = 10
X = dataset.values[:, 0:-1]
Y = dataset['y_no']

kf = KFold(n_splits=k)
kf.get_n_splits(X)

total_acc = 0

for train_index, test_index in kf.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    clf = neighbors.KNeighborsClassifier(metric="euclidean", n_neighbors=11)

    clf = clf.fit(X_train, y_train)

    acc = clf.score(X_train, y_train)
    print("Acuracia de trainamento clf: %0.3f" % acc)

    total_acc += acc

    print(f"KNN: Media de acuracia = {total_acc / k}")
