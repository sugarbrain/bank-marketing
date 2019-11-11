import pandas
import numpy as np
from sklearn import preprocessing
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

K_SPLITS = 10

# KNN Params
params = [
    {
        "name": "A",
        "metric": "manhattan",
        "n_neighbors": 11
    },
    # {
    #     "name": "B",
    #     "metric": "manhattan",
    #     "n_neighbors": 101
    # },
    # {
    #     "name": "C",
    #     "metric": "manhattan",
    #     "n_neighbors": 303
    # },
    # {
    #     "name": "D",
    #     "metric": "euclidean",
    #     "n_neighbors": 11
    # },
    # {
    #     "name": "E",
    #     "metric": "euclidean",
    #     "n_neighbors": 101
    # },
    # {
    #     "name": "F",
    #     "metric": "euclidean",
    #     "n_neighbors": 303
    # },
    # {
    #     "name": "G",
    #     "metric": "chebyshev",
    #     "n_neighbors": 11
    # },
    # {
    #     "name": "H",
    #     "metric": "chebyshev",
    #     "n_neighbors": 101
    # },
    # {
    #     "name": "I",
    #     "metric": "chebyshev",
    #     "n_neighbors": 303
    # },
    # {
    #     "name": "J",
    #     "metric": "minkowski",
    #     "n_neighbors": 11
    # },
    # {
    #     "name": "K",
    #     "metric": "minkowski",
    #     "n_neighbors": 101
    # },
    # {
    #     "name": "L",
    #     "metric": "minkowski",
    #     "n_neighbors": 303
    # },
]

np.random.seed(42) 

### KFold
X = dataset.values[:, 0:-1]
Y = dataset['y_no']

kf = KFold(n_splits=K_SPLITS)
kf.get_n_splits(X)

best_acc_mean = 0
best_param = {}


for p in params:

    print(f"\nUsing param -> {p}\n")

    total_acc = 0
    split_num = 1

    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        total_acc
        best_acc = 0

        clf = neighbors.KNeighborsClassifier(metric=p["metric"], n_neighbors=p["n_neighbors"])
        clf = clf.fit(X_train, y_train)
        acc = clf.score(X_train, y_train)

        print("Variation " + p["name"])
        print(f"Accuracy on split {split_num}: %0.3f" % acc)

        total_acc += acc
        split_num += 1
   
    acc_mean = total_acc / K_SPLITS
            
    print("\n=============================================")
    print(f"KFold's KNN mean accuracy: {acc_mean}")
    print("=============================================\n")
    
    if acc_mean > best_acc_mean:
        best_acc_mean = acc_mean
        best_param = p

print(f"Best accuracy mean: {best_acc_mean}")
print(f"Best param: {best_param}")
