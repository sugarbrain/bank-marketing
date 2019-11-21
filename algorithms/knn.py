import pandas
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import KFold
from sklearn import neighbors

np.random.seed(42)

file_name = "../datasets/bank_additional_full.csv"

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
K_SPLITS = 10
X = dataset.values[:, 0:-1]
Y = dataset['y_no']

kf = KFold(n_splits=K_SPLITS)
kf.get_n_splits(X)

total_acc = 0
split_num = 1

for train_index, test_index in kf.split(X):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]

    param = {
        "metric": "euclidean",
        "n_neighbors": 11
    }

    clf = neighbors.KNeighborsClassifier(
        metric=param["metric"],
        n_neighbors=param["n_neighbors"]
    )

    clf = clf.fit(X_train, y_train)

    acc = clf.score(X_train, y_train)
    print(f"Accuracy on split {split_num}: %0.3f" % acc)
    
    total_acc += clf.score(X_train, y_train)
    split_num += 1

print("\n=============================================")
print(f"KFold's KNN mean: {total_acc/K_SPLITS}")
print("=============================================")
