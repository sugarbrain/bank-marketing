import pandas
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn import neighbors

# Tira limite de vizualição do dataframe quando printado 
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)

seed = 42
np.random.seed(seed)

train_file = "../datasets/train.csv"
dataset = pandas.read_csv(train_file, sep=";", na_values="unknown")

### KFold

K_SPLITS = 10
X = dataset.values[:, 0:-1]
Y = dataset['y_no']

kf = StratifiedKFold(n_splits=K_SPLITS, random_state=seed)
kf.get_n_splits(X)

param = {
    "metric": "euclidean",
    "n_neighbors": 11
}

clf = neighbors.KNeighborsClassifier(
    metric=param["metric"],
    n_neighbors=param["n_neighbors"]
)

scores = cross_val_score(clf, X, Y, cv=kf)
print("Scores:", scores)

print("Best score: ", max(scores))
print("Mean: ", scores.mean())
print("Std dev: ", np.std(scores));



# total_acc = 0
# split_num = 1

# for train_index, test_index in kf.split(X):

#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = Y[train_index], Y[test_index]

#     param = {
#         "metric": "euclidean",
#         "n_neighbors": 11
#     }

#     clf = neighbors.KNeighborsClassifier(
#         metric=param["metric"],
#         n_neighbors=param["n_neighbors"]
#     )

#     clf = clf.fit(X_train, y_train)

#     acc = clf.score(X_train, y_train)
#     print(f"Accuracy on split {split_num}: %0.3f" % acc)
    
#     total_acc += clf.score(X_train, y_train)
#     split_num += 1

# print("\n=============================================")
# print(f"KFold's KNN mean: {total_acc/K_SPLITS}")
# print("=============================================")
