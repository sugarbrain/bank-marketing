import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Tira limite de vizualição do dataframe quando printado
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)

SEED = 42
np.random.seed(SEED)

# Full test set
test_file = "../datasets/test.csv"


def get_test_set(filepath, size=0.20):
    dataset = pandas.read_csv(test_file)

    test_size = 1.0 - size

    # use 20% of the train to search best params
    test, _ = train_test_split(dataset,
                                test_size=test_size,
                                random_state=SEED)

    return test

def setup_kfold(X, Y, n_splits):
    kf = StratifiedKFold(n_splits=n_splits, random_state=SEED)
    kf.get_n_splits(X)

    return kf

def get_neural_networks_committee_estimators():
        # Classifiers with best params
    sgd43_2 = MLPClassifier(solver = 'sgd', 
                            hidden_layer_sizes=(42,2)) # 0.908
    sgd6_1 = MLPClassifier(solver = 'sgd', 
                           hidden_layer_sizes=(6,1)) # 0.907    
    sgd18_2 = MLPClassifier(solver = 'sgd',
                            hidden_layer_sizes=(18,2)) # 0.907
    sgd36_1 = MLPClassifier(solver = 'sgd',
                            hidden_layer_sizes=(36,1)) # 0.907
    ada2_1 = MLPClassifier(solver = 'adam', 
                           hidden_layer_sizes=(2,1)) # 0.907

    estimators = [
        ('sgd43_2', sgd43_2),
        ('sgd6_1', sgd6_1),
        ('sgd18_2', sgd18_2),
        ('sgd36_1', sgd36_1),
        ('ada2_1', ada2_1),
    ]
    return estimators

def get_heterogen_committee_estimators():
    # Classifiers with best params
    dt = tree.DecisionTreeClassifier(criterion='gini',
                                     max_depth=5,
                                     random_state=SEED)
    rf = RandomForestClassifier(n_estimators=29,
                                max_features='sqrt',
                                random_state=SEED)
    knn = KNeighborsClassifier(metric="euclidean", n_neighbors=13)
    mlpc = MLPClassifier(solver='sgd',
                         hidden_layer_sizes=(43, 2),
                         random_state=SEED)

    estimators = [
        ('DecisionTree', dt),
        ('RandomForestClassifier', rf),
        ('KNeighborsClassifier', knn),
        ('MLPClassifier', mlpc),
    ]

    return estimators

def get_models():
    # Estimators with best params
    dt = tree.DecisionTreeClassifier(criterion='gini',
                                     max_depth=5,
                                     random_state=SEED)
    rf = RandomForestClassifier(n_estimators=29,
                                max_features='sqrt',
                                random_state=SEED)
    knn = KNeighborsClassifier(metric="euclidean", n_neighbors=13)
    mlpc = MLPClassifier(solver='sgd',
                         hidden_layer_sizes=(43, 2),
                         random_state=SEED)
    nnc = VotingClassifier(get_neural_networks_committee_estimators(), voting='hard')
    hc = VotingClassifier(get_heterogen_committee_estimators(), voting='soft')

    models = [
        ('Decision Tree', dt),
        ('Random Forest', rf),
        ('K Neighbors', knn),
        ('MLP', mlpc),
        ('Neural Network Committee', nnc),
        ('Heterogen Committee', hc)
    ]

    return models


def get_model_score(model, X, Y, kfold):
    return cross_val_score(model, X, Y, cv=kfold, scoring='accuracy')

def plot(names, scores):
    #options
    fig = plt.figure(figsize=(25, 8))
    ax = fig.add_subplot(111)
    ax.set_xticklabels(names)
    plt.boxplot(scores)
    plt.margins(x=0.005)
    plt.rc('font', size=18)
    plt.grid(linestyle='--')
    plt.show()


scores = []
names = []

K_SPLITS = 10

# split test set by 20%
test = get_test_set(test_file, 0.20)

# separate class from other columns
X = test.values[:, :-1]
Y = test['y']

# KFold
kfold = setup_kfold(X, Y, K_SPLITS)

models = get_models()

for name, model in models:
	model_score = get_model_score(model, X, Y, kfold)
	scores.append(model_score)
	names.append(name)

plot(names, scores)