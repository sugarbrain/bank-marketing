import pandas
import numpy as np
from sklearn import tree
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import model_selection

# Tira limite de vizualição do dataframe quando printado
pandas.set_option('display.max_columns', None)
pandas.set_option('display.max_rows', None)

SEED = 42
np.random.seed(SEED)

# Full train set
train_file = "../datasets/train.csv"


def get_train_set(filepath, size=0.20):
    dataset = pandas.read_csv(train_file)

    test_size = 1.0 - size

    # use 20% of the train to search best params
    train, _ = train_test_split(dataset,
                                test_size=test_size,
                                random_state=SEED)

    return train


def get_estimators():
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


def fit_single_models(X, Y, estimators, kfold):
    results = list()

    for name, clf in estimators:
        result = model_selection.cross_val_score(clf, X, Y, cv=kfold)
        results.append((name, result.mean()))

    return results


def setup_kfold(X, Y, n_splits):
    kf = StratifiedKFold(n_splits=n_splits, random_state=SEED)
    kf.get_n_splits(X)

    return kf


def run_neural_network_committee_score(X, Y, estimators, voting_option, kfold):

    # create the ensemble model
    ensemble = VotingClassifier(estimators, voting=voting_option)
    results = model_selection.cross_val_score(ensemble, X, Y, cv=kfold)

    return results.mean()

K_SPLITS = 10

# split train set by 20%
train = get_train_set(train_file, 0.20)

# separate class from other columns
X = train.values[:, :-1]
Y = train['y']

# KFold
kfold = setup_kfold(X, Y, K_SPLITS)

# Get estimators
estimators = get_estimators()

single_model_results = fit_single_models(X, Y, estimators, kfold)

# Run scoring for best params
score_hard = run_neural_network_committee_score(X = X, Y = Y, estimators = estimators, 
                                                voting_option = "hard", kfold = kfold)

for name, mean in single_model_results:
    print("%s result mean: %0.4f" % (name, mean))

print("Neural network committee (hard) score: %0.4f" % score_hard)

# Run scoring for best params
score_soft = run_neural_network_committee_score(X = X, Y = Y, estimators = estimators, 
                                                voting_option = "soft", kfold = kfold)

print("Neural network committee (soft) score: %0.4f" % score_soft)