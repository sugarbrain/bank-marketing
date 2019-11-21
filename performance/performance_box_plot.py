import pandas
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

seed = 42

file_name = "../datasets/bank_additional_full.csv"

dataset = pandas.read_csv(file_name, sep=";", na_values="unknown")

dataset.fillna(method='ffill', inplace=True)

ss = preprocessing.StandardScaler()
for column_name in dataset.columns:
   if dataset[column_name].dtype == object:
       dataset = pandas.get_dummies(dataset, columns=[column_name])
   else:
       dataset[column_name] = ss.fit_transform(dataset[[column_name]])

X = dataset.values[:, 0:-2]
Y = dataset['y_no']

kfold = model_selection.StratifiedKFold(n_splits=10, random_state=seed)

# prepare models
models = []
models.append(('Decision Tree', tree.DecisionTreeClassifier(criterion="gini", max_depth=12, random_state=seed)))

models.append(('kNN', KNeighborsClassifier(metric="euclidean", n_neighbors=11)))

models.append(('Neural Networks', MLPClassifier(solver="adam", hidden_layer_sizes=(30,30), activation="tanh", random_state=seed)))

models.append(('Random Forest', RandomForestClassifier(n_estimators=12,max_features=6, random_state=seed)))

# create single models
mlp44 = MLPClassifier(solver='adam', hidden_layer_sizes=(30,30), activation='tanh', random_state=seed)
mlp48 = MLPClassifier(solver='adam', hidden_layer_sizes=(30,30), activation='relu', random_state=seed)
mlp42 = MLPClassifier(solver='adam', hidden_layer_sizes=(30,3), activation='tanh', random_state=seed)
mlp16 = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(30,3), activation='relu', random_state=seed)
mlp10 = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(30,3), activation='tanh', random_state=seed)

# create the sub models
MLP_estimators = [
('mlp44', mlp44),
('mlp48', mlp48),
('mlp42', mlp42),
('mlp16', mlp16),
('mlp10', mlp10),
]

models.append(('Neural Network Comittee', VotingClassifier(MLP_estimators, voting='hard')))

rf = RandomForestClassifier(n_estimators=16, max_features=None, random_state=seed)
knn = KNeighborsClassifier(metric="euclidean", n_neighbors=11)
mlpc = MLPClassifier(solver='adam', hidden_layer_sizes=(30, 30),
activation="tanh", random_state=seed)

heterogenous_estimators = [
('RandomForestClassifier', rf),
('KNeighborsClassifier', knn),
('MLPClassifier', mlpc)
]

models.append(('Heterogeneous Comittee', VotingClassifier(heterogenous_estimators, voting='hard')))

# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()