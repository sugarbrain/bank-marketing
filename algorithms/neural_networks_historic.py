# -*- coding: utf-8 -*-

import pandas
import numpy as np
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.model_selection import StratifiedKFold, cross_val_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

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


# Neural Networks Params
def generate_nn_params():
    solvers = ["lbfgs", "sgd", "adam"]
    # hidden_layer_sizes = (neurons_per_layer, number_of_layers)
    neurons_per_layer = list(range(1, 51))
    number_of_layers = list(range(1, 3))

    params = []

    for solver in solvers:
        for num_layers in number_of_layers:
            for num_neurons in neurons_per_layer:
                params.append({
                    "id": f"{solver[0:3].upper()}_{num_neurons}_{num_layers}",
                    "solver": solver,
                    "hidden_layer_sizes": (num_neurons, num_layers)
                })

    return params


def setup_kfold(X, Y, n_splits):
    kf = StratifiedKFold(n_splits=n_splits, random_state=SEED)
    kf.get_n_splits(X)

    return kf


def run_nn_score(X, Y, params, kfold):
    print("Busca de Parametros Neural Networks")

    all_scores = []

    for param in params:
        clf = MLPClassifier(solver=param["solver"],
                            hidden_layer_sizes=param["hidden_layer_sizes"],
                            random_state=SEED)

        scores = cross_val_score(clf, X, Y, cv=kfold)

        mean = scores.mean()

        all_scores.append({
            "id": param["id"],
            "solver": param["solver"],
            "hidden_layer_sizes": param["hidden_layer_sizes"],
            "result": mean
        })

        print("%s | %0.4f" % (param["id"], mean))

    best = max(all_scores, key=lambda s: s["result"])
    print(f"Best param: {best}")
    print(all_scores)
    return all_scores


def plot(scores):
    # options
    plt.figure(figsize=(70, 8))
    plt.margins(x=0.005)
    plt.rc('font', size=14)
    plt.xticks(rotation=90)
    plt.grid(linestyle='--')
    
    x = list(map(lambda x: x["id"], scores))  # names
    y = list(map(lambda x: x["result"], scores))  # scores

    plt.plot(x, y, 'o--')
    plt.suptitle('Busca de Parametros Neural Networks')
    plt.show()


def print_markdown_table(scores):
    print("Variação | *solver* | *hidden_layer_sizes* | Acurácia média")
    print("------ | ------- | -------- | ----------")

    for s in scores:
        name = s["id"]
        solver = s["solver"]
        sizes = s["hidden_layer_sizes"]
        result = '{:0.4f}'.format(s["result"])

        print(f"{name} | {solver} | {sizes} | {result}")


K_SPLITS = 10

# split train set by 20%
train = get_train_set(train_file, 0.20)

# separate class from other columns
X = train.values[:, :-1]
Y = train['y']

# KFold
kfold = setup_kfold(X, Y, K_SPLITS)

# Generate params
params = generate_nn_params()

# Run scoring for best params
#scores = run_nn_score(X, Y, params, kfold)
scores = [{'id': 'LBF_1_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (1, 1), 'result': 0.8837815297949678}, {'id': 'LBF_2_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (2, 1), 'result': 0.9028840353941454}, {'id': 'LBF_3_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (3, 1), 'result': 0.8920389398540804}, {'id': 'LBF_4_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (4, 1), 'result': 0.892684090992536}, {'id': 'LBF_5_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (5, 1), 'result': 0.8928456410337106}, {'id': 'LBF_6_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (6, 1), 'result': 0.9035307600719632}, {'id': 'LBF_7_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (7, 1), 'result': 0.8894457469853609}, {'id': 'LBF_8_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (8, 1), 'result': 0.8837815297949678}, {'id': 'LBF_9_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (9, 1), 'result': 0.898837416667978}, {'id': 'LBF_10_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (10, 1), 'result': 0.894461665958574}, {'id': 'LBF_11_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (11, 1), 'result': 0.8900950942287821}, {'id': 'LBF_12_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (12, 1), 'result': 0.90385254887151}, {'id': 'LBF_13_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (13, 1), 'result': 0.8879917966147923}, {'id': 'LBF_14_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (14, 1), 'result': 0.8837825788212091}, {'id': 'LBF_15_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (15, 1), 'result': 0.8837815297949678}, {'id': 'LBF_16_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (16, 1), 'result': 0.8837815297949678}, {'id': 'LBF_17_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (17, 1), 'result': 0.8847524035813755}, {'id': 'LBF_18_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (18, 1), 'result': 0.8824854578737288}, {'id': 'LBF_19_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (19, 1), 'result': 0.9025611975683571}, {'id': 'LBF_20_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (20, 1), 'result': 0.8797343865556797}, {'id': 'LBF_21_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (21, 1), 'result': 0.8837815297949678}, {'id': 'LBF_22_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (22, 1), 'result': 0.8863726246112046}, {'id': 'LBF_23_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (23, 1), 'result': 0.8844303525252684}, {'id': 'LBF_24_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (24, 1), 'result': 0.8839457024017456}, {'id': 'LBF_25_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (25, 1), 'result': 0.8875032126428642}, {'id': 'LBF_26_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (26, 1), 'result': 0.8829753531284584}, {'id': 'LBF_27_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (27, 1), 'result': 0.8873437606541728}, {'id': 'LBF_28_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (28, 1), 'result': 0.8852407252967431}, {'id': 'LBF_29_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (29, 1), 'result': 0.8858877122311215}, {'id': 'LBF_30_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (30, 1), 'result': 0.8837815297949678}, {'id': 'LBF_31_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (31, 1), 'result': 0.8837815297949678}, {'id': 'LBF_32_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (32, 1), 'result': 0.8823239078325544}, {'id': 'LBF_33_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (33, 1), 'result': 0.883943866605823}, {'id': 'LBF_34_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (34, 1), 'result': 0.8842693269972148}, {'id': 'LBF_35_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (35, 1), 'result': 0.8881541334256475}, {'id': 'LBF_36_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (36, 1), 'result': 0.8860461151935717}, {'id': 'LBF_37_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (37, 1), 'result': 0.8837815297949678}, {'id': 'LBF_38_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (38, 1), 'result': 0.8837815297949678}, {'id': 'LBF_39_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (39, 1), 'result': 0.8866954624369928}, {'id': 'LBF_40_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (40, 1), 'result': 0.879897510136216}, {'id': 'LBF_41_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (41, 1), 'result': 0.8868562257084862}, {'id': 'LBF_42_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (42, 1), 'result': 0.8837815297949678}, {'id': 'LBF_43_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (43, 1), 'result': 0.8837815297949678}, {'id': 'LBF_44_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (44, 1), 'result': 0.8883135854143391}, {'id': 'LBF_45_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (45, 1), 'result': 0.8837815297949678}, {'id': 'LBF_46_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (46, 1), 'result': 0.8899343309572888}, {'id': 'LBF_47_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (47, 1), 'result': 0.8871832596392398}, {'id': 'LBF_48_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (48, 1), 'result': 0.8884772335079963}, {'id': 'LBF_49_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (49, 1), 'result': 0.8837815297949678}, {'id': 'LBF_50_1', 'solver': 'lbfgs', 'hidden_layer_sizes': (50, 1), 'result': 0.8837815297949678}, {'id': 'LBF_1_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (1, 2), 'result': 0.8837815297949678}, {'id': 'LBF_2_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (2, 2), 'result': 0.9002934650910293}, {'id': 'LBF_3_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (3, 2), 'result': 0.8837815297949678}, {'id': 'LBF_4_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (4, 2), 'result': 0.8936586363707889}, {'id': 'LBF_5_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (5, 2), 'result': 0.8925207051554395}, {'id': 'LBF_6_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (6, 2), 'result': 0.8978639203159666}, {'id': 'LBF_7_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (7, 2), 'result': 0.9001321773064154}, {'id': 'LBF_8_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (8, 2), 'result': 0.8975402957204975}, {'id': 'LBF_9_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (9, 2), 'result': 0.8931721504513435}, {'id': 'LBF_10_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (10, 2), 'result': 0.8936583741142284}, {'id': 'LBF_11_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (11, 2), 'result': 0.8892826234048246}, {'id': 'LBF_12_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (12, 2), 'result': 0.893005879792083}, {'id': 'LBF_13_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (13, 2), 'result': 0.8850789129990086}, {'id': 'LBF_14_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (14, 2), 'result': 0.886048737759175}, {'id': 'LBF_15_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (15, 2), 'result': 0.8858861386917594}, {'id': 'LBF_16_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (16, 2), 'result': 0.8813548698420691}, {'id': 'LBF_17_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (17, 2), 'result': 0.8960863453499289}, {'id': 'LBF_18_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (18, 2), 'result': 0.8826512040198686}, {'id': 'LBF_19_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (19, 2), 'result': 0.8907441792156432}, {'id': 'LBF_20_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (20, 2), 'result': 0.8834581674560589}, {'id': 'LBF_21_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (21, 2), 'result': 0.884915527161912}, {'id': 'LBF_22_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (22, 2), 'result': 0.8808723177710289}, {'id': 'LBF_23_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (23, 2), 'result': 0.886698609515717}, {'id': 'LBF_24_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (24, 2), 'result': 0.8803803244638164}, {'id': 'LBF_25_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (25, 2), 'result': 0.8810286226809962}, {'id': 'LBF_26_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (26, 2), 'result': 0.8852394140139417}, {'id': 'LBF_27_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (27, 2), 'result': 0.8847521413248153}, {'id': 'LBF_28_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (28, 2), 'result': 0.8862118613397115}, {'id': 'LBF_29_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (29, 2), 'result': 0.8933337004925178}, {'id': 'LBF_30_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (30, 2), 'result': 0.8826498927370668}, {'id': 'LBF_31_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (31, 2), 'result': 0.8834584297126193}, {'id': 'LBF_32_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (32, 2), 'result': 0.885079175255569}, {'id': 'LBF_33_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (33, 2), 'result': 0.8881536089125269}, {'id': 'LBF_34_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (34, 2), 'result': 0.8891239581858141}, {'id': 'LBF_35_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (35, 2), 'result': 0.8873448096804142}, {'id': 'LBF_36_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (36, 2), 'result': 0.8807081451642512}, {'id': 'LBF_37_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (37, 2), 'result': 0.8837820543080885}, {'id': 'LBF_38_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (38, 2), 'result': 0.8866967737197946}, {'id': 'LBF_39_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (39, 2), 'result': 0.8879881250229473}, {'id': 'LBF_40_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (40, 2), 'result': 0.8847534526076171}, {'id': 'LBF_41_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (41, 2), 'result': 0.8826491059673858}, {'id': 'LBF_42_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (42, 2), 'result': 0.8828127540610428}, {'id': 'LBF_43_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (43, 2), 'result': 0.8865352236786203}, {'id': 'LBF_44_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (44, 2), 'result': 0.8844332373474322}, {'id': 'LBF_45_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (45, 2), 'result': 0.889124744955495}, {'id': 'LBF_46_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (46, 2), 'result': 0.8892847214573072}, {'id': 'LBF_47_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (47, 2), 'result': 0.883460527765102}, {'id': 'LBF_48_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (48, 2), 'result': 0.8815187801922866}, {'id': 'LBF_49_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (49, 2), 'result': 0.8857264244465075}, {'id': 'LBF_50_2', 'solver': 'lbfgs', 'hidden_layer_sizes': (50, 2), 'result': 0.8836205042669143}, {'id': 'SGD_1_1', 'solver': 'sgd', 'hidden_layer_sizes': (1, 1), 'result': 0.8837815297949678}, {'id': 'SGD_2_1', 'solver': 'sgd', 'hidden_layer_sizes': (2, 1), 'result': 0.9051491453058699}, {'id': 'SGD_3_1', 'solver': 'sgd', 'hidden_layer_sizes': (3, 1), 'result': 0.8912293538522865}, {'id': 'SGD_4_1', 'solver': 'sgd', 'hidden_layer_sizes': (4, 1), 'result': 0.8896067725134144}, {'id': 'SGD_5_1', 'solver': 'sgd', 'hidden_layer_sizes': (5, 1), 'result': 0.9064420701483847}, {'id': 'SGD_6_1', 'solver': 'sgd', 'hidden_layer_sizes': (6, 1), 'result': 0.9077373552999429}, {'id': 'SGD_7_1', 'solver': 'sgd', 'hidden_layer_sizes': (7, 1), 'result': 0.9066044069592399}, {'id': 'SGD_8_1', 'solver': 'sgd', 'hidden_layer_sizes': (8, 1), 'result': 0.9061197568357173}, {'id': 'SGD_9_1', 'solver': 'sgd', 'hidden_layer_sizes': (9, 1), 'result': 0.9038535978977513}, {'id': 'SGD_10_1', 'solver': 'sgd', 'hidden_layer_sizes': (10, 1), 'result': 0.9041753866972982}, {'id': 'SGD_11_1', 'solver': 'sgd', 'hidden_layer_sizes': (11, 1), 'result': 0.9012653879036783}, {'id': 'SGD_12_1', 'solver': 'sgd', 'hidden_layer_sizes': (12, 1), 'result': 0.8837815297949678}, {'id': 'SGD_13_1', 'solver': 'sgd', 'hidden_layer_sizes': (13, 1), 'result': 0.9040172459914085}, {'id': 'SGD_14_1', 'solver': 'sgd', 'hidden_layer_sizes': (14, 1), 'result': 0.9041780092629018}, {'id': 'SGD_15_1', 'solver': 'sgd', 'hidden_layer_sizes': (15, 1), 'result': 0.8837815297949678}, {'id': 'SGD_16_1', 'solver': 'sgd', 'hidden_layer_sizes': (16, 1), 'result': 0.8837815297949678}, {'id': 'SGD_17_1', 'solver': 'sgd', 'hidden_layer_sizes': (17, 1), 'result': 0.9070893193393234}, {'id': 'SGD_18_1', 'solver': 'sgd', 'hidden_layer_sizes': (18, 1), 'result': 0.9043411328434381}, {'id': 'SGD_19_1', 'solver': 'sgd', 'hidden_layer_sizes': (19, 1), 'result': 0.9035299733022821}, {'id': 'SGD_20_1', 'solver': 'sgd', 'hidden_layer_sizes': (20, 1), 'result': 0.9067667437700953}, {'id': 'SGD_21_1', 'solver': 'sgd', 'hidden_layer_sizes': (21, 1), 'result': 0.9067667437700953}, {'id': 'SGD_22_1', 'solver': 'sgd', 'hidden_layer_sizes': (22, 1), 'result': 0.9061192323225965}, {'id': 'SGD_23_1', 'solver': 'sgd', 'hidden_layer_sizes': (23, 1), 'result': 0.9045008470886898}, {'id': 'SGD_24_1', 'solver': 'sgd', 'hidden_layer_sizes': (24, 1), 'result': 0.898674555344002}, {'id': 'SGD_25_1', 'solver': 'sgd', 'hidden_layer_sizes': (25, 1), 'result': 0.903692572369698}, {'id': 'SGD_26_1', 'solver': 'sgd', 'hidden_layer_sizes': (26, 1), 'result': 0.8837815297949678}, {'id': 'SGD_27_1', 'solver': 'sgd', 'hidden_layer_sizes': (27, 1), 'result': 0.9062805201072106}, {'id': 'SGD_28_1', 'solver': 'sgd', 'hidden_layer_sizes': (28, 1), 'result': 0.8837815297949678}, {'id': 'SGD_29_1', 'solver': 'sgd', 'hidden_layer_sizes': (29, 1), 'result': 0.9056335331728324}, {'id': 'SGD_30_1', 'solver': 'sgd', 'hidden_layer_sizes': (30, 1), 'result': 0.8837815297949678}, {'id': 'SGD_31_1', 'solver': 'sgd', 'hidden_layer_sizes': (31, 1), 'result': 0.9045000603190088}, {'id': 'SGD_32_1', 'solver': 'sgd', 'hidden_layer_sizes': (32, 1), 'result': 0.9036920478565771}, {'id': 'SGD_33_1', 'solver': 'sgd', 'hidden_layer_sizes': (33, 1), 'result': 0.9001300792539325}, {'id': 'SGD_34_1', 'solver': 'sgd', 'hidden_layer_sizes': (34, 1), 'result': 0.9057963944968084}, {'id': 'SGD_35_1', 'solver': 'sgd', 'hidden_layer_sizes': (35, 1), 'result': 0.9054738189275806}, {'id': 'SGD_36_1', 'solver': 'sgd', 'hidden_layer_sizes': (36, 1), 'result': 0.9074142552175942}, {'id': 'SGD_37_1', 'solver': 'sgd', 'hidden_layer_sizes': (37, 1), 'result': 0.9056340576859532}, {'id': 'SGD_38_1', 'solver': 'sgd', 'hidden_layer_sizes': (38, 1), 'result': 0.9036915233434565}, {'id': 'SGD_39_1', 'solver': 'sgd', 'hidden_layer_sizes': (39, 1), 'result': 0.8837815297949678}, {'id': 'SGD_40_1', 'solver': 'sgd', 'hidden_layer_sizes': (40, 1), 'result': 0.8837815297949678}, {'id': 'SGD_41_1', 'solver': 'sgd', 'hidden_layer_sizes': (41, 1), 'result': 0.8837815297949678}, {'id': 'SGD_42_1', 'solver': 'sgd', 'hidden_layer_sizes': (42, 1), 'result': 0.8913867077884954}, {'id': 'SGD_43_1', 'solver': 'sgd', 'hidden_layer_sizes': (43, 1), 'result': 0.9062813068768915}, {'id': 'SGD_44_1', 'solver': 'sgd', 'hidden_layer_sizes': (44, 1), 'result': 0.8837815297949678}, {'id': 'SGD_45_1', 'solver': 'sgd', 'hidden_layer_sizes': (45, 1), 'result': 0.9062810446203311}, {'id': 'SGD_46_1', 'solver': 'sgd', 'hidden_layer_sizes': (46, 1), 'result': 0.9040169837348483}, {'id': 'SGD_47_1', 'solver': 'sgd', 'hidden_layer_sizes': (47, 1), 'result': 0.905794558700886}, {'id': 'SGD_48_1', 'solver': 'sgd', 'hidden_layer_sizes': (48, 1), 'result': 0.9017500380272011}, {'id': 'SGD_49_1', 'solver': 'sgd', 'hidden_layer_sizes': (49, 1), 'result': 0.9056337954293927}, {'id': 'SGD_50_1', 'solver': 'sgd', 'hidden_layer_sizes': (50, 1), 'result': 0.8837815297949678}, {'id': 'SGD_1_2', 'solver': 'sgd', 'hidden_layer_sizes': (1, 2), 'result': 0.8837815297949678}, {'id': 'SGD_2_2', 'solver': 'sgd', 'hidden_layer_sizes': (2, 2), 'result': 0.9046637084126659}, {'id': 'SGD_3_2', 'solver': 'sgd', 'hidden_layer_sizes': (3, 2), 'result': 0.9054735566710201}, {'id': 'SGD_4_2', 'solver': 'sgd', 'hidden_layer_sizes': (4, 2), 'result': 0.9056337954293927}, {'id': 'SGD_5_2', 'solver': 'sgd', 'hidden_layer_sizes': (5, 2), 'result': 0.8837815297949678}, {'id': 'SGD_6_2', 'solver': 'sgd', 'hidden_layer_sizes': (6, 2), 'result': 0.899969053725879}, {'id': 'SGD_7_2', 'solver': 'sgd', 'hidden_layer_sizes': (7, 2), 'result': 0.9025590995158744}, {'id': 'SGD_8_2', 'solver': 'sgd', 'hidden_layer_sizes': (8, 2), 'result': 0.9009401897688469}, {'id': 'SGD_9_2', 'solver': 'sgd', 'hidden_layer_sizes': (9, 2), 'result': 0.9048260452235214}, {'id': 'SGD_10_2', 'solver': 'sgd', 'hidden_layer_sizes': (10, 2), 'result': 0.8837815297949678}, {'id': 'SGD_11_2', 'solver': 'sgd', 'hidden_layer_sizes': (11, 2), 'result': 0.8837815297949678}, {'id': 'SGD_12_2', 'solver': 'sgd', 'hidden_layer_sizes': (12, 2), 'result': 0.8983490949526104}, {'id': 'SGD_13_2', 'solver': 'sgd', 'hidden_layer_sizes': (13, 2), 'result': 0.8991615657765679}, {'id': 'SGD_14_2', 'solver': 'sgd', 'hidden_layer_sizes': (14, 2), 'result': 0.9054746056972615}, {'id': 'SGD_15_2', 'solver': 'sgd', 'hidden_layer_sizes': (15, 2), 'result': 0.9067680550528969}, {'id': 'SGD_16_2', 'solver': 'sgd', 'hidden_layer_sizes': (16, 2), 'result': 0.9007799510104746}, {'id': 'SGD_17_2', 'solver': 'sgd', 'hidden_layer_sizes': (17, 2), 'result': 0.9041761734669793}, {'id': 'SGD_18_2', 'solver': 'sgd', 'hidden_layer_sizes': (18, 2), 'result': 0.9075758052587686}, {'id': 'SGD_19_2', 'solver': 'sgd', 'hidden_layer_sizes': (19, 2), 'result': 0.9033668497217459}, {'id': 'SGD_20_2', 'solver': 'sgd', 'hidden_layer_sizes': (20, 2), 'result': 0.9032076599896147}, {'id': 'SGD_21_2', 'solver': 'sgd', 'hidden_layer_sizes': (21, 2), 'result': 0.8837815297949678}, {'id': 'SGD_22_2', 'solver': 'sgd', 'hidden_layer_sizes': (22, 2), 'result': 0.9061210681185191}, {'id': 'SGD_23_2', 'solver': 'sgd', 'hidden_layer_sizes': (23, 2), 'result': 0.8842669666881717}, {'id': 'SGD_24_2', 'solver': 'sgd', 'hidden_layer_sizes': (24, 2), 'result': 0.9051475717665077}, {'id': 'SGD_25_2', 'solver': 'sgd', 'hidden_layer_sizes': (25, 2), 'result': 0.9020739248792309}, {'id': 'SGD_26_2', 'solver': 'sgd', 'hidden_layer_sizes': (26, 2), 'result': 0.9033697345439096}, {'id': 'SGD_27_2', 'solver': 'sgd', 'hidden_layer_sizes': (27, 2), 'result': 0.9049870707515749}, {'id': 'SGD_28_2', 'solver': 'sgd', 'hidden_layer_sizes': (28, 2), 'result': 0.9048244716841591}, {'id': 'SGD_29_2', 'solver': 'sgd', 'hidden_layer_sizes': (29, 2), 'result': 0.9041774847497811}, {'id': 'SGD_30_2', 'solver': 'sgd', 'hidden_layer_sizes': (30, 2), 'result': 0.9038546469239929}, {'id': 'SGD_31_2', 'solver': 'sgd', 'hidden_layer_sizes': (31, 2), 'result': 0.9033692100307888}, {'id': 'SGD_32_2', 'solver': 'sgd', 'hidden_layer_sizes': (32, 2), 'result': 0.9025598862855553}, {'id': 'SGD_33_2', 'solver': 'sgd', 'hidden_layer_sizes': (33, 2), 'result': 0.8837815297949678}, {'id': 'SGD_34_2', 'solver': 'sgd', 'hidden_layer_sizes': (34, 2), 'result': 0.9066065050117228}, {'id': 'SGD_35_2', 'solver': 'sgd', 'hidden_layer_sizes': (35, 2), 'result': 0.9048247339407194}, {'id': 'SGD_36_2', 'solver': 'sgd', 'hidden_layer_sizes': (36, 2), 'result': 0.9046637084126659}, {'id': 'SGD_37_2', 'solver': 'sgd', 'hidden_layer_sizes': (37, 2), 'result': 0.882001594519887}, {'id': 'SGD_38_2', 'solver': 'sgd', 'hidden_layer_sizes': (38, 2), 'result': 0.9069298673506317}, {'id': 'SGD_39_2', 'solver': 'sgd', 'hidden_layer_sizes': (39, 2), 'result': 0.9032084467592958}, {'id': 'SGD_40_2', 'solver': 'sgd', 'hidden_layer_sizes': (40, 2), 'result': 0.904339559304076}, {'id': 'SGD_41_2', 'solver': 'sgd', 'hidden_layer_sizes': (41, 2), 'result': 0.9066038824461196}, {'id': 'SGD_42_2', 'solver': 'sgd', 'hidden_layer_sizes': (42, 2), 'result': 0.9041780092629018}, {'id': 'SGD_43_2', 'solver': 'sgd', 'hidden_layer_sizes': (43, 2), 'result': 0.9083867025433641}, {'id': 'SGD_44_2', 'solver': 'sgd', 'hidden_layer_sizes': (44, 2), 'result': 0.9053099085773629}, {'id': 'SGD_45_2', 'solver': 'sgd', 'hidden_layer_sizes': (45, 2), 'result': 0.9066078162945246}, {'id': 'SGD_46_2', 'solver': 'sgd', 'hidden_layer_sizes': (46, 2), 'result': 0.902720125043928}, {'id': 'SGD_47_2', 'solver': 'sgd', 'hidden_layer_sizes': (47, 2), 'result': 0.9072521806632994}, {'id': 'SGD_48_2', 'solver': 'sgd', 'hidden_layer_sizes': (48, 2), 'result': 0.9030453231787592}, {'id': 'SGD_49_2', 'solver': 'sgd', 'hidden_layer_sizes': (49, 2), 'result': 0.9062818313900124}, {'id': 'SGD_50_2', 'solver': 'sgd', 'hidden_layer_sizes': (50, 2), 'result': 0.9046639706692264}, {'id': 'ADA_1_1', 'solver': 'adam', 'hidden_layer_sizes': (1, 1), 'result': 0.8837815297949678}, {'id': 'ADA_2_1', 'solver': 'adam', 'hidden_layer_sizes': (2, 1), 'result': 0.9074166155266374}, {'id': 'ADA_3_1', 'solver': 'adam', 'hidden_layer_sizes': (3, 1), 'result': 0.9049886442909371}, {'id': 'ADA_4_1', 'solver': 'adam', 'hidden_layer_sizes': (4, 1), 'result': 0.9009388784860454}, {'id': 'ADA_5_1', 'solver': 'adam', 'hidden_layer_sizes': (5, 1), 'result': 0.8994838790892354}, {'id': 'ADA_6_1', 'solver': 'adam', 'hidden_layer_sizes': (6, 1), 'result': 0.9027237966357727}, {'id': 'ADA_7_1', 'solver': 'adam', 'hidden_layer_sizes': (7, 1), 'result': 0.9014298227670168}, {'id': 'ADA_8_1', 'solver': 'adam', 'hidden_layer_sizes': (8, 1), 'result': 0.900132701819536}, {'id': 'ADA_9_1', 'solver': 'adam', 'hidden_layer_sizes': (9, 1), 'result': 0.9007770661883108}, {'id': 'ADA_10_1', 'solver': 'adam', 'hidden_layer_sizes': (10, 1), 'result': 0.8965686351644087}, {'id': 'ADA_11_1', 'solver': 'adam', 'hidden_layer_sizes': (11, 1), 'result': 0.8999701027521203}, {'id': 'ADA_12_1', 'solver': 'adam', 'hidden_layer_sizes': (12, 1), 'result': 0.8853996527723142}, {'id': 'ADA_13_1', 'solver': 'adam', 'hidden_layer_sizes': (13, 1), 'result': 0.9027214363267296}, {'id': 'ADA_14_1', 'solver': 'adam', 'hidden_layer_sizes': (14, 1), 'result': 0.9015895370122685}, {'id': 'ADA_15_1', 'solver': 'adam', 'hidden_layer_sizes': (15, 1), 'result': 0.8842695892537753}, {'id': 'ADA_16_1', 'solver': 'adam', 'hidden_layer_sizes': (16, 1), 'result': 0.8868588482740897}, {'id': 'ADA_17_1', 'solver': 'adam', 'hidden_layer_sizes': (17, 1), 'result': 0.8985124807897069}, {'id': 'ADA_18_1', 'solver': 'adam', 'hidden_layer_sizes': (18, 1), 'result': 0.9015887502425874}, {'id': 'ADA_19_1', 'solver': 'adam', 'hidden_layer_sizes': (19, 1), 'result': 0.8944653375504188}, {'id': 'ADA_20_1', 'solver': 'adam', 'hidden_layer_sizes': (20, 1), 'result': 0.8952754480653333}, {'id': 'ADA_21_1', 'solver': 'adam', 'hidden_layer_sizes': (21, 1), 'result': 0.8990013270181955}, {'id': 'ADA_22_1', 'solver': 'adam', 'hidden_layer_sizes': (22, 1), 'result': 0.8907378850581947}, {'id': 'ADA_23_1', 'solver': 'adam', 'hidden_layer_sizes': (23, 1), 'result': 0.896733332284307}, {'id': 'ADA_24_1', 'solver': 'adam', 'hidden_layer_sizes': (24, 1), 'result': 0.8962473708779826}, {'id': 'ADA_25_1', 'solver': 'adam', 'hidden_layer_sizes': (25, 1), 'result': 0.8998075036847049}, {'id': 'ADA_26_1', 'solver': 'adam', 'hidden_layer_sizes': (26, 1), 'result': 0.8913901171237798}, {'id': 'ADA_27_1', 'solver': 'adam', 'hidden_layer_sizes': (27, 1), 'result': 0.8930087646142468}, {'id': 'ADA_28_1', 'solver': 'adam', 'hidden_layer_sizes': (28, 1), 'result': 0.8875058352084677}, {'id': 'ADA_29_1', 'solver': 'adam', 'hidden_layer_sizes': (29, 1), 'result': 0.8943061478182877}, {'id': 'ADA_30_1', 'solver': 'adam', 'hidden_layer_sizes': (30, 1), 'result': 0.8837815297949678}, {'id': 'ADA_31_1', 'solver': 'adam', 'hidden_layer_sizes': (31, 1), 'result': 0.8983514552616534}, {'id': 'ADA_32_1', 'solver': 'adam', 'hidden_layer_sizes': (32, 1), 'result': 0.8933337004925178}, {'id': 'ADA_33_1', 'solver': 'adam', 'hidden_layer_sizes': (33, 1), 'result': 0.8931697901423004}, {'id': 'ADA_34_1', 'solver': 'adam', 'hidden_layer_sizes': (34, 1), 'result': 0.8904202923636134}, {'id': 'ADA_35_1', 'solver': 'adam', 'hidden_layer_sizes': (35, 1), 'result': 0.8960852963236874}, {'id': 'ADA_36_1', 'solver': 'adam', 'hidden_layer_sizes': (36, 1), 'result': 0.8985148410987499}, {'id': 'ADA_37_1', 'solver': 'adam', 'hidden_layer_sizes': (37, 1), 'result': 0.89608713211961}, {'id': 'ADA_38_1', 'solver': 'adam', 'hidden_layer_sizes': (38, 1), 'result': 0.898351193005093}, {'id': 'ADA_39_1', 'solver': 'adam', 'hidden_layer_sizes': (39, 1), 'result': 0.8954372603630679}, {'id': 'ADA_40_1', 'solver': 'adam', 'hidden_layer_sizes': (40, 1), 'result': 0.8928474768296327}, {'id': 'ADA_41_1', 'solver': 'adam', 'hidden_layer_sizes': (41, 1), 'result': 0.8876660739668404}, {'id': 'ADA_42_1', 'solver': 'adam', 'hidden_layer_sizes': (42, 1), 'result': 0.8931690033726195}, {'id': 'ADA_43_1', 'solver': 'adam', 'hidden_layer_sizes': (43, 1), 'result': 0.896894620068921}, {'id': 'ADA_44_1', 'solver': 'adam', 'hidden_layer_sizes': (44, 1), 'result': 0.8934973485861748}, {'id': 'ADA_45_1', 'solver': 'adam', 'hidden_layer_sizes': (45, 1), 'result': 0.8999701027521205}, {'id': 'ADA_46_1', 'solver': 'adam', 'hidden_layer_sizes': (46, 1), 'result': 0.8989981799394713}, {'id': 'ADA_47_1', 'solver': 'adam', 'hidden_layer_sizes': (47, 1), 'result': 0.8970580059060177}, {'id': 'ADA_48_1', 'solver': 'adam', 'hidden_layer_sizes': (48, 1), 'result': 0.8949526102395451}, {'id': 'ADA_49_1', 'solver': 'adam', 'hidden_layer_sizes': (49, 1), 'result': 0.897704992840396}, {'id': 'ADA_50_1', 'solver': 'adam', 'hidden_layer_sizes': (50, 1), 'result': 0.8964091831757172}, {'id': 'ADA_1_2', 'solver': 'adam', 'hidden_layer_sizes': (1, 2), 'result': 0.8837815297949678}, {'id': 'ADA_2_2', 'solver': 'adam', 'hidden_layer_sizes': (2, 2), 'result': 0.9002937273475897}, {'id': 'ADA_3_2', 'solver': 'adam', 'hidden_layer_sizes': (3, 2), 'result': 0.9028840353941453}, {'id': 'ADA_4_2', 'solver': 'adam', 'hidden_layer_sizes': (4, 2), 'result': 0.904828143276004}, {'id': 'ADA_5_2', 'solver': 'adam', 'hidden_layer_sizes': (5, 2), 'result': 0.9030419138434749}, {'id': 'ADA_6_2', 'solver': 'adam', 'hidden_layer_sizes': (6, 2), 'result': 0.8994849281154768}, {'id': 'ADA_7_2', 'solver': 'adam', 'hidden_layer_sizes': (7, 2), 'result': 0.8977026325313527}, {'id': 'ADA_8_2', 'solver': 'adam', 'hidden_layer_sizes': (8, 2), 'result': 0.9030461099484404}, {'id': 'ADA_9_2', 'solver': 'adam', 'hidden_layer_sizes': (9, 2), 'result': 0.9028824618547834}, {'id': 'ADA_10_2', 'solver': 'adam', 'hidden_layer_sizes': (10, 2), 'result': 0.8955967123517595}, {'id': 'ADA_11_2', 'solver': 'adam', 'hidden_layer_sizes': (11, 2), 'result': 0.9014272002014131}, {'id': 'ADA_12_2', 'solver': 'adam', 'hidden_layer_sizes': (12, 2), 'result': 0.8988379411810987}, {'id': 'ADA_13_2', 'solver': 'adam', 'hidden_layer_sizes': (13, 2), 'result': 0.8926861890450191}, {'id': 'ADA_14_2', 'solver': 'adam', 'hidden_layer_sizes': (14, 2), 'result': 0.8951128489979177}, {'id': 'ADA_15_2', 'solver': 'adam', 'hidden_layer_sizes': (15, 2), 'result': 0.8967320210015053}, {'id': 'ADA_16_2', 'solver': 'adam', 'hidden_layer_sizes': (16, 2), 'result': 0.8944655998069793}, {'id': 'ADA_17_2', 'solver': 'adam', 'hidden_layer_sizes': (17, 2), 'result': 0.8981901674770395}, {'id': 'ADA_18_2', 'solver': 'adam', 'hidden_layer_sizes': (18, 2), 'result': 0.9023957136787777}, {'id': 'ADA_19_2', 'solver': 'adam', 'hidden_layer_sizes': (19, 2), 'result': 0.8993215422783802}, {'id': 'ADA_20_2', 'solver': 'adam', 'hidden_layer_sizes': (20, 2), 'result': 0.8926848777622174}, {'id': 'ADA_21_2', 'solver': 'adam', 'hidden_layer_sizes': (21, 2), 'result': 0.8972166711250281}, {'id': 'ADA_22_2', 'solver': 'adam', 'hidden_layer_sizes': (22, 2), 'result': 0.8986735063177607}, {'id': 'ADA_23_2', 'solver': 'adam', 'hidden_layer_sizes': (23, 2), 'result': 0.902237048459767}, {'id': 'ADA_24_2', 'solver': 'adam', 'hidden_layer_sizes': (24, 2), 'result': 0.9002934650910293}, {'id': 'ADA_25_2', 'solver': 'adam', 'hidden_layer_sizes': (25, 2), 'result': 0.8930087646142468}, {'id': 'ADA_26_2', 'solver': 'adam', 'hidden_layer_sizes': (26, 2), 'result': 0.8930077155880056}, {'id': 'ADA_27_2', 'solver': 'adam', 'hidden_layer_sizes': (27, 2), 'result': 0.8954343755409042}, {'id': 'ADA_28_2', 'solver': 'adam', 'hidden_layer_sizes': (28, 2), 'result': 0.8946279366178347}, {'id': 'ADA_29_2', 'solver': 'adam', 'hidden_layer_sizes': (29, 2), 'result': 0.8978644448290873}, {'id': 'ADA_30_2', 'solver': 'adam', 'hidden_layer_sizes': (30, 2), 'result': 0.8970564323666557}, {'id': 'ADA_31_2', 'solver': 'adam', 'hidden_layer_sizes': (31, 2), 'result': 0.8968917352467572}, {'id': 'ADA_32_2', 'solver': 'adam', 'hidden_layer_sizes': (32, 2), 'result': 0.8967330700277467}, {'id': 'ADA_33_2', 'solver': 'adam', 'hidden_layer_sizes': (33, 2), 'result': 0.8936552270355044}, {'id': 'ADA_34_2', 'solver': 'adam', 'hidden_layer_sizes': (34, 2), 'result': 0.8977031570444733}, {'id': 'ADA_35_2', 'solver': 'adam', 'hidden_layer_sizes': (35, 2), 'result': 0.8938165148201183}, {'id': 'ADA_36_2', 'solver': 'adam', 'hidden_layer_sizes': (36, 2), 'result': 0.8951154715635212}, {'id': 'ADA_37_2', 'solver': 'adam', 'hidden_layer_sizes': (37, 2), 'result': 0.8957611472150976}, {'id': 'ADA_38_2', 'solver': 'adam', 'hidden_layer_sizes': (38, 2), 'result': 0.8865346991654997}, {'id': 'ADA_39_2', 'solver': 'adam', 'hidden_layer_sizes': (39, 2), 'result': 0.8970548588272935}, {'id': 'ADA_40_2', 'solver': 'adam', 'hidden_layer_sizes': (40, 2), 'result': 0.8926830419662949}, {'id': 'ADA_41_2', 'solver': 'adam', 'hidden_layer_sizes': (41, 2), 'result': 0.8980288796924254}, {'id': 'ADA_42_2', 'solver': 'adam', 'hidden_layer_sizes': (42, 2), 'result': 0.8938180883594804}, {'id': 'ADA_43_2', 'solver': 'adam', 'hidden_layer_sizes': (43, 2), 'result': 0.8967312342318243}, {'id': 'ADA_44_2', 'solver': 'adam', 'hidden_layer_sizes': (44, 2), 'result': 0.8920389398540804}, {'id': 'ADA_45_2', 'solver': 'adam', 'hidden_layer_sizes': (45, 2), 'result': 0.8946281988743949}, {'id': 'ADA_46_2', 'solver': 'adam', 'hidden_layer_sizes': (46, 2), 'result': 0.8886377345229292}, {'id': 'ADA_47_2', 'solver': 'adam', 'hidden_layer_sizes': (47, 2), 'result': 0.8894491563206455}, {'id': 'ADA_48_2', 'solver': 'adam', 'hidden_layer_sizes': (48, 2), 'result': 0.8917142662323698}, {'id': 'ADA_49_2', 'solver': 'adam', 'hidden_layer_sizes': (49, 2), 'result': 0.8973813682449266}, {'id': 'ADA_50_2', 'solver': 'adam', 'hidden_layer_sizes': (50, 2), 'result': 0.8918742427341819}]
# plot
plot(scores)

#print_markdown_table(scores)