## Projeto de Mineração de Dados: Bank Marketing Data Set

### Relatório 3: Algoritmos de Classificação

#### Equipe
* Antonio Augusto Correa Gondim Neto (aacgn)
* Eduardo Santos de Moura (esm7)
* Marcos Vinicius Holanda Borges (mvhb)
* Vinícius Giles Costa Paulino (vgcp)

### Validação cruzada
*Cross-validation* (validação cruzada) é um procedimento de reamostragem usado para avaliar modelos de aprendizado de máquina em uma amostra de dados limitada. O **k-fold** é utilizado em amostragens menores com o objetivo de estimar o quanto é esperado da performance do modelo de maneira geral quando utilizado para fazer predições em dados que não foram utilizados durante o treinamento do modelo. O procedimento possui um único parâmetro (*k*) que se refere ao número de grupos nos quais uma determinada amostra de dados deve ser dividida. Uma escolha errada do número *k* pode acabar levando a distorções do modelo (com altas variações), sendo assim, para o nosso conjunto de dados foi utilizado o *k* como `10`, uma vez que esse valor tem demonstrado ser seguro (baixa tendência e uma variação modesta) dado inúmeras experimentações feitas com esse valor. 

```python
K_SPLITS = 10
kf = KFold(n_splits=K_SPLITS)
```
Além disso, a semente de geração de números aleatórios utilizada em todos as aplicações dos algoritmos é `42`:

```python
np.random.seed(42)
```

### Decision Tree
Foi utilizado o **DecisionTreeClassifier** para criação do modelo, que é uma classe capaz de executar a classificação multi-classe em um conjunto de dados. De modo a criar configurações diferentes para tentar extrair o melhor resultado possível, foram feitas variações dos parâmetros *criterion* e *max_depth*. Além disso, o parâmetro *random_state* (`42`) também foi utilizado.

No *criterion*, foram utilizados dois valores: *gini* e *entropy*. O primeiro diz respeito a impureza de Gini, enquanto o segundo para o ganho de informações. Além disso, Gini é mais utilizado para minimizar erros de classificação e a *entropy* é mais voltado para análise exploratória. A tabela abaixo apresenta um histórico todas as variações de parâmetros verificadas:

##### Histórico das buscas pelos parâmetros do Decision Tree
Variação |     *criterion*       | *max_depth* | Acurácia média
-------- | ------------------ | ------------- | --------
c1 | gini |  7 | 0.923
c2 | gini |  5 | 0.917
c3 | gini |  8 | 0.927
c4 | gini | 12 | **0.951** &ast;
c5 | gini | 3  | 0.909
c6 | entropy | 1 |  0.887
c7 | entropy | 5 | 0.915
c8 | entropy | 7 | 0.921
c9 | entropy | 12 | 0.946
c10 | *entropy* | 6 | 0.918

&ast;Melhor variação

Ou seja, é possível observar que a medida que o valor de *max_depth* aumenta — independente do valor de *criterion* —  assim também ocorre com a média de **k-fold** obtida. Nesse algoritmo, a melhor variação foi a **c4**.

### KNN: K-Vizinhos mais próximos

No algoritmo `KNN` (*k-nearest neighbors*), os parâmetros necessários são *metric* e *n_neighbors*. O *metric* serve define a métrica de distância que será utilizada para comparar a distância entre dois pontos, podendo assumir os seguintes valores: [*manhattan*, *euclidean*, *chebyshev*, *minkowski*]. Já o *n_neighbors* representa o **k**, ou seja, a quantidade de vizinhos mais próximos para se fazer a classificação. Em todas as variações dos parâmetros foram utilizados valores ímpares para essa parâmetro, pois, caso fosse par, existe uma possibilidade de empate entre duas classes.

##### Histórico das buscas pelos parâmetros do KNN

Variação |     *metric*       | *n_neighbors* | Acurácia média
-------- | ------------------ | ------------- | --------
A | manhattan | 11  | 0.959
B | manhattan | 101 | 0.923
C | manhattan | 303 | 0.912
D | euclidean | 11  | **0.960** *
E | euclidean | 101 | 0.931
F | euclidean | 303 | 0.919
G | chebyshev | 11  | 0.924
H | chebyshev | 101 | 0.910
I | chebyshev | 303 | 0.906
J | minkowski | 11  | **0.960** *
K | minkowski | 101 | 0.931
L | minkowski | 303 | 0.919

&ast;Melhores variações

A tabela acima apresenta um histórico todas as variações de parâmetros verificadas, nomeadas de A a L e suas acurácias médias no conjunto de treinamento do nosso conjunto de dados. A busca pelos melhores parâmetros pode ser encontrada no arquivo *knn_historic.py*. Como visto na tabela, as variações **D** (*metric* = `'euclidean'`, *n_neighbors* = `11`) e **J** (*metric* = `'minkowski'`, *n_neighbors* = `11`) conseguiram os melhores desempenhos.

### Redes Neurais

Para a base de dados em questão, foi utilizado o **MLPClassifier** para construção do modelo. As variações feitas para encontrar o melhor resultado possível foram feitas nos parâmetros *solver*, *hidden_layer_size*, e *activation*. Além disso também foi utilizado o *random_state* (`42`).

Sobre o *hidden_layer_sizes*, ele recebe o i-ésimo elemento que representa o número de neurônios na i-ésima camada oculta. Quanto ao parâmetro ‘solver’, foram utilizadas três tipos diferentes, são eles: {`'lbfgs'`, `'sgd'`, `'adam'`}. `'lbfgs'` é um otimizador na família de métodos quasi-Newton. Já `'sgd'` refere-se à descida do gradiente estocástico e `'adam'` refere-se a um otimizador estocástico baseado em gradiente proposto por Kingma, Diederik e Jimmy Ba. O ‘solver’ padrão, que é `'adam'`, funciona muito bem em conjuntos de dados relativamente grandes (com milhares de amostras de treinamento ou mais) em termos de tempo de treinamento e pontuação de validação. Para conjuntos de dados pequenos, no entanto, o `'lbfgs'` pode convergir mais rapidamente e ter um desempenho melhor. Já parâmetro *activation*, tem as seguintes opções de variação: *identity*, *logistic*, *tanh* e *relu*. Ela é a função de ativação para o hidden layer.

##### Histórico das buscas pelos parâmetros

Variação |       solver       |  hidden_layer_sizes  | activation | Acurácia média
-------- | ------------------ | -------------------- | ---------- | ---------------- |
mlp01 | lbfgs | (3, 3) | identity | 0.911
mlp02 | lbfgs | (30, 3) | identity | 0.911
mlp03 | lbfgs | (3, 30) | identity | 0.911
mlp04 | lbfgs | (30, 30) | identity | 0.911
mlp05 | lbfgs | (3, 3) | logistic | 0.914
mlp06 | lbfgs | (30, 3) | logistic | 0.924
mlp07 | lbfgs | (3, 30) | logistic | 0.913
mlp08 | lbfgs | (30, 30) | logistic | 0.925
mlp09 | lbfgs | (3, 3) | tanh | 0.916
mlp10 | lbfgs | (30, 3) | tanh | 0.933
mlp11 | lbfgs | (3, 30) | tanh | 0.916
mlp12 | lbfgs | (30, 30) | tanh | 0.943
mlp13 | lbfgs | (3, 3) | relu | 0.915
mlp14 | lbfgs | (30, 3) | relu | 0.928
mlp15 | lbfgs | (3, 30) | relu | 0.913
mlp16 | lbfgs | (30, 30) | relu | 0.936
mlp17 | sgd | (3, 3) | identity | 0.910
mlp18 | sgd | (30, 3) | identity | 0.911
mlp19 | sgd | (3, 30) | identity | 0.910
mlp20 | sgd | (30, 30) | identity | 0.910
mlp21 | sgd | (3, 3) | logistic | 0.887
mlp22 | sgd | (30, 3) | logistic | 0.911
mlp23 | sgd | (3, 30) | logistic | 0.911
mlp24 | sgd | (30, 30) | logistic | 0.913
mlp25 | sgd | (3, 3) | tanh | 0.911
mlp26 | sgd | (30, 3) | tanh | 0.913
mlp27 | sgd | (3, 30) | tanh | 0.910
mlp28 | sgd | (30, 30) | tanh | 0.915
mlp29 | sgd | (3, 3) | relu | 0.910
mlp30 | sgd | (30, 3) | relu | 0.915
mlp31 | sgd | (3, 30) | relu | 0.910
mlp32 | sgd | (30, 30) | relu | 0.915
mlp33 | adam | (3, 3) | identity | 0.911
mlp34 | adam | (30, 3) | identity | 0.911
mlp35 | adam | (3, 30) | identity | 0.915
mlp36 | adam | (30, 30) | identity | 0.911
mlp37 | adam | (3, 3) | logistic | 0.915
mlp38 | adam | (30, 3) | logistic | 0.924
mlp39 | adam | (3, 30) | logistic | 0.915
mlp40 | adam | (30, 30) | logistic | 0.926
mlp41 | adam | (3, 3) | tanh | 0.915
mlp42 | adam | (30, 3) | tanh | 0.937
mlp43 | adam | (3, 30) | tanh | 0.916
mlp44 | adam | (30, 30) | tanh | **0.946** *
mlp45 | adam | (3, 3) | relu | 0.913
mlp46 | adam | (30, 3) | relu | 0.931
mlp47 | adam | (3, 30) | relu | 0.916
mlp48 | adam | (30, 30) | relu | 0.943
&ast;Melhor variação

A configuração com maior desepenho foi a **mlp44**. É possível perceber que um maior número de hidden layers e o de neurônios por camada não necessariamente implica em uma maior acurácia, mas também é possível constatar que os melhores resultados no conjunto de treinamento foram obtidos com a configuração *(30, 30)*.