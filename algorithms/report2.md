## Projeto de Mineração de Dados: Bank Marketing Data Set

### Relatório 4: Algoritmos de Classificação

#### Equipe
* Antonio Augusto Correa Gondim Neto (aacgn)
* Eduardo Santos de Moura (esm7)
* Marcos Vinicius Holanda Borges (mvhb)
* Vinícius Giles Costa Paulino (vgcp)

### Random Forest
É utilizado para construção de *Ensemble Based Systems* (EBS) com árvores de decisão. Esse classificador utiliza variações na quantidade de dados, características além de árvores de decisão com diferentes inicializações para obter uma predição com maior acurácia e mais estável.

Para esse modelo em questão, foi utilizada a função `RandomForestClassifier` para construção do modelo. As variações feitas para encontrar o melhor resultado possível foram feitas nos parâmetros ‘n_estimators’ e ‘max_features’. Além disso, também foi utilizado o ‘random_state’ (`42`).

O primeiro parâmetro citado (*n_estimators*) indica o número de árvores construídas pelo algoritmo antes de tomar uma votação ou fazer uma média de predições. Normalmente, quanto maior a quantidade de árvores, maior também é a performance e custo computacional, além de deixar as predições mais estáveis. Já o segundo parâmetro utilizado (*max_features*), diz respeito ao número máximo de características a serem utilizadas pelo Random Forest na construção de uma dada árvore.

##### Histórico das buscas pelos parâmetros do Random Forest
Variação |     *n_estimators*       | *max_features* | Acurácia média
-------- | ------------------------ | -------------- | --------
rf1  | 10 |   7    | 0.995
rf2  | 16 | `None` | **0.997** &ast;
rf3  | 4  |   8    | 0.982
rf4  | 8  |  10    | 0.993
rf5  | 4  |   7    | 0.983
rf6  | 12 |   6    | 0.996
rf7  | 6  |  12    | 0.989
rf8  | 5  |   7    | 0.988
rf9  | 7  |   4    | 0.992
rf10 | 9  |   6    | 0.994

&ast;Melhor variação

O classificador Random Forest teve a melhor acurácia média, no valor de **0.997**, com a configuração **rf2** (*n_estimators* = 16, *max_features* = `None`).

### Comitê de Redes Neurais

Um comitê de Redes Neurais é um método de aprendizado, supervisionado ou não, cujo o objetivo é aumentar a capacidade de generalização de parâmetros de estimadores MLP. Para esse experimento, selecionamos as redes que melhor performaram na busca pelos melhores parâmetros:

Variação |       solver       |  hidden_layer_sizes  | activation | Acurácia média
-------- | ------------------ | -------------------- | ---------- | ---------------- |
MLP44 | adam | (30, 30) | tanh | 0.946
MLP48 | adam | (30, 30) | relu | 0.943
MLP42 | adam | (30, 3) | tanh | 0.937
MLP16 | lbfgs | (30, 30) | relu | 0.936
MLP10 | lbfgs | (30, 3) | tanh | 0.933

As 5 redes neurais foram agrupadas em um comitê `VotingClassifier`, um comitê baseado no método de votação majoritária. Para buscar a melhor configuração, variamos o parâmetro *voting*, que pode assumir os valores `'hard'` e`'soft'`.

```python
estimators = [
    ('mlp44', mlp44),
    ('mlp48', mlp48),
    ('mlp42', mlp42),
    ('mlp16', mlp16),
    ('mlp10', mlp10),
]

ensembleSoft = VotingClassifier(estimators, voting='soft')
ensembleHard = VotingClassifier(estimators, voting='hard')
```
Dessa forma, os resultados foram:

##### Resultado comitê de redes neurais
voting |  Acurácia | 
-------- | ------ |
soft | **0.8886**
hard | **0.8882**

O comitê com `voting` setado como **soft** apresentou ligeiramente uma melhor acurácia média.

### Comitê Heterogêneo

No comitê heterogêneo, foram escolhidos os seguintes classificadores com os parâmetros que melhor se saíram na etapa anterior:

* **Random Forest** (*n_estimators* = 16, 'max_features' = `None`)
* **KNN** (*metric* = `euclidean`, *n_neighbors* = 11)
* **Rede Neural** (*solver* = `'adam'`, *hidden_layer_sizes*=(30,30), *activation* = `'tahn'`)
* **Ridge** (parâmetros default) 

Os 4 classificadores foram agrupados em outro comitê `VotingClassifier`, e para buscar a melhor configuração, variamos o parâmetro *voting* novamente. Os resultados individuais de cada classificador seguem na tabela abaixo:

Classificador | Acurácia média |
--- | ---
RandomForestClassifier |  0.893
KNeighborsClassifier | 0.899
MLPClassifier | 0.877
RidgeClassifier | 0.899

E os resultados do comitê heterogêneo:
##### Resultado comitê heterogêneo
voting |  Acurácia | 
-------- | ------ |
soft | **0.9035**
hard | **0.9012**

O comitê com `voting` setado como **soft** novamente apresentou ligeiramente uma melhor acurácia média.