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

### Comitê Heterogêneo

