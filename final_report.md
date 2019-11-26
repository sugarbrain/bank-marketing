## Projeto de Mineração de Dados: Bank Marketing Data Set

### Relatório Final

#### Equipe
* Antonio Augusto Correa Gondim Neto (aacgn)
* Eduardo Santos de Moura (esm7)
* Marcos Vinicius Holanda Borges (mvhb)
* Vinícius Giles Costa Paulino (vgcp)

### Resumo
Este projeto de mineração de dados foi executado com o objetivo de conferir aprendizado a respeito das etapas do processo CRISP-DM (*CRoss Industry Standard Process for Data Mining*), algoritmos de mineração de dados e ferramentas que auxiliam a elaboração de modelos de predição.

### Contexto e domínio do problema
O conjunto de dados Bank Marketing Data Set está relacionado com campanhas de marketing direto de uma instituição bancária portuguesa. As campanhas de marketing foram baseadas em telefonemas. Frequentemente era necessário mais de um contato para o mesmo cliente, para acessar se o produto (depósito bancário) seria ('yes') ou não ('no') assinado. O objetivo da classificação realizada na base de dados é prever se o cliente assinará um depósito a prazo (variável y).

O arquivo **bank-additional-full.csv**, que será utilizado neste projeto de classificação, apresenta 41.188 registros contando com 21 colunas descritas na seguinte tabela:

Coluna | Tipo | Descrição | Categorias
---|---|---|---
y (classe) | Categórico | O cliente assinará o depósito a prazo? | “yes”, “no”
age |Numérico | Idade | N/A
job |Categórico|Emprego|"admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur","student","blue-collar", "self-employed", "retired", "technician", "services"
marital|Categórico|Estado civil|"married", "divorced", "single"
education|Categórico|Nível educacional|"unknown", "secondary","primary", "tertiary"
default|Categórico|Tem crédito por padrão?|"no","yes","unknown"
housing|Categórico|Tem empréstimo para moradia?|"no","yes","unknown"
loan|Categórico|Tem empréstimo pessoal?|"no","yes","unknown"
contact|Categórico|O tipo de comunicação utilizada|"cellular","telephone"
month|Categórico|Mês do último contato|"jan", "feb", ..., "nov", "dec"
day of week|Categórico|Dia da semana do último contato|"mon","tue","wed","thu","fri"
duration|Numérico|Duração do último contato feito|N/A
campaign|Numérico|Número de contatos realizados durante esta campanha e para este mesmo cliente|N/A
pdays|Numérico|Número de dias desde que o cliente foi contactado pela última vez por uma campanha anterior|N/A
previous|Numérico|Número de contatos realizados antes desta campanha para este mesmo cliente|N/A
poutcome|Categórico|Resultado da campanha anterior|"failure","nonexistent","success"
emp.var.rate|Numérico|Taxa de variação do emprego (indicador trimestral)|N/A
cons.price.idx|Numérico|Índice de preços ao consumidor (indicador mensal)|N/A
cons.conf.idx|Numérico|Índice de confiança do consumidor (iindicador mensal)|N/A
euribor3m|Numérico|Euribor taxa de 3 meses (indicador diário)|N/A
nr.employed|Numérico|Número de funcionários (indicador trimestral)|N/A

#### Descrição dos dados
Abaixo está apresentada a descrição dos dados a partir dos métodos de análise exploratória da biblioteca pandas:

```
# dataset.dtypes
age                 int64
job                object
marital            object
education          object
default            object
housing            object
loan               object
contact            object
month              object
day_of_week        object
duration            int64
campaign            int64
pdays               int64
previous            int64
poutcome           object
emp.var.rate      float64
cons.price.idx    float64
cons.conf.idx     float64
euribor3m         float64
nr.employed       float64
y                  object
dtype: object

# dataset.describe()
               age      duration  ...     euribor3m   nr.employed
count  41188.00000  41188.000000  ...  41188.000000  41188.000000
mean      40.02406    258.285010  ...      3.621291   5167.035911
std       10.42125    259.279249  ...      1.734447     72.251528
min       17.00000      0.000000  ...      0.634000   4963.600000
25%       32.00000    102.000000  ...      1.344000   5099.100000
50%       38.00000    180.000000  ...      4.857000   5191.000000
75%       47.00000    319.000000  ...      4.961000   5228.100000
max       98.00000   4918.000000  ...      5.045000   5228.100000
```
#### Valores ausentes
Existem vários valores ausentes em alguns atributos categóricos, todos codificados com o rótulo **unknown**. Esses valores ausentes serão tratados como um possível rótulo de classe ou usando técnicas de exclusão ou imputação. Os valores ausentes desse conjunto de dados são listados no detalhamento abaixo:
```
# dataset.isnull().sum()
age                  0
job                330
marital             80
education         1731
default           8597
housing            990
loan               990
contact              0
month                0
day_of_week          0
duration             0
campaign             0
pdays                0
previous             0
poutcome             0
emp.var.rate         0
cons.price.idx       0
cons.conf.idx        0
euribor3m            0
nr.employed          0
y                    0
dtype: int64
```

#### Gráficos das colunas categóricas
![job](/final/bar_job.png)

![marital](/final/bar_marital.png)

![education](/final/bar_education.png)

![default](/final/bar_default.png)

![housing](/final/bar_housing.png)

![loan](/final/bar_loan.png)

![contact](/final/bar_contact.png)

![month](/final/bar_month.png)

![day_of_week](/final/bar_day_of_week.png)

![poutcome](/final/bar_poutcome.png)

![y](/final/bar_y.png)

![box](/final/box_plots.png)

### Limpeza e pré-processamento
#### Tratamento de valores ausentes
Durante a análise exploratória do conjunto de dados `Bank Marketing Data Set`, foi observado que das 21 colunas, apenas seis apresentam dados faltantes, sendo elas: *job*, *marital*, *education*, *default*, *housing* e *loan*. Todos os valores ausentes estão identificados por padrão como a string `'unknown'`. Dessa forma, executamos a leitura do arquivo `csv` levando isso em consideração:

```python
dataset = pandas.read_csv(file_name, sep=";", na_values="unknown")
```

Com um alto número de valores ausentes, foi realizada sua substituição com a função `fillna` com o método `ffill`:
```python
dataset.fillna(method='ffill', inplace=True)
```
Essa função preenche os valores faltantes utilizando sempre o valor do registro anterior. Ao realizar uma comparação dos histogramas de cada coluna antes e depois de preencher os valores ausentes, comparando médias, desvios padrão, mínimos e máximos e demais dados estatísticos, foi constatado que não houve mudanças significativas nas distribuições.

#### Normalização
O segundo passo do pré-processamento foi realizar normalização nas colunas. Para as colunas numéricas (`age`, `duration`, `campaign`, `pdays`,`previous`, `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m` e `nr.employed`) foi utilizado o `StandardScaler` para facilitar as futuras análises, como com KNN eliminando as diferentes dispersões entre as colunas.

```python
> print(dataset.head(20))
```

```
    day_of_week  duration  campaign     pdays  previous  poutcome  \
0             1  0.005792 -0.559335  0.211887 -0.371616         1  
2             1 -0.127944 -0.559335  0.211887 -0.371616         1  
3             1 -0.414520 -0.559335  0.211887 -0.371616         1  
4             1  0.181559 -0.559335  0.211887 -0.371616         1  
6             1 -0.460373 -0.559335  0.211887 -0.371616         1  
8             1  0.460494 -0.559335  0.211887 -0.371616         1  
9             1 -0.800444 -0.559335  0.211887 -0.371616         1  
11            1 -0.143228 -0.559335  0.211887 -0.371616         1  
12            1 -0.468015 -0.559335  0.211887 -0.371616         1  
13            1  0.128065 -0.559335  0.211887 -0.371616         1  
14            1 -0.433625 -0.559335  0.211887 -0.371616         1  
16            1  0.200665 -0.559335  0.211887 -0.371616         1  
18            1  0.357327 -0.559335  0.211887 -0.371616         1  
20            1 -0.846296 -0.559335  0.211887 -0.371616         1  
22            1  0.315295 -0.559335  0.211887 -0.371616         1  
23            1 -0.299890 -0.559335  0.211887 -0.371616         1  
24            1 -0.334279 -0.559335  0.211887 -0.371616         1  
25            1 -0.613214 -0.559335  0.211887 -0.371616         1  
34            1 -0.196722 -0.559335  0.211887 -0.371616         1  
36            1  0.403179 -0.559335  0.211887 -0.371616         1  

   emp.var.rate  cons.price.idx  cons.conf.idx  euribor3m  nr.employed  y  
0       0.727477        0.804095       0.877451   0.786102     0.401648  0  
2       0.727477        0.804095       0.877451   0.786102     0.401648  0  
3       0.727477        0.804095       0.877451   0.786102     0.401648  0  
4       0.727477        0.804095       0.877451   0.786102     0.401648  0  
6       0.727477        0.804095       0.877451   0.786102     0.401648  0  
8       0.727477        0.804095       0.877451   0.786102     0.401648  0  

```
Para as colunas categóricas, (`job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day of week` e `poutcome`) foi utilizado o `get_dummies`. Essa função cria uma nova coluna para cada categoria da coluna inicial de forma a binarizar seus valores. Por exemplo: a coluna `job`, contando com 11 categorias, foi transformada em 11 colunas: `job_admin.`, `job_technician`, `job_blue-collar`... cada uma admitindo `0` ou `1` como valor.

Além disso, para a coluna classe (`y`) foi utilizado o `LabelBinarizer` para transformar as strings `'no'` e `'yes'` nos valores `0` e `1`.

O total de colunas do conjunto de dados após o pré-processamento, após a operação, foi de `58`:
```
# dataset.dtypes
age                              float64
duration                         float64
campaign                         float64
pdays                            float64
previous                         float64
emp.var.rate                     float64
cons.price.idx                   float64
cons.conf.idx                    float64
euribor3m                        float64
nr.employed                      float64
job_admin.                         uint8
job_blue-collar                    uint8
job_entrepreneur                   uint8
job_housemaid                      uint8
job_management                     uint8
job_retired                        uint8
job_self-employed                  uint8
job_services                       uint8
job_student                        uint8
job_technician                     uint8
job_unemployed                     uint8
marital_divorced                   uint8
marital_married                    uint8
marital_single                     uint8
education_basic.4y                 uint8
education_basic.6y                 uint8
education_basic.9y                 uint8
education_high.school              uint8
education_illiterate               uint8
education_professional.course      uint8
education_university.degree        uint8
default_no                         uint8
default_yes                        uint8
housing_no                         uint8
housing_yes                        uint8
loan_no                            uint8
loan_yes                           uint8
contact_cellular                   uint8
contact_telephone                  uint8
month_apr                          uint8
month_aug                          uint8
month_dec                          uint8
month_jul                          uint8
month_jun                          uint8
month_mar                          uint8
month_may                          uint8
month_nov                          uint8
month_oct                          uint8
y                                  uint8
```

### Preparação do conjunto de dados
Após a etapa de pré-processamento, o conjunto de dados foi dividido em conjunto de treino e conjuno de teste, numa proporção de 25:75. Assim, apenas o conjunto de treino, uma amostra de 25% dos dados, foi utilizado nas etapas seguintes.

### Algoritmos de Mineração

#### Árvore de Decisão
Foi utilizado o **DecisionTreeClassifier** para criação do modelo, que é uma classe capaz de executar a classificação multi-classe em um conjunto de dados. De modo a criar configurações diferentes para tentar extrair o melhor resultado possível, foram feitas variações dos parâmetros *criterion* e *max_depth*. Além disso, o parâmetro *random_state* (`42`) também foi utilizado.

No *criterion*, foram utilizados dois valores: *gini* e *entropy*. O primeiro diz respeito a impureza de Gini, enquanto o segundo para o ganho de informações. Além disso, Gini é mais utilizado para minimizar erros de classificação e a *entropy* é mais voltado para análise exploratória. No parâmetro *max_depth*, aplicamos um range de valores de 1 a 50 a fim de obter uma variedade maior na busca pela melhor configuração. 

Dessa forma, obtivemos 100 diferentes configurações de parâmetros, as quais foram submetidas a validação cruzada utilizando o `StratifiedKFold` com `k` = 10 em 20% do conjunto de treinamento. A média dos valores obtidos com o `KFold` foi calculada e guardada para posterior análise. A partir dessas informações foi plotado o gráfico abaixo:

![Resultados Árvore de Decisão](/algorithms/plots/decision_tree_historic_plot2.png)

A melhor configuração encontrada a partir da experimentação foi a **GIN5**, conseguindo uma acurácia média de **0.9048** com os seguintes parâmetros:

```
GIN5 => { 'criterion': 'gini', 'max_depth': 5 }
``` 

#### KNN: K-Vizinhos Mais Próximos

No algoritmo `KNN` (*k-nearest neighbors*), os parâmetros necessários são *metric* e *n_neighbors*. O *metric* serve define a métrica de distância que será utilizada para comparar a distância entre dois pontos, podendo assumir os seguintes valores: [*manhattan*, *euclidean*, *chebyshev*, *minkowski*]. Já o *n_neighbors* representa o **k**, ou seja, a quantidade de vizinhos mais próximos para se fazer a classificação. Em todas as variações dos parâmetros foram utilizados valores ímpares para essa parâmetro, pois, caso fosse par, existe uma possibilidade de empate entre duas classes.

Obtivemos 96 diferentes configurações de parâmetros, as quais foram submetidas a validação cruzada utilizando o `StratifiedKFold` com `k` = 10 em 20% do conjunto de treinamento. Para o parâmetro *n_neighbors*, aplicamos um range de valores ímpares entre 3 e 50 para cada *metric* a fim de obter uma variedade maior na busca pela melhor configuração. A média dos valores obtidos com o `KFold` foi calculada e guardada para posterior análise. A partir dos dados coletados, foi plotado o gráfico abaixo:
![KNN](/algorithms/plots/knn_historic_plot.png)

Foi observado que as configurações com o parâmetro *metric* setado como `euclidean` ou `minkowski` tiveram acurácias médias idênticas. As melhores configurações encontradas a partir da experimentação foram **EUC13** e **MIN13**, ambas com uma acurácia média de **0.9050** com os seguintes parâmetros:

```
EUC13 => { 'metric': 'euclidean', 'n_neighbors': 13 }
MIN13 => { 'metric': 'minkowski', 'n_neighbors': 13 }
``` 
Devido ao empate, daqui em diante a configuração **EUC13** foi escolhida para representar a melhor performance do algoritmo KNN.

#### Rede Neural

Para a base de dados em questão, foi utilizado o **MLPClassifier** para construção do modelo. As variações feitas para encontrar o melhor resultado possível foram feitas nos parâmetros *solver* e *hidden_layer_size*. Além disso também foi utilizado o *random_state* (`42`).

Sobre o *hidden_layer_sizes*, ele recebe o i-ésimo elemento que representa o número de neurônios na i-ésima camada oculta. Quanto ao parâmetro ‘solver’, foram utilizadas três tipos diferentes, são eles: {`'lbfgs'`, `'sgd'`, `'adam'`}. `'lbfgs'` é um otimizador na família de métodos quasi-Newton. Já `'sgd'` refere-se à descida do gradiente estocástico e `'adam'` refere-se a um otimizador estocástico baseado em gradiente proposto por Kingma, Diederik e Jimmy Ba. O ‘solver’ padrão, que é `'adam'`, funciona muito bem em conjuntos de dados relativamente grandes (com milhares de amostras de treinamento ou mais) em termos de tempo de treinamento e pontuação de validação. Para conjuntos de dados pequenos, no entanto, o `'lbfgs'` pode convergir mais rapidamente e ter um desempenho melhor.

Obtivemos 300 diferentes configurações de parâmetros para a rede neural, as quais foram submetidas a validação cruzada utilizando o `StratifiedKFold` com `k` = 10 em 20% do conjunto de treinamento. Para o parâmetro *hidden_layer_size*, aplicamos uma combinação de no máximo 2 camadas e no máximo 50 neurônios para cada camada por *solver*, a fim de obter uma variedade maior na busca pela melhor configuração. A média dos valores obtidos com o `KFold` foi calculada e guardada para posterior análise. A partir dos dados coletados, foi plotado o gráfico abaixo:

![MLP](/algorithms/plots/neural_networks_historic_plot.png)

Foi observado que as configurações com o parâmetro *solver* setado como `sgd` tiveram performances superioes às demais. A melhor configuração dentre essas, foi a **SGD_43_2**, com uma acurácia média de **0.9083** com os seguintes parâmetros:

```
SGD_43_2 => { 'solver': 'sgd', 'hidden_layer_sizes': (43, 2) }
```

#### Random Forest
É utilizado para construção de *Ensemble Based Systems* (EBS) com árvores de decisão. Esse classificador utiliza variações na quantidade de dados, características além de árvores de decisão com diferentes inicializações para obter uma predição com maior acurácia e mais estável.

Para esse modelo em questão, foi utilizada a função `RandomForestClassifier` para construção do modelo. As variações feitas para encontrar o melhor resultado possível foram feitas nos parâmetros ‘n_estimators’ e ‘max_features’. Além disso, também foi utilizado o ‘random_state’ (`42`).

O primeiro parâmetro citado (*n_estimators*) indica o número de árvores construídas pelo algoritmo antes de tomar uma votação ou fazer uma média de predições. Normalmente, quanto maior a quantidade de árvores, maior também é a performance e custo computacional, além de deixar as predições mais estáveis. Já o segundo parâmetro utilizado (*max_features*), diz respeito ao número máximo de características a serem utilizadas pelo Random Forest na construção de uma dada árvore, tendo os seguintes valores: [`'sqrt'`, `'log2'`, `None`].

Obtivemos 96 diferentes configurações de parâmetros para o random forest, as quais foram submetidas a validação cruzada utilizando o `StratifiedKFold` com `k` = 10 em 20% do conjunto de treinamento. Para o parâmetro *n_estimators*, utilizamos valores de 1 a 32 para cada valor de *max_features*, a fim de obter uma variedade maior na busca pela melhor configuração. A média dos valores obtidos com o `KFold` foi calculada e guardada para posterior análise. A partir dessas informações foi plotado o gráfico abaixo:

![Random Forest](/algorithms/plots/random_forest_historic_plot.png)

As melhores configurações encontrada a partir da experimentação foram **S16**, **S27** e **S29**, todas elas conseguindo a mesma acurácia média de **0.9069** com os seguintes parâmetros:

```
S16 =>  { 'n_estimators': 16, 'max_features': 'sqrt' }
S27 =>  { 'n_estimators': 27, 'max_features': 'sqrt' }
S29 =>  { 'n_estimators': 29, 'max_features': 'sqrt' }
```

A configuração **S16** foi escolhida para representar a melhor acurácia do algoritmo *random forest*, por ter o menor número de árvores (16).

#### Comitê de Redes Neurais

Um comitê de Redes Neurais é um método de aprendizado, supervisionado ou não, cujo o objetivo é aumentar a capacidade de generalização de parâmetros de estimadores MLP. Para esse experimento, selecionamos as redes que melhor performaram na busca pelos melhores parâmetros:

Variação |solver |  hidden_layer_sizes  | Acurácia média
-------- | ------| -------------------- | ------------ |
SGD43_2	 | sgd	 | (42, 2) 		|  0.9082
SGD6_1 	 | sgd	 | (6, 1) 		|  0.8951
SGD18_2  | sgd	 | (18, 2) 		|  0.9045
SGD36_1	 | sgd	 | (36, 1) 		|  0.8954
ADA2_1 	 | adam  | (2, 1) 		|  0.9014

As 5 redes neurais foram agrupadas utilizando o `VotingClassifier`, que é um comitê baseado no método de votação majoritária. Para buscar a melhor configuração, variamos o parâmetro *voting*, que pode assumir os valores `'hard'` e`'soft'`.

```python
estimators = [
    ('sgd43_2', sgd43_2),
    ('sgd6_1', sgd6_1),
    ('sgd18_2', sgd18_2),
    ('sgd36_1', sgd36_1),
    ('ada2_1', ada2_1),
]

ensembleSoft = VotingClassifier(estimators, voting='soft')
ensembleHard = VotingClassifier(estimators, voting='hard')
```
Dessa forma, os resultados foram:

##### Resultado comitê de redes neurais
voting |  Acurácia | 
-------- | ------ |
soft | **0.8993**
hard | **0.9064**

Como pode ser observado acima, o comitê com `voting` setado como **hard** apresentou uma acurácia média ligeiramente maior.

#### Comitê heterogêneo

Assim como visto no Comitê de Redes Neurais, o Comitê heterogêneo é um método de aprendizado, supervisionado ou não, cujo o objetivo é aumentar a capacidade de generalização de parâmetros. Porém, utilizando diferentes estimadores. Para esse experimento, selecionamos os estimadores trabalhados até então e os parâmetros que melhor perfomaram na busca por suas melhores configurações:

Name | Parâmetros  | Acurácia média
-------- | ------| -------------------- |
DecisionTree | {criterion='gini', max_depth=5, random_state=42} | 0.9048
RandomForestClassifier | {n_estimators=29, max_features='sqrt', random_state=42} | 0.9069
KNeighborsClassifier | {metric="euclidean", n_neighbors=13} | 0.9050
MLPClassifier | {solver='sgd', hidden_layer_sizes=(43, 2), random_state=42} |  0.9084

Os 4 estimadores foram agrupados utilizando o `VotingClassifier`, que é um comitê baseado no método de votação majoritária. Para buscar a melhor configuração, variamos o parâmetro *voting*, que pode assumir os valores `'hard'` e`'soft'`.

```python
    estimators = [
        ('DecisionTree', dt),
        ('RandomForestClassifier', rf),
        ('KNeighborsClassifier', knn),
        ('MLPClassifier', mlpc),
    ]

ensembleSoft = VotingClassifier(estimators, voting='soft')
ensembleHard = VotingClassifier(estimators, voting='hard')
```
Dessa forma, os resultados foram:

##### Resultado comitê heterogêno
voting |  Acurácia | 
-------- | ------ |
soft | **0.9116**
hard | **0.9090**

Como pode ser observado acima, o comitê com `voting` setado como **soft** apresentou uma acurácia média ligeiramente maior.

### Testes de Significância Estatística

O teste de significância estatística é feito para determinar se existem discrepâncias estatísticas entre diferentes conjuntos de dados. Para se realizar esse teste, foi escolhido o **Kruskal-Wallis H-test**, que é eficiente para amostras com muitos dados, e foram passados os escores do *kfold*, retornados pelo método *cross_val_score* de cada algoritmo de classificação executado anteriormente, utilizando seus melhores parâmetros obtidos.

Um teste de *Kruskal-Wallis* significante indica que ao menos uma amostra domina estocasticamente (de modo aleatório e não determinístico) uma outra amostra. 

Para verificar se a amostra possuía significância estatística, foi utilizada a medida do *p-value*, que é basicamente a chance de uma estatística observada ocorrer ao acaso. Como limite, foi utilizado um alpha de **0.05**, representando que se a chance de um dado observado não conseguir rejeitar a hipótese nula for maior que **5%**, ele é estatisticamente insignificante, ao passo que quanto menor for o valor do *p-value*, maior é sua significância estatística. Como resultado do teste, foi possível se obter **16.315** como a estatística de Kruskal-Wallis H e **p=0.006**, sendo assim, já que o p é menor do que o alpha (0.05), o H0 é rejeitado, ou seja, os dados têm origem de **diferentes** distribuições.

### Comparação de performance e resultado final
A partir das buscas de melhores combinações de parâmetros para os algoritmos escolhidos acima, foi possível reunir todos os classificadores em sua melhor fase e comparar o resultados de aprendizado entre eles.

Abaixo encontra-se o plot da comparação de performance dos classificadores.
![Algorithms Performance Comparation](/performance/plots/performance_box_plot.png)

Executando todos os algoritmos em suas melhores configurações com o conjunto de teste, tivemos o seguinte resultado geral:

Classificador | Acurácia
--- | ---
Árvore de Decisão | 0.914
Random Forest | 0.913
Rede Neural | 0.913
KNN | 0.908
Comitê de Redes Neurais | 0.915
Comitê Heterogêneo | 0.917

Assim, concluímos que o melhor classificador foi o **Comitê Heterogêneo**. Utilizamos o `classification_report` e `confusion_matrix` para verificar outros dados sobre o comitê heterogêneo além da acurácia média. Abaixo encontram-se o relatório de classificação do algoritmo vencedor:

``` 
# Relatório de classificação
              precision    recall  f1-score   support

           0       0.94      0.97      0.95      9144
           1       0.68      0.50      0.57      1153

    accuracy                           0.92     10297
   macro avg       0.81      0.73      0.76     10297
weighted avg       0.91      0.92      0.91     10297

# Matriz de confusão
    0     1
0 [8876  268]
1 [ 582  571]]
```

Com o relatório de classificação do comitê heterogêneo, foi possível observar que esse classificador teve mais êxito em classificar clientes na classe `0` (não) do que em `1` (sim), com base nos valores de `precision` e `recall` e na matriz de confusão.

