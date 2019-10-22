## Projeto de Mineração de Dados: Bank Marketing Data Set

### Entrega 2: Pré-processamento dos dados

#### Equipe
* Antonio Augusto Correa Gondim Neto (aacgn)
* Eduardo Santos de Moura (esm7)
* Marcos Vinicius Holanda Borges (mvhb)
* Vinícius Giles Costa Paulino (vgcp)

### 1. Tratamento de valores ausentes
Durante a análise exploratória do conjunto de dados `Bank Marketing Data Set`, foi observado que das 21 colunas, apenas seis apresentam dados faltantes, sendo elas: *job*, *marital*, *education*, *default*, *housing* e *loan*. Todos os valores ausentes estão identificados por padrão como a string `'unknown'`. Dessa forma, executamos a leitura do arquivo `csv` levando isso em consideração:

```python
dataset = pandas.read_csv(file_name, sep=";", na_values="unknown")
```
Contando os valores ausentes de cada coluna, temos o seguinte resultado:

```python
> print(dataset.isnull().sum())
```

```
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
```
Com um alto número de valores ausentes, foi realizada sua substituição com a função `fillna` com o método `ffill`:
```python
dataset.fillna(method='ffill', inplace=True)
```
Essa função preenche os valores faltantes utilizando sempre o valor do registro anterior. Ao realizar uma comparação dos histogramas de cada coluna antes e depois de preencher os valores ausentes, comparando médias, desvios padrão, mínimos e máximos e demais dados estatísticos, foi constatado que não houve mudanças significativas nas distribuições.

### 2. Normalização
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
Para as colunas categóricas, (`job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `day of week` e `poutcome`) foi utilizado o `get_dummies`. Essa função cria uma nova coluna para cada categoria da coluna inicial de forma a binarizar seus valores. Por exemplo: a coluna `job`, contando com 11 categorias, foi transformada em 11 colunas: `job_admin.`, `job_technician`, `job_blue-collar`... cada uma admitindo `0` ou `1` como valor. O total de colunas do conjunto de dados, após a operação, foi de `59`:
```python
> print(dataset.dtypes)
```
```
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
```

