## Projeto de Mineração de Dados: Bank Marketing Data Set

### Relatório 5: Desempenhos

#### Equipe
* Antonio Augusto Correa Gondim Neto (aacgn)
* Eduardo Santos de Moura (esm7)
* Marcos Vinicius Holanda Borges (mvhb)
* Vinícius Giles Costa Paulino (vgcp)

### Estatísticas


Classificador          | Melhor acurácia           | Desvio padrão
---------------------- | ------------------------- | ----------------
Árvore de Decisão      | 0.951 |
KNN                    | 0.960 |
Redes Neurais          | 0.946 |
Random Forest          | 0.997 |
Comitê de Redes Neurais| 0.888 |
Comitê Heterogêneo     | 0.903 |

(gráfico de caixas dos classificadores)
A partir dos parâmetros de melhor acurácia dos classificadores explorados nos experimentos anteriores, pudemos realizar a plotagem de um gráfico de caixas para comparar a acurácia, desvio padrão e outras informações de cada classificador.


### Relatórios de Classificação

O relatório de classificação é usado para medir a qualidade das predições de um algoritmo de classificação, quantas predições são verdadeiras e quantas são falsas. Mais especificamente, verdadeiros positivos e negativos e falsos positivos e negativos são usados para prever as métricas de um relatório de classificação. Abaixo estão situados cada relatório de cada classificador, utilizando a razão 75/25 para conjunto de testes.

#### Árvore de Decisão
```
              precision    recall  f1-score   support

          0       0.82      0.78      0.80      3446
          1       0.97      0.98      0.98     27445

   accuracy                           0.96     30891
  macro avg       0.90      0.88      0.89     30891
weighted avg      0.96      0.96      0.96     30891

Acurácia no conjunto de treinamento: 0.957
Acuracia no conjunto de teste: 0.900
```
#### KNN
```
              precision    recall  f1-score   support

          0       0.72      0.48      0.57      3446
          1       0.94      0.98      0.96     27445

   accuracy                           0.92     30891
  macro avg       0.83      0.73      0.76     30891
weighted avg      0.91      0.92      0.91     30891

Acurácia de treinamento: 0.920
Acurácia de testes: 0.902
```
#### Redes Neurais
```
              precision    recall  f1-score   support

          0       0.82      0.74      0.78      3446
          1       0.97      0.98      0.97     27445

   accuracy                           0.95     30891
  macro avg       0.89      0.86      0.88     30891
weighted avg      0.95      0.95      0.95     30891

Acurácia de treinamento: 0.953
Acurácia de testes: 0.898
```
#### Random Forest
```
              precision    recall  f1-score   support

          0       0.99      0.99      0.99      3446
          1       1.00      1.00      1.00     27445

   accuracy                           1.00     30891
  macro avg       0.99      0.99      0.99     30891
weighted avg      1.00      1.00      1.00     30891

Acurácia de treinamento: 0.997
Acurácia de testes: 0.903
```

#### Comitê de Redes Neurais
```
              precision    recall  f1-score   support

          0       0.82      0.72      0.77      3446
          1       0.97      0.98      0.97     27445

   accuracy                           0.95     30891
  macro avg       0.89      0.85      0.87     30891
weighted avg      0.95      0.95      0.95     30891

Acurácia de treinamento: 0.952
Acurácia de testes: 0.907
```
#### Comitê Heterogêneo
```
              precision    recall  f1-score   support

          0       0.85      0.78      0.82      3446
          1       0.97      0.98      0.98     27445

   accuracy                           0.96     30891
  macro avg       0.91      0.88      0.90     30891
weighted avg      0.96      0.96      0.96     30891

Acurácia de treinamento: 0.960
Acurácia de testes: 0.908
```