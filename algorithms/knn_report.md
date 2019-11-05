## KNN

### Busca de melhores parâmetros
No algoritmo `KNN` (*k-nearest neighbors*), os parâmetros necessários são *metric* e *n_neighbors*. O *metric* serve define a métrica de distância que será utilizada para comparar a distância entre dois pontos, podendo assumir os seguintes valores: [*manhattan*, *euclidean*, *chebyshev*, *minkowski*]. Já o *n_neighbors* representa o **k**, ou seja, a quantidade de vizinhos mais próximos para se fazer a classificação. Em todas as variações dos parâmetros foram utilizados valores ímpares para essa parâmetro, pois, caso fosse par, existe uma possibilidade de empate entre duas classes.


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

A tabela acima apresenta todas as variações de parâmetros utilizadas, nomeadas de A a L e suas acurácias médias no conjunto de treinamento do nosso conjunto de dados. A busca pelos melhores parâmetros pode ser encontrada no arquivo *knn_params.py*. Como visto na tabela, as variações D e J conseguiram os melhores desempenhos, mas apenas a variação **D** (*metric* = `'euclidean'`, *n_neighbors* = `11`) será usada para os próximos passos da implementação.