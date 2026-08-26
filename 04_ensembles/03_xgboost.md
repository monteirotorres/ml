# XGBoost e LightGBM

O gradient boosting do capítulo anterior é a ideia; **XGBoost** e **LightGBM** são
as implementações que a levaram ao estado da arte em dados tabulares. Por anos,
elas foram as ferramentas por trás da maioria das soluções vencedoras em
competições com tabelas — e continuam sendo o padrão prático quando o dado não é
imagem nem texto.

## O que elas acrescentam

Sobre o gradient boosting clássico, essas bibliotecas trazem três ganhos:

- **Regularização explícita.** A função objetivo penaliza a complexidade de cada
  árvore, não só o erro:

$$
\text{Obj} = \sum_{i=1}^{n} \ell\bigl(y_i, \hat{y}_i\bigr) \;+\; \sum_{k=1}^{K} \Omega(f_k)
$$

Onde:

- $\ell(y_i, \hat{y}_i)$ é a **perda** no exemplo $i$ (o quanto a previsão erra);
- $f_k$ é a **$k$-ésima árvore** do conjunto e $K$ o total de árvores;
- $\Omega(f_k)$ é a **penalidade de complexidade** da árvore (número de folhas e
  tamanho dos pesos) — o mesmo espírito da regularização Ridge/Lasso, agora sobre
  as árvores. É o que dá ao XGBoost sua resistência a overfitting.

- **Velocidade.** Ambas constroem as árvores usando **histogramas** de valores (em
  vez de testar todos os cortes possíveis), paralelizam e usam a memória com
  esperteza. O LightGBM, em particular, cresce as árvores por folha (*leaf-wise*),
  o que costuma ser mais rápido.

- **Praticidade.** Tratam **valores faltantes** nativamente e trazem *early
  stopping* embutido.

## Os hiperparâmetros que importam

Poucos parâmetros concentram quase todo o ajuste:

| Parâmetro | O que controla |
| --- | --- |
| `n_estimators` | número de árvores (estágios de boosting) |
| `learning_rate` | a taxa $\nu$ — passo de cada árvore |
| `max_depth` | profundidade de cada árvore (complexidade) |
| `subsample` | fração das amostras usadas por árvore (aleatoriedade) |
| `colsample_bytree` | fração das colunas usadas por árvore |
| `reg_lambda` / `reg_alpha` | regularização $\ell_2$ / $\ell_1$ |

A estratégia prática recomendada: fixar uma `learning_rate` pequena (0,05–0,1),
usar *early stopping* para achar o número de árvores, e então ajustar `max_depth` e
a subamostragem. Regularizar sempre — é o que separa um bom modelo de um
sobreajustado.

## No notebook

O notebook `03_xgboost.ipynb` usa o `HistGradientBoostingClassifier` do
scikit-learn — o mesmo estilo de boosting por histogramas, disponível sem instalar
nada — para mostrar *early stopping* e o efeito da regularização, e traz uma célula
**opcional** que instala e roda o XGBoost de verdade para quem quiser comparar.

## Exercícios

**1.** Por que uma `learning_rate` menor quase sempre exige aumentar o
`n_estimators`?

<details><summary>Ver solução</summary>

Cada árvore contribui com um passo proporcional à `learning_rate`. Com passos
menores, é preciso **mais passos** (mais árvores) para percorrer a mesma distância
até um bom ajuste. Por isso os dois andam juntos: reduzir a taxa sem aumentar o
número de árvores deixa o modelo **subajustado**, por não ter avançado o
suficiente.

</details>

**2.** O termo $\Omega(f_k)$ na função objetivo penaliza árvores complexas. Que
problema isso combate, e a que técnica do capítulo de regressão ele se assemelha?

<details><summary>Ver solução</summary>

Combate o **overfitting**: sem penalizar a complexidade, o boosting tende a criar
árvores cada vez mais elaboradas que ajustam o ruído do treino. O termo $\Omega$ é
diretamente análogo à **regularização Ridge/Lasso** — adiciona ao objetivo um custo
pelo "tamanho" do modelo (folhas e pesos), empurrando para soluções mais simples e
que generalizam melhor.

</details>

## Referências

- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD, 785–794.
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS, 30.
- Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. Annals of Statistics, 29(5), 1189–1232.
