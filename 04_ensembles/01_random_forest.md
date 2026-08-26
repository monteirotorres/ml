# Bagging e Random Forests

Uma árvore de decisão sozinha é instável: muda bastante se os dados mudam um
pouco (variância alta). A ideia central deste capítulo é que **muitos modelos
instáveis, combinados, formam um modelo estável**. O *bagging* e a *random forest*
são a forma mais direta e robusta de fazer isso.

## Bagging: a média conserta a variância

*Bagging* é a sigla de **bootstrap aggregating**. A receita:

1. gere $B$ **reamostras bootstrap** do conjunto de treino (amostragem com
   reposição, cada uma do mesmo tamanho do original);
2. treine uma árvore em cada reamostra;
3. **combine** as previsões — voto majoritário (classificação) ou média
   (regressão).

Por que funciona? A média de $B$ estimativas com variância $\sigma^2$ tem variância

$$
\operatorname{Var}(\bar{f}) = \rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2
$$

Onde:

- $\sigma^2$ é a **variância de uma única árvore**;
- $B$ é o **número de árvores**;
- $\rho$ é a **correlação média entre as árvores**;
- o segundo termo encolhe conforme $B$ cresce — mais árvores reduzem a variância.
  Mas o primeiro termo, $\rho\sigma^2$, **não** depende de $B$: se as árvores forem
  muito parecidas ($\rho$ alto), a média ajuda pouco. A lição: para o conjunto
  ganhar, as árvores precisam ser **diferentes** umas das outras.

## Random forest: forçando a diferença

A **random forest** é bagging de árvores com um tempero a mais para **reduzir
$\rho$**: em cada divisão, a árvore só pode escolher entre um **subconjunto
aleatório** das características (tipicamente $\sqrt{p}$ delas, onde $p$ é o número
total de preditores). Isso impede que todas as árvores usem sempre a mesma
variável dominante, tornando-as menos correlacionadas — e, pela fórmula acima, o
conjunto fica ainda melhor.

O widget abaixo ilustra a redução de variância: cada árvore individual (linhas
finas) é um ajuste ruidoso; a média delas (linha grossa) é suave e estável. Aumente
o número de árvores e veja a média se firmar.

## Dois brindes da random forest

- **Erro out-of-bag (OOB):** cada árvore é treinada sem cerca de 1/3 dos exemplos
  (os que ficaram de fora do seu bootstrap). Esses exemplos servem de validação
  **gratuita** — dá para estimar o erro sem separar um conjunto de teste.
- **Importância das variáveis:** medindo quanto cada variável reduz a impureza (ou
  quanto o erro piora ao embaralhá-la), a floresta entrega um ranking natural de
  quais preditores importam.

Por tudo isso — robustez, pouco ajuste, resistência a overfitting — a random
forest é um dos primeiros modelos a se tentar em dados tabulares.

## No notebook

O notebook `01_random_forest.ipynb` treina uma `RandomForestClassifier` no conjunto
**breast cancer**, mostra a acurácia crescer e estabilizar com o número de árvores,
compara com uma árvore isolada, calcula a **importância das variáveis** e a exibe
com Plotly.

## Exercícios

**1.** Pela fórmula da variância, por que aumentar o número de árvores $B$ para o
infinito **não** zera a variância do conjunto?

<details><summary>Ver solução</summary>

Porque o termo $\rho\sigma^2$ **não depende de $B$**: quando $B \to \infty$, o
segundo termo $\frac{1-\rho}{B}\sigma^2$ vai a zero, mas sobra $\rho\sigma^2$. Ou
seja, a variância do conjunto é limitada por baixo pela **correlação entre as
árvores**. É exatamente por isso que a random forest injeta aleatoriedade nas
variáveis: para reduzir $\rho$ e baixar esse piso.

</details>

**2.** Por que o erro OOB pode substituir uma validação cruzada na random forest?

<details><summary>Ver solução</summary>

Cada exemplo ficou **de fora** do bootstrap de cerca de 1/3 das árvores; usando só
essas árvores para prevê-lo, temos uma previsão em um exemplo que elas **não
viram** — exatamente o que a validação faz. Agregando isso sobre todos os exemplos,
o erro OOB é uma estimativa de generalização quase de graça, obtida durante o
próprio treino, sem separar dados nem retreinar.

</details>

## Referências

- Breiman, L. (1996). *Bagging Predictors*. Machine Learning, 24, 123–140.
- Breiman, L. (2001). *Random Forests*. Machine Learning, 45, 5–32 — o artigo seminal.
- James, G., Witten, D., Hastie, T., Tibshirani, R. & Taylor, J. (2023). *An Introduction to Statistical Learning with Applications in Python*, cap. 8. Livro aberto: https://www.statlearning.com/
