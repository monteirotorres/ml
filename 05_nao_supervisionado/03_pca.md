# Análise de Componentes Principais (PCA)

Dados de alta dimensão — dezenas ou milhares de variáveis — são difíceis de
visualizar e muitas vezes cheios de redundância (variáveis que dizem quase a mesma
coisa). A **PCA** (análise de componentes principais) resolve os dois problemas de
uma vez: encontra um punhado de novas direções que capturam a maior parte da
informação e projeta os dados nelas.

## A ideia: direções de máxima variância

A PCA procura as direções ao longo das quais os dados **mais variam** — porque é aí
que está a informação. A primeira componente principal é a direção de **maior
variância**; a segunda é a de maior variância **perpendicular** à primeira; e assim
por diante.

Formalmente, essas direções são os **autovetores** da **matriz de covariância** dos
dados, e a variância capturada por cada uma é o **autovalor** correspondente:

$$
\mathbf{C}\,\mathbf{v}_j = \lambda_j\,\mathbf{v}_j
$$

Onde:

- $\mathbf{C}$ é a **matriz de covariância** dos dados (mede como as variáveis
  variam juntas);
- $\mathbf{v}_j$ é o **$j$-ésimo autovetor** — a direção da $j$-ésima componente
  principal;
- $\lambda_j$ é o **$j$-ésimo autovalor** — a **variância** dos dados ao longo dessa
  direção.

Ordenando os autovalores do maior para o menor, ficamos com as direções que mais
importam e descartamos as de variância desprezível.

## Quanta informação cada componente guarda

A **variância explicada** pela componente $j$ é sua fração do total:

$$
\text{variância explicada}_j = \frac{\lambda_j}{\sum_{i} \lambda_i}
$$

Somando as maiores, obtemos a **variância explicada acumulada**. Uma regra comum:
manter componentes suficientes para reter, digamos, 90% ou 95% da variância — assim
comprimimos os dados perdendo pouco.

O widget abaixo mostra a intuição: uma nuvem de pontos correlacionados e uma direção
de projeção que você gira. Encontre o ângulo que **maximiza** a variância dos pontos
projetados — essa é, por definição, a primeira componente principal. O cartão mostra
quanto da variância total aquela direção captura.

## Padronizar quase sempre

A PCA persegue variância, e variância depende de **unidades**. Uma variável em
milhares (renda) teria variância muito maior que uma proporção (0 a 1), e a PCA a
escolheria como primeira componente só pela escala. Por isso, salvo quando todas as
variáveis já estão na mesma unidade, **padroniza-se** antes (média 0, desvio 1) —
aí a PCA opera sobre a matriz de **correlação**, não de covariância.

## No notebook

O notebook `03_pca.ipynb` aplica a PCA (com padronização) ao Iris e ao conjunto
**breast cancer**, traça o **scree plot** e a variância explicada acumulada, projeta
os dados em 2D com Plotly — vendo as espécies/classes se separarem sem nunca terem
sido usadas — e interpreta as cargas (o peso de cada variável em cada componente).

## Exercícios

**1.** As duas primeiras componentes de um conjunto explicam 62% e 21% da variância.
Quanto se perde ao visualizar os dados apenas no plano dessas duas componentes?

<details><summary>Ver solução</summary>

As duas juntas retêm $62\% + 21\% = 83\%$ da variância total, então a projeção 2D
**perde 17%** da variância (a informação nas demais componentes). Costuma ser um
ótimo negócio para visualização: enxergamos a estrutura dominante num plano, cientes
de que uma fração menor da variação fica de fora.

</details>

**2.** Por que a PCA sem padronizar pode ser enganada por uma única variável de
escala grande?

<details><summary>Ver solução</summary>

Porque a PCA maximiza **variância**, e a variância é medida nas unidades de cada
variável. Uma variável em unidades grandes (ex.: renda em reais) tem variância
numérica enorme comparada a uma proporção, então a primeira componente aponta quase
que só na direção dela — não porque seja a mais informativa, mas porque tem a maior
escala. Padronizar coloca todas na mesma régua e evita esse artefato.

</details>

## Referências

- Pearson, K. (1901). *On Lines and Planes of Closest Fit to Systems of Points in Space*. Philosophical Magazine, 2(11), 559–572 — a origem da PCA.
- Hotelling, H. (1933). *Analysis of a Complex of Statistical Variables into Principal Components*. Journal of Educational Psychology, 24, 417–441.
- James, G., Witten, D., Hastie, T., Tibshirani, R. & Taylor, J. (2023). *An Introduction to Statistical Learning with Applications in Python*, cap. 12. Livro aberto: https://www.statlearning.com/
