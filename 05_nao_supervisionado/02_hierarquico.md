# Clustering hierárquico

O k-means exige que você diga **quantos** grupos existem. O clustering hierárquico
não: ele constrói uma **árvore de agrupamentos** (o **dendrograma**) que revela
estrutura em todas as escalas de uma vez — de cada ponto isolado até um único grupo
com tudo. Você decide onde cortar essa árvore depois de vê-la.

## Como a árvore é construída

A versão mais comum é **aglomerativa** (de baixo para cima):

1. comece com cada ponto sendo seu próprio grupo;
2. una os **dois grupos mais próximos** em um só;
3. repita até sobrar um único grupo.

Cada união vira um "galho" do dendrograma, na altura igual à distância entre os
grupos unidos. A pergunta central é: o que significa "distância entre **grupos**"?
Isso é definido pelo **critério de ligação**:

- **simples** (single) — distância entre os **dois pontos mais próximos** dos
  grupos. Tende a formar grupos alongados, em "corrente".
- **completa** (complete) — distância entre os **dois pontos mais distantes**.
  Favorece grupos compactos.
- **média** (average) — média das distâncias entre todos os pares.
- **Ward** — une os grupos que **menos aumentam a variância interna** total. É o
  padrão na prática e costuma dar grupos equilibrados; minimiza, a cada passo,

$$
\Delta = \frac{n_a\, n_b}{n_a + n_b} \, \lVert \boldsymbol\mu_a - \boldsymbol\mu_b \rVert^2
$$

Onde $n_a, n_b$ são os **tamanhos** dos grupos unidos e $\boldsymbol\mu_a,
\boldsymbol\mu_b$ seus **centroides**: unir grupos grandes ou distantes custa caro,
então o Ward prefere fusões que mantêm os grupos coesos.

## Lendo e cortando o dendrograma

A **altura** de cada união mede o quão diferentes eram os grupos fundidos: uniões
baixas juntam pontos muito parecidos; uniões altas juntam grupos já distintos.
Para obter $k$ grupos, **corta-se** o dendrograma numa altura que atravesse $k$
galhos verticais. Uma altura de corte logo abaixo de um "salto" grande costuma ser
uma boa escolha — ali, unir mais custaria juntar grupos genuinamente diferentes.

O widget abaixo mostra um dendrograma com uma linha de corte que você desliza: veja
o número de grupos mudar conforme a altura, e como um corte abaixo de um salto
grande separa a estrutura natural.

## Onipresente em bioinformática

O dendrograma é a estrutura por trás dos **heatmaps de expressão gênica**: genes e
amostras são reordenados por proximidade e as árvores nas bordas mostram quais
agrupam com quais. É a ferramenta visual padrão para explorar dados ômicos.

## No notebook

O notebook `02_hierarquico.ipynb` aplica clustering aglomerativo ao Iris,
**desenha o dendrograma** com Plotly (usando a ligação de Ward), compara os critérios
de ligação e mostra como cortar a árvore em $k$ grupos com o
`AgglomerativeClustering`.

## Exercícios

**1.** Uma vantagem prática do clustering hierárquico sobre o k-means é não precisar
fixar $k$ de antemão. Cite uma **desvantagem** em conjuntos de dados grandes.

<details><summary>Ver solução</summary>

O **custo computacional**: construir o dendrograma aglomerativo exige comparar
todos os pares de grupos a cada passo, com custo de tempo e memória que cresce
tipicamente com $O(n^2)$ ou pior. Para dezenas ou centenas de milhares de pontos
isso fica inviável, enquanto o k-means escala bem melhor. Por isso, em dados
grandes, é comum usar k-means (ou amostrar antes do hierárquico).

</details>

**2.** Com a ligação **simples** (single), dois grupos claramente separados podem
acabar unidos por causa de poucos pontos entre eles. Por quê?

<details><summary>Ver solução</summary>

Porque a ligação simples define a distância entre grupos pelos **dois pontos mais
próximos**. Se houver uma "ponte" de pontos intermediários ligando os dois grupos,
o par mais próximo através da ponte terá distância pequena, e o algoritmo os unirá
cedo — o efeito de **encadeamento** (*chaining*). A ligação completa ou de Ward, que
olham o grupo como um todo, são mais robustas a isso.

</details>

## Referências

- Ward, J. H. (1963). *Hierarchical Grouping to Optimize an Objective Function*. Journal of the American Statistical Association, 58(301), 236–244.
- Sokal, R. & Michener, C. (1958). *A Statistical Method for Evaluating Systematic Relationships*. University of Kansas Science Bulletin.
- Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*, cap. 14.3.
