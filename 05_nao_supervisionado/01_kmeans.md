# k-means clustering

Entramos na aprendizagem **não supervisionada**: agora não há rótulos $y$ para
prever. O objetivo muda de "acertar a resposta" para "encontrar estrutura". O
k-means é o algoritmo mais usado para isso — ele agrupa os dados em $k$
conglomerados (clusters), colocando cada ponto no grupo cujo centro está mais
perto.

## O que o k-means otimiza

O k-means busca $k$ centros (chamados **centroides**) que minimizem a soma das
distâncias ao quadrado de cada ponto ao seu centro:

$$
J = \sum_{c=1}^{k} \sum_{\mathbf{x} \in C_c} \lVert \mathbf{x} - \boldsymbol\mu_c \rVert^2
$$

Onde:

- $k$ é o **número de grupos** (você escolhe);
- $C_c$ é o **conjunto de pontos** atribuídos ao grupo $c$;
- $\boldsymbol\mu_c$ é o **centroide** do grupo $c$ (a média dos pontos dele);
- $\lVert \mathbf{x} - \boldsymbol\mu_c \rVert^2$ é a **distância ao quadrado** do
  ponto ao seu centroide;
- $J$ é a **inércia** (ou WCSS, *within-cluster sum of squares*) — quanto menor,
  mais compactos os grupos.

## Como ele encontra os grupos (algoritmo de Lloyd)

Não dá para testar todas as atribuições possíveis, então o k-means alterna dois
passos simples até estabilizar:

1. **atribuição** — cada ponto vai para o centroide mais próximo;
2. **atualização** — cada centroide vira a média dos pontos que recebeu.

Repetindo, a inércia $J$ só diminui, até parar de mudar. Como o resultado depende
de **onde os centroides começam**, a inicialização inteligente **k-means++**
(usada por padrão) espalha os centros iniciais para evitar soluções ruins — e
roda-se o algoritmo várias vezes, ficando com a de menor $J$.

O widget abaixo executa o k-means ao vivo: escolha $k$, gere uma nova amostra e
veja os pontos serem coloridos pelo grupo e os centroides se acomodarem. O cartão
mostra a inércia.

## Escolhendo k

Como não há rótulo, $k$ é decisão sua. Dois guias:

- **método do cotovelo** — trace a inércia $J$ contra $k$. Ela sempre cai (mais
  grupos ajustam melhor), mas há um "cotovelo" onde a queda desacelera: acrescentar
  grupos além dali rende pouco. Esse joelho sugere um bom $k$.
- **silhueta** — mede, para cada ponto, quão mais perto ele está do próprio grupo
  do que do grupo vizinho mais próximo (varia de $-1$ a $1$). A silhueta média alta
  indica grupos bem separados.

## As limitações (importante)

O k-means supõe grupos **esféricos, de tamanho parecido**, e usa distância
euclidiana — então é **sensível à escala** (padronize antes!) e falha quando os
grupos reais têm formas alongadas ou densidades muito diferentes. Saber onde ele
quebra é tão importante quanto saber usá-lo.

## No notebook

O notebook `01_kmeans.ipynb` aplica o `KMeans` ao Iris (sem usar os rótulos),
traça a curva do **cotovelo** e a **silhueta** para escolher $k$, compara o
agrupamento encontrado com as espécies verdadeiras e mostra um caso em que o
k-means falha (grupos não esféricos), tudo com Plotly.

## Exercícios

**1.** Por que a inércia $J$ **nunca** aumenta ao passarmos de $k$ para $k+1$
grupos? Isso significa que $k+1$ é sempre melhor?

<details><summary>Ver solução</summary>

Com mais centroides, cada ponto pode ficar **igual ou mais perto** de algum centro,
então a soma das distâncias ao quadrado só pode diminuir ou empatar — no extremo
$k = n$, cada ponto vira seu próprio grupo e $J = 0$. Mas isso **não** significa que
mais grupos sejam melhores: um $J$ menor à custa de grupos artificiais não revela
estrutura real. Por isso escolhemos $k$ pelo cotovelo/silhueta, não minimizando $J$.

</details>

**2.** Você aplicou k-means sem padronizar, com "idade" (20–80) e "renda" (milhares).
O que dominou os grupos?

<details><summary>Ver solução</summary>

A **renda**, por ter valores muito maiores, dominou a distância euclidiana: os
grupos se formaram quase só por faixa de renda, ignorando a idade. Como no k-NN, o
k-means baseia-se em distância, então **padronizar** (média 0, desvio 1) antes é
essencial para que todas as variáveis pesem de forma comparável.

</details>

## Referências

- MacQueen, J. (1967). *Some Methods for Classification and Analysis of Multivariate Observations*. Proc. 5th Berkeley Symposium.
- Lloyd, S. (1982). *Least Squares Quantization in PCM*. IEEE Transactions on Information Theory, 28(2), 129–137.
- Arthur, D. & Vassilvitskii, S. (2007). *k-means++: The Advantages of Careful Seeding*. SODA, 1027–1035.
- James, G., Witten, D., Hastie, T., Tibshirani, R. & Taylor, J. (2023). *An Introduction to Statistical Learning with Applications in Python*, cap. 12. Livro aberto: https://www.statlearning.com/
