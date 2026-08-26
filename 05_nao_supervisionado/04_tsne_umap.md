# t-SNE e UMAP

A PCA é **linear**: ela projeta os dados em um plano. Mas muita estrutura biológica
é curva, enovelada — e uma projeção linear a achata. O **t-SNE** e o **UMAP** são
técnicas **não lineares** feitas para um objetivo específico: **visualizar** dados
de altíssima dimensão em 2D, preservando quem está perto de quem.

## A ideia: preservar a vizinhança local

Ambos partem do mesmo princípio: se dois pontos são **vizinhos** no espaço original
(de milhares de dimensões), eles devem continuar **vizinhos** no mapa 2D. Em vez de
tentar preservar todas as distâncias (impossível ao espremer mil dimensões em duas),
eles preservam a **estrutura local**.

- O **t-SNE** converte distâncias em **probabilidades de vizinhança** e procura um
  arranjo 2D cujas probabilidades sejam as mais parecidas possíveis com as
  originais. Seu hiperparâmetro-chave é a **perplexidade** — grosso modo, quantos
  vizinhos cada ponto "considera" (valores típicos: 5 a 50).
- O **UMAP** parte de fundamentos geométricos diferentes, costuma ser **mais rápido**
  e preservar um pouco melhor a estrutura mais global. Seu parâmetro análogo é o
  `n_neighbors`.

Nos dois, o número de vizinhos regula o equilíbrio entre enxergar **detalhe local**
(poucos vizinhos) e **estrutura ampla** (muitos vizinhos).

## As armadilhas de interpretação (leia com atenção)

Esses mapas são lindos e reveladores, mas fáceis de superinterpretar. Três regras:

- **O tamanho dos grupos não significa nada.** t-SNE e UMAP expandem grupos densos e
  encolhem os esparsos; a área de um aglomerado no mapa não reflete sua dispersão
  real.
- **As distâncias entre grupos enganam.** A técnica preserva vizinhança **local**,
  não distâncias globais. Dois grupos afastados no mapa não são necessariamente mais
  diferentes que dois grupos próximos.
- **O resultado muda entre execuções** e com os hiperparâmetros. Vale rodar com
  algumas perplexidades/sementes e confiar no que é **estável**, não em um mapa
  único.

Em resumo: use t-SNE e UMAP para **gerar hipóteses** sobre agrupamentos, não para
medir distâncias ou tamanhos. A quantificação fica para outros métodos.

## No notebook

O notebook `04_tsne_umap.ipynb` usa o conjunto de **dígitos manuscritos** (64
dimensões) para comparar, lado a lado com Plotly, a projeção **linear da PCA** com a
do **t-SNE** — vendo os dígitos se separarem em ilhas que a PCA mistura. O UMAP entra
como célula **opcional** (se a biblioteca estiver instalada), e variamos a
perplexidade para mostrar sua influência.

## Exercícios

**1.** Em um mapa t-SNE de células, um grupo aparece com o **dobro** da área de
outro. Um colega conclui que esse tipo celular é "mais variável". Ele está certo?

<details><summary>Ver solução</summary>

**Não.** O tamanho de um grupo em t-SNE (e UMAP) **não** corresponde à sua dispersão
real: a técnica expande regiões densas e comprime as esparsas para caber no plano. A
área no mapa é um artefato do método, não uma medida de variabilidade. Para comparar
dispersão, é preciso quantificar no espaço original (por exemplo, a variância dentro
de cada grupo).

</details>

**2.** Por que a PCA, mesmo sendo mais simples, ainda é útil **antes** de aplicar
t-SNE ou UMAP em dados com milhares de dimensões?

<details><summary>Ver solução</summary>

Rodar t-SNE/UMAP diretamente em milhares de dimensões é lento e sensível a ruído. É
prática comum reduzir primeiro com **PCA** (digamos, para 30–50 componentes),
removendo ruído e acelerando muito o cálculo, e só então aplicar t-SNE/UMAP sobre
essa representação já comprimida. A PCA faz o trabalho pesado de compressão linear; o
t-SNE/UMAP cuida do arranjo local final.

</details>

## Referências

- van der Maaten, L. & Hinton, G. (2008). *Visualizing Data using t-SNE*. Journal of Machine Learning Research, 9, 2579–2605.
- McInnes, L., Healy, J. & Melville, J. (2018). *UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction*. arXiv:1802.03426.
- Wattenberg, M., Viégas, F. & Johnson, I. (2016). *How to Use t-SNE Effectively*. Distill — sobre as armadilhas de interpretação.
