# k-Vizinhos mais próximos (k-NN)

O k-NN é o classificador mais intuitivo que existe: para prever a classe de um
ponto novo, olhamos seus $k$ vizinhos mais próximos entre os dados de treino e
deixamos que eles **votem**. Não há treino propriamente dito — o modelo apenas
memoriza os exemplos e adia todo o trabalho para a hora da predição. Por isso é
chamado de aprendizagem **preguiçosa** (lazy learning).

## A regra, em uma frase

Para classificar um ponto $\mathbf{x}$:

1. calcule a distância de $\mathbf{x}$ a **todos** os pontos de treino;
2. selecione os $k$ mais próximos;
3. atribua a $\mathbf{x}$ a classe **majoritária** entre esses $k$.

A distância padrão é a **euclidiana**:

$$
d(\mathbf{x}, \mathbf{x}') = \sqrt{\sum_{j=1}^{p} (x_j - x'_j)^2}
$$

Onde:

- $\mathbf{x}$ e $\mathbf{x}'$ são dois pontos (o novo e um de treino);
- $x_j$ é o valor da **$j$-ésima característica**;
- $p$ é o **número de características**;
- a soma percorre todas as dimensões — cada característica contribui com o
  quadrado da diferença.

## O papel do $k$

O único hiperparâmetro importante é $k$, e ele controla o compromisso
viés–variância de forma muito visual:

- $k = 1$ → a fronteira de decisão gruda em cada ponto, criando ilhas ao redor de
  exemplos isolados (inclusive ruído). Variância alta, propenso a overfitting.
- $k$ grande → a votação envolve muitos vizinhos, a fronteira fica suave e o modelo
  ignora detalhes locais. Viés alto; no extremo $k = n$, prevê sempre a classe mais
  comum.

O widget abaixo deixa isso concreto: mova o slider de $k$ e veja a fronteira sair
de recortada (baixo $k$) a suave (alto $k$), sobre dois grupos de pontos.

## Por que padronizar é obrigatório

A distância euclidiana soma diferenças ao quadrado **nas unidades de cada
característica**. Se uma variável está em milhares (colesterol) e outra entre 0 e 1
(uma proporção), a primeira domina a distância e a segunda é ignorada. A solução é
**padronizar** todas as características (média 0, desvio 1) antes de medir
distâncias. No k-NN isso não é um detalhe — é o que decide quais variáveis pesam.

## A maldição da dimensionalidade

Em muitas dimensões, algo contraintuitivo acontece: **todos os pontos ficam
aproximadamente à mesma distância uns dos outros**. O conceito de "vizinho
próximo" perde sentido, e o k-NN degrada. É a **maldição da dimensionalidade**, e
é uma das razões para reduzir dimensões (PCA, capítulo 5) antes de aplicar métodos
baseados em distância.

## No notebook

O notebook `01_knn.ipynb` classifica o conjunto **Iris** com `KNeighborsClassifier`
dentro de um `Pipeline` com padronização, varre vários valores de $k$ medindo a
acurácia de validação (a curva que revela o $k$ ótimo) e desenha a fronteira de
decisão em 2D com Plotly para dois valores de $k$ contrastantes.

## Exercícios

**1.** Por que é comum escolher um $k$ **ímpar** em problemas de duas classes?

<details><summary>Ver solução</summary>

Com duas classes e $k$ par, a votação pode **empatar** (por exemplo, 2 a 2 com
$k=4$), exigindo um critério de desempate arbitrário. Um $k$ ímpar garante sempre
uma maioria, eliminando o empate. (Com mais de duas classes o empate pode ocorrer
mesmo com $k$ ímpar, e aí se recorre a desempate por distância.)

</details>

**2.** Um colega treinou k-NN sem padronizar, com uma variável em mg/dL (ordem de
centenas) e outra em proporção (0 a 1). O que provavelmente aconteceu?

<details><summary>Ver solução</summary>

A variável em mg/dL, por ter valores muito maiores, **dominou** a distância
euclidiana: as diferenças nela abafaram completamente as diferenças na proporção.
Na prática, o modelo classificou quase só com base nessa variável, ignorando a
outra. Padronizar (ou normalizar) antes teria dado peso comparável às duas.

</details>

## Referências

- Cover, T. & Hart, P. (1967). *Nearest Neighbor Pattern Classification*. IEEE Transactions on Information Theory, 13(1), 21–27 — o artigo seminal. [DOI](https://doi.org/10.1109/TIT.1967.1053964)
- James, G., Witten, D., Hastie, T., Tibshirani, R. & Taylor, J. (2023). *An Introduction to Statistical Learning with Applications in Python*, cap. 2 e 4. [Livro aberto](https://www.statlearning.com/)
- Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*, cap. 13. [DOI](https://doi.org/10.1007/978-0-387-84858-7)
