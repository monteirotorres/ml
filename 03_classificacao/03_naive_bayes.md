# Naive Bayes

O Naive Bayes é um classificador probabilístico que aplica o **teorema de Bayes**
com uma suposição deliberadamente ingênua: a de que os preditores são
**independentes entre si, dada a classe**. Essa hipótese quase nunca é literalmente
verdadeira, mas o modelo funciona surpreendentemente bem, treina em tempo
recorde e continua sendo um dos melhores pontos de partida para classificação de
texto.

## O teorema de Bayes, aplicado à classificação

Queremos a probabilidade de a classe ser $c$ dado o vetor de características
$\mathbf{x}$. Bayes a reescreve em termos que conseguimos estimar:

$$
P(c \mid \mathbf{x}) = \frac{P(\mathbf{x} \mid c)\, P(c)}{P(\mathbf{x})}
$$

Onde:

- $P(c \mid \mathbf{x})$ é a **probabilidade a posteriori** — o que queremos: a
  chance da classe $c$ depois de ver os dados $\mathbf{x}$;
- $P(\mathbf{x} \mid c)$ é a **verossimilhança** — quão típico é ver $\mathbf{x}$
  dentro da classe $c$;
- $P(c)$ é a **probabilidade a priori** — quão comum é a classe $c$ antes de olhar
  as características (a proporção dela nos dados);
- $P(\mathbf{x})$ é a **evidência**, um fator de normalização igual para todas as
  classes — por isso pode ser ignorado na hora de escolher a classe vencedora.

## Onde entra a "ingenuidade"

Estimar $P(\mathbf{x} \mid c)$ para um vetor inteiro de características é inviável.
A suposição *naive* de independência quebra isso num **produto** de termos
simples, um por característica:

$$
P(\mathbf{x} \mid c) = \prod_{j=1}^{p} P(x_j \mid c)
$$

Onde $P(x_j \mid c)$ é a distribuição de **uma única característica** dentro da
classe — muito fácil de estimar. A classe escolhida é a que maximiza o produto da
priori pela verossimilhança:

$$
\hat{y} = \arg\max_{c} \; P(c) \prod_{j=1}^{p} P(x_j \mid c)
$$

Na prática somam-se **logaritmos** em vez de multiplicar probabilidades minúsculas,
para evitar que o produto vá a zero por limite numérico.

## As três variantes

A forma de $P(x_j \mid c)$ define o sabor do modelo:

- **Gaussiano** — características **contínuas**; supõe que cada uma segue uma normal
  dentro da classe (estima média e desvio por classe). É o usado para dados como o
  Iris.
- **Multinomial** — **contagens** (quantas vezes cada palavra aparece); o padrão
  para classificação de texto por frequência de termos.
- **Bernoulli** — características **binárias** (a palavra apareceu ou não).

O widget abaixo ilustra o caso Gaussiano de uma característica: duas classes, cada
uma com sua curva normal. Mova as médias e veja a **probabilidade a posteriori** e
a fronteira de decisão (onde as duas curvas, ponderadas pela priori, se cruzam)
se deslocarem.

## No notebook

O notebook `03_naive_bayes.ipynb` usa o `GaussianNB` no Iris — mostrando as curvas
normais estimadas por classe — e depois um `MultinomialNB` em um problema de texto
(classificar mensagens), comparando a velocidade e a acurácia com a de um modelo
mais pesado. Tudo com probabilidade condicional explícita, sem caixa-preta.

## Exercícios

**1.** Numa triagem, 1% dos laudos são "urgentes" ($P(\text{urgente}) = 0{,}01$).
Por que a priori importa tanto quando uma classe é rara?

<details><summary>Ver solução</summary>

Porque a posteriori é proporcional a $P(c)\,P(\mathbf{x}\mid c)$: um $P(c)$ muito
baixo puxa fortemente a decisão para a classe comum, a menos que a verossimilhança
$P(\mathbf{x}\mid \text{urgente})$ seja muito maior. Ignorar a priori (assumir
classes iguais) faria o modelo superestimar a classe rara. É a mesma lógica dos
falsos positivos em testes de doenças raras.

</details>

**2.** A suposição de independência é quase sempre falsa (em texto, "vetores" e
"suporte" aparecem juntos). Por que o modelo ainda classifica bem?

<details><summary>Ver solução</summary>

Para **classificar**, não precisamos das probabilidades exatas — basta que a classe
correta receba a **maior** pontuação. Mesmo com estimativas de probabilidade
distorcidas pela independência falsa, a **ordem** entre as classes costuma se
preservar, então a decisão final acerta. O Naive Bayes é um bom *classificador*
mesmo sendo um estimador de probabilidade medíocre.

</details>

## Referências

- Maron, M. E. (1961). *Automatic Indexing: An Experimental Inquiry*. Journal of the ACM, 8(3), 404–417.
- Domingos, P. & Pazzani, M. (1997). *On the Optimality of the Simple Bayesian Classifier under Zero-One Loss*. Machine Learning, 29, 103–130.
- Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*, cap. 6.6.
