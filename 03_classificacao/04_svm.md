# Máquinas de vetores de suporte (SVM)

A máquina de vetores de suporte (SVM) parte de uma pergunta geométrica elegante:
entre as infinitas retas que separam duas classes, qual é a **melhor**? A resposta
da SVM é a que deixa a **maior margem** — a maior faixa vazia possível entre as
classes. Essa ideia de margem máxima dá à SVM uma excelente capacidade de
generalização em dados de dimensão moderada.

## A margem máxima

Um separador linear é definido por $\boldsymbol\theta^\top\mathbf{x} + b = 0$. A
SVM escolhe $\boldsymbol\theta$ e $b$ que **maximizam a margem** $M$ — a distância
do separador ao ponto mais próximo de cada classe:

$$
M = \frac{2}{\lVert \boldsymbol\theta \rVert}
$$

Onde:

- $\boldsymbol\theta$ é o **vetor de pesos** que define a orientação do separador;
- $\lVert \boldsymbol\theta \rVert$ é a **norma** (o "tamanho") desse vetor —
  maximizar a margem equivale a **minimizar** essa norma;
- $b$ é o **deslocamento** (bias) que posiciona o separador;
- $M$ é a **largura da margem**.

O detalhe crucial: a margem é definida apenas pelos pontos que ficam na sua borda
— os **vetores de suporte**. Todos os outros pontos poderiam ser removidos sem
mudar o separador. O modelo depende só dos exemplos mais difíceis, o que o torna
compacto e robusto.

O widget abaixo torna isso tangível: gire e desloque um separador sobre duas
classes e observe a margem (a faixa até o ponto mais próximo de cada lado). Procure
a orientação que **maximiza** essa faixa — é o que a SVM faz automaticamente.

## Margem suave: o hiperparâmetro $C$

Dados reais raramente são perfeitamente separáveis. A **margem suave** permite
algumas violações (pontos dentro da margem ou do lado errado), controladas por $C$:

- $C$ **grande** → pune muito as violações; margem estreita, ajuste apertado aos
  dados (mais risco de overfitting);
- $C$ **pequeno** → tolera violações em troca de uma margem larga e mais suave
  (mais regularização).

$C$ é, portanto, o botão viés–variância da SVM.

## O truque do kernel

E quando a fronteira não é uma reta? O **truque do kernel** projeta os dados,
implicitamente, num espaço de dimensão muito mais alta onde eles **passam a ser
separáveis por um hiperplano** — sem nunca calcular essas coordenadas
explicitamente. O kernel mais usado é o **RBF** (base radial):

$$
K(\mathbf{x}, \mathbf{x}') = \exp\!\bigl(-\gamma\,\lVert \mathbf{x} - \mathbf{x}' \rVert^2\bigr)
$$

Onde $\gamma$ controla o **alcance** de cada ponto: $\gamma$ grande gera fronteiras
muito onduladas e locais (risco de overfitting); $\gamma$ pequeno, fronteiras
suaves. Junto com $C$, é o par de hiperparâmetros que se ajusta por validação
cruzada.

## No notebook

O notebook `04_svm.ipynb` treina uma SVM linear e uma com kernel RBF, visualiza
as fronteiras e destaca os vetores de suporte com Plotly, e mostra o efeito de $C$
e $\gamma$ numa varredura — inclusive o caso clássico dos dados "em círculos", onde
o kernel resolve o que nenhuma reta consegue.

## Exercícios

**1.** Se você remover um ponto que **não** é vetor de suporte e retreinar, o que
acontece com o separador?

<details><summary>Ver solução</summary>

**Nada.** O separador de margem máxima é determinado exclusivamente pelos vetores
de suporte (os pontos na borda da margem). Pontos fora da margem não participam da
definição do hiperplano, então removê-los deixa a solução idêntica. É por isso que
a SVM é "econômica": a decisão depende de poucos exemplos.

</details>

**2.** Aumentar muito $\gamma$ no kernel RBF costuma dar 100% de acerto no treino e
péssimo desempenho no teste. Por quê?

<details><summary>Ver solução</summary>

Com $\gamma$ muito grande, a influência de cada ponto fica **muito local** — cada
exemplo de treino cria uma pequena "bolha" da sua classe ao redor de si. A
fronteira envolve ponto a ponto, memorizando o treino (acerto ~100%), mas essas
bolhas não correspondem a nenhuma estrutura real, então o teste vai mal. É
overfitting controlado por $\gamma$, e se corrige reduzindo-o (com validação
cruzada).

</details>

## Referências

- Cortes, C. & Vapnik, V. (1995). *Support-Vector Networks*. Machine Learning, 20, 273–297 — o artigo seminal.
- Boser, B., Guyon, I. & Vapnik, V. (1992). *A Training Algorithm for Optimal Margin Classifiers*. COLT.
- James, G., Witten, D., Hastie, T., Tibshirani, R. & Taylor, J. (2023). *An Introduction to Statistical Learning with Applications in Python*, cap. 9. Livro aberto: https://www.statlearning.com/
