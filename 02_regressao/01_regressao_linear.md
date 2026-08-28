# Regressão linear simples

A regressão linear simples é o modelo mais elementar de aprendizagem
supervisionada: supomos que a variável resposta $y$ depende de um único preditor
$x$ por uma **reta**. Apesar da simplicidade, ela introduz todas as ideias
centrais do restante do curso — função de custo, ajuste por mínimos quadrados e
interpretação de coeficientes.

Um exemplo do domínio biomédico: prever a progressão de uma doença a partir de um
único exame, ou a pressão arterial a partir da idade. Vamos estimar os parâmetros
que melhor descrevem essa relação e discutir o que cada um significa.

## O modelo

$$
\hat{y} = \theta_0 + \theta_1 \, x
$$

Onde:

- $x$ é o **preditor** (a variável de entrada — a idade, por exemplo);
- $\hat{y}$ é a **predição** do modelo (o "chapéu" indica que é uma estimativa,
  não o valor real $y$);
- $\theta_0$ é o **intercepto** — o valor previsto quando $x = 0$;
- $\theta_1$ é o **coeficiente angular** (a inclinação) — de quanto $\hat{y}$ muda
  quando $x$ aumenta em **uma unidade**. É a peça que carrega o significado do
  modelo.

## O que "melhor reta" quer dizer

Para cada exemplo $i$, o **resíduo** é a distância vertical entre o valor real e a
predição: $r_i = y_i - \hat{y}_i$. Uma reta é boa quando os resíduos são pequenos.
Medimos isso pela **função de custo** de mínimos quadrados — o **erro quadrático
médio** (MSE):

$$
J(\theta_0, \theta_1) = \frac{1}{n} \sum_{i=1}^{n} \bigl(y_i - \theta_0 - \theta_1 x_i\bigr)^2
$$

Onde:

- $n$ é o **número de exemplos**;
- $y_i$ é o **valor real** do exemplo $i$;
- o termo ao quadrado penaliza erros grandes muito mais que pequenos (um resíduo
  de 2 pesa 4; um de 3 pesa 9), e elevar ao quadrado também impede que resíduos
  positivos e negativos se cancelem.

"Ajustar o modelo" é encontrar o par $(\theta_0, \theta_1)$ que **minimiza** $J$.

## A solução fechada (mínimos quadrados)

Diferente da maioria dos modelos do curso, aqui o mínimo tem **fórmula exata**.
Derivando $J$ e igualando a zero, chega-se a:

$$
\theta_1 = \frac{\sum_{i}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i}(x_i - \bar{x})^2},
\qquad
\theta_0 = \bar{y} - \theta_1 \bar{x}
$$

Onde $\bar{x}$ e $\bar{y}$ são as **médias** de $x$ e de $y$. Repare que o
numerador de $\theta_1$ é a **covariância** entre $x$ e $y$ e o denominador é a
**variância** de $x$: a inclinação é, literalmente, o quanto $x$ e $y$ variam
juntos, normalizado pela variação de $x$. E a segunda fórmula garante que a reta
**sempre passa pelo ponto médio** $(\bar{x}, \bar{y})$.

## Experimente

O widget abaixo mostra alguns pontos fixos e uma reta que você controla com dois
sliders — o intercepto $\theta_0$ e a inclinação $\theta_1$. As linhas verticais
são os resíduos; o cartão mostra o MSE da sua reta e o MSE da reta ótima (a de
mínimos quadrados). Tente bater a reta ótima na mão e sinta por que o problema
tem uma única solução: o MSE é uma "tigela" (função convexa), sem mínimos locais
para enganar.

## No notebook

O notebook `01_regressao_linear.ipynb` faz o ajuste de duas maneiras e mostra que
dão o mesmo resultado:

- **na mão**, aplicando as fórmulas fechadas acima com NumPy explícito;
- **com o scikit-learn**, via `LinearRegression`.

Depois interpretamos o coeficiente $\theta_1$ nas unidades do problema e
inspecionamos o gráfico de resíduos — a ferramenta diagnóstica que revela quando
a suposição de linearidade não se sustenta.

## Exercícios

**1.** Se $\theta_1 = 1{,}8$ mmHg/ano em um modelo que prevê pressão arterial a
partir da idade, o que isso significa em palavras? E o que seria $\theta_0$?

<details><summary>Ver solução</summary>

- $\theta_1 = 1{,}8$ mmHg/ano: a cada ano a mais de idade, o modelo prevê um
  aumento **médio** de 1,8 mmHg na pressão. É uma associação no modelo, não uma
  relação causal comprovada.
- $\theta_0$ é a pressão prevista para idade $= 0$ — em geral uma **extrapolação**
  sem sentido físico (recém-nascidos não estão na faixa dos dados). O intercepto
  costuma ser só o que ancora a reta na altura certa, não uma quantidade
  interpretável por si só.

</details>

**2.** Por que elevamos os resíduos ao quadrado em vez de somar seus valores
absolutos?

<details><summary>Ver solução</summary>

Duas razões. (1) Ao quadrado, o custo é **derivável** em todo ponto e leva à
solução fechada acima; o valor absoluto tem um "bico" em zero. (2) O quadrado
penaliza fortemente resíduos grandes, tornando o ajuste sensível a desvios
grandes. (Somar valores absolutos também é um método legítimo — a *regressão de
mínimos desvios absolutos* —, mais robusta a outliers, porém sem fórmula fechada.)

</details>

## Referências

- James, G., Witten, D., Hastie, T., Tibshirani, R. & Taylor, J. (2023). *An Introduction to Statistical Learning with Applications in Python*, cap. 3. [Livro aberto](https://www.statlearning.com/)
- Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*, cap. 3. [DOI](https://doi.org/10.1007/978-0-387-84858-7)
- Galton, F. (1886). *Regression Towards Mediocrity in Hereditary Stature*. Journal of the Anthropological Institute — o trabalho que deu nome à "regressão". [DOI](https://doi.org/10.2307/2841583)
