# Regularização: Ridge e Lasso

O tópico anterior mostrou que flexibilidade demais leva ao overfitting. A
**regularização** é a forma mais elegante de conter isso: em vez de limitar o
número de preditores, adicionamos à função de custo uma **penalidade sobre o
tamanho dos coeficientes**. O modelo passa a equilibrar dois objetivos — ajustar
os dados **e** manter os coeficientes pequenos.

## A ideia, em uma fórmula

O custo regularizado soma o erro quadrático a um termo de penalidade:

$$
J(\boldsymbol\theta) = \underbrace{\sum_{i=1}^{n}\bigl(y_i - \boldsymbol\theta^\top \mathbf{x}_i\bigr)^2}_{\text{ajuste aos dados}} \;+\; \underbrace{\alpha \, \Omega(\boldsymbol\theta)}_{\text{penalidade}}
$$

Onde:

- $\Omega(\boldsymbol\theta)$ é a **penalidade** — uma medida do tamanho dos
  coeficientes;
- $\alpha \ge 0$ é o **hiperparâmetro de regularização** — a "força" da penalidade.
  Com $\alpha = 0$ voltamos aos mínimos quadrados comuns; quanto maior $\alpha$,
  mais o modelo é empurrado a encolher os coeficientes, aceitando um ajuste pior em
  troca de mais simplicidade e estabilidade.

A escolha de $\Omega$ define os dois métodos clássicos.

## Ridge ($\ell_2$) — encolher suavemente

A regressão **Ridge** penaliza a soma dos **quadrados** dos coeficientes:

$$
\Omega_{\text{Ridge}}(\boldsymbol\theta) = \sum_{j=1}^{p} \theta_j^2
$$

Ela **encolhe** todos os coeficientes na direção de zero de forma suave, mas
raramente os zera. É especialmente boa contra **colinearidade**: onde os mínimos
quadrados dariam coeficientes gigantes e instáveis (tópico anterior), a Ridge
distribui o peso entre os preditores correlacionados e estabiliza as estimativas.

## Lasso ($\ell_1$) — encolher e selecionar

A regressão **Lasso** penaliza a soma dos **valores absolutos**:

$$
\Omega_{\text{Lasso}}(\boldsymbol\theta) = \sum_{j=1}^{p} |\theta_j|
$$

A diferença parece pequena, mas a consequência é grande: a penalidade $\ell_1$
tende a zerar **exatamente** os coeficientes dos preditores menos úteis. A Lasso
faz, portanto, **seleção automática de variáveis** — entrega um modelo esparso,
que usa só um subconjunto dos preditores. Isso é ouro em dados de alta dimensão,
como expressão gênica, onde há mais variáveis do que amostras e queremos descobrir
quais poucas importam.

> **Observação importante:** como a penalidade depende da escala dos coeficientes,
> a regularização **exige padronizar os preditores** antes (média 0, desvio 1).
> Sem isso, um preditor medido em unidades grandes seria penalizado de forma
> injusta. No notebook fazemos isso com um `Pipeline`.

O **ElasticNet** combina as duas penalidades, herdando a estabilidade da Ridge e a
esparsidade da Lasso.

## O caminho de regularização

O widget abaixo traça o **caminho de regularização**: como cada coeficiente muda
conforme $\alpha$ cresce. Mova o slider de $\alpha$ (em escala logarítmica) e
compare Ridge e Lasso — na Ridge os coeficientes encolhem juntos e suavemente; na
Lasso, um a um eles chegam a **zero** e ali ficam. O cartão conta quantos
coeficientes continuam diferentes de zero.

## No notebook

O notebook `04_regularizacao.ipynb` ajusta Ridge, Lasso e ElasticNet ao conjunto
**diabetes**, sempre dentro de um `Pipeline` com padronização, traça os caminhos
de regularização com Plotly e escolhe o $\alpha$ por **validação cruzada**
(`RidgeCV` / `LassoCV`), mostrando o valor que minimiza o erro em dados não vistos.

## Exercícios

**1.** Você tem 20 000 genes (preditores) e 200 pacientes. Quer um modelo
interpretável, que aponte um punhado de genes relevantes. Ridge ou Lasso?

<details><summary>Ver solução</summary>

**Lasso.** Ela zera a maioria dos coeficientes e devolve um modelo esparso, usando
só alguns genes — exatamente o que se quer para interpretação e para o caso
$p \gg n$ (muito mais variáveis que amostras), em que os mínimos quadrados comuns
nem têm solução única. A Ridge encolheria todos, mas manteria os 20 000 no modelo.

</details>

**2.** O que acontece com os coeficientes quando $\alpha \to \infty$? E quando
$\alpha = 0$?

<details><summary>Ver solução</summary>

- $\alpha \to \infty$: a penalidade domina e todos os coeficientes são empurrados
  a **zero** — o modelo vira uma constante (só o intercepto), prevendo a média de
  $y$. Viés máximo, variância mínima.
- $\alpha = 0$: nenhuma penalidade — recuperamos a **regressão de mínimos quadrados
  comum**, com todo o risco de overfitting que ela traz em alta dimensão.

O bom $\alpha$ fica no meio, e se escolhe por validação cruzada.

</details>

## Referências

- James, G., Witten, D., Hastie, T., Tibshirani, R. & Taylor, J. (2023). *An Introduction to Statistical Learning with Applications in Python*, cap. 6. [Livro aberto](https://www.statlearning.com/)
- Hoerl, A. E. & Kennard, R. W. (1970). *Ridge Regression: Biased Estimation for Nonorthogonal Problems*. Technometrics, 12(1), 55–67. [DOI](https://doi.org/10.1080/00401706.1970.10488634)
- Tibshirani, R. (1996). *Regression Shrinkage and Selection via the Lasso*. Journal of the Royal Statistical Society B, 58(1), 267–288. [DOI](https://doi.org/10.1111/j.2517-6161.1996.tb02080.x)
- Zou, H. & Hastie, T. (2005). *Regularization and Variable Selection via the Elastic Net*. JRSS B, 67(2), 301–320. [DOI](https://doi.org/10.1111/j.1467-9868.2005.00503.x)
