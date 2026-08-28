# Regressão logística

Apesar do nome, a regressão logística é um modelo de **classificação** — talvez o
mais importante da estatística aplicada à saúde. Ela fecha o capítulo de regressão
porque reaproveita toda a maquinaria linear que vimos, mudando apenas o que fazemos
com a combinação linear no final: em vez de prever um número qualquer, prevemos uma
**probabilidade**.

## Da reta à probabilidade

Uma combinação linear $z = \boldsymbol\theta^\top\mathbf{x}$ pode dar qualquer valor
real, de $-\infty$ a $+\infty$ — imprestável como probabilidade. A regressão
logística passa esse $z$ pela função **sigmoide** (ou logística), que espreme a
reta inteira no intervalo $(0, 1)$:

$$
p = \sigma(z) = \frac{1}{1 + e^{-z}}, \qquad z = \boldsymbol\theta^\top \mathbf{x}
$$

Onde:

- $z$ é o **logito** — a combinação linear dos preditores, igualzinha à da
  regressão múltipla;
- $\sigma(\cdot)$ é a **sigmoide**: vale $0{,}5$ quando $z = 0$, tende a $1$ quando
  $z \to +\infty$ e a $0$ quando $z \to -\infty$;
- $p = P(y = 1 \mid \mathbf{x})$ é a **probabilidade estimada** de a classe ser
  positiva (doente, por exemplo). Decidimos a classe comparando $p$ a um limiar
  (em geral $0{,}5$).

## Como se treina: verossimilhança e log-loss

Aqui não há fórmula fechada. Ajustamos $\boldsymbol\theta$ **maximizando a
verossimilhança** — equivalente a minimizar a **entropia cruzada binária**
(log-loss):

$$
J(\boldsymbol\theta) = -\frac{1}{n}\sum_{i=1}^{n} \Bigl[\, y_i \ln p_i + (1 - y_i)\ln(1 - p_i) \Bigr]
$$

Onde:

- $y_i \in \{0, 1\}$ é a **classe verdadeira** do exemplo $i$;
- $p_i$ é a **probabilidade prevista** para esse exemplo;
- o termo funciona como um "castigo por confiança errada": se o modelo diz $p$
  perto de 1 e a classe era 0, $\ln(1 - p)$ dispara para $-\infty$. Prever com
  confiança e errar é o pior dos mundos.

Esse custo é **convexo**, então o gradiente descendente (que vimos no PyTorch)
encontra o mínimo global. É exatamente esse laço que o `LogisticRegression` do
scikit-learn resolve por baixo.

## O presente para a epidemiologia: razão de chances

Reorganizando a sigmoide, o logito vira o logaritmo das **chances** (odds):

$$
\ln\!\left(\frac{p}{1-p}\right) = \theta_0 + \theta_1 x_1 + \cdots + \theta_p x_p
$$

Isso dá aos coeficientes uma leitura clínica direta: $e^{\theta_j}$ é a **razão de
chances** (odds ratio) associada a aumentar $x_j$ em uma unidade, mantidos os
demais fixos. Um $\theta_j > 0$ (razão de chances $> 1$) marca um **fator de
risco**; $\theta_j < 0$, um fator de proteção. É por isso que a regressão logística
é a espinha dorsal dos estudos de fatores de risco.

O widget abaixo mostra a sigmoide de um preditor: mova a **inclinação** (o peso
$\theta_1$) e o **limiar** de decisão e veja a fronteira separar os dois grupos de
pontos, com a acurácia atualizando ao vivo.

## No notebook

O notebook `05_regressao_logistica.ipynb` treina um classificador no conjunto
**breast cancer** do scikit-learn (diagnóstico benigno × maligno), interpreta os
coeficientes como razões de chance, desenha a **fronteira de decisão** em 2D com
Plotly e liga o desempenho às métricas de classificação (matriz de confusão, AUC)
do capítulo de Fundamentos. Fazemos o mesmo modelo também com um laço de treino
explícito em PyTorch, para ver que "logística" e "uma camada + sigmoide" são a
mesma coisa.

## Exercícios

**1.** Um coeficiente de regressão logística para "fumante" (0/1) vale
$\theta = 0{,}92$. Qual é a razão de chances e como interpretá-la?

<details><summary>Ver solução</summary>

A razão de chances é $e^{0{,}92} \approx 2{,}5$. Interpretação: mantidos os demais
fatores fixos, um fumante tem cerca de **2,5 vezes mais chances** (odds) do
desfecho do que um não fumante. Atenção: razão de *chances* não é razão de
*probabilidades* — as duas só se aproximam quando o desfecho é raro.

</details>

**2.** Por que não treinamos a regressão logística minimizando o erro quadrático,
como na regressão linear?

<details><summary>Ver solução</summary>

Com a sigmoide no meio, o erro quadrático em função de $\boldsymbol\theta$ deixa de
ser convexo — ganha mínimos locais que prendem o otimizador — e penaliza mal os
erros de probabilidade. A entropia cruzada (log-loss), derivada da verossimilhança,
é convexa e penaliza fortemente a confiança errada, o que dá um treino estável e
bem-calibrado.

</details>

## Referências

- James, G., Witten, D., Hastie, T., Tibshirani, R. & Taylor, J. (2023). *An Introduction to Statistical Learning with Applications in Python*, cap. 4. [Livro aberto](https://www.statlearning.com/)
- Hosmer, D. W., Lemeshow, S. & Sturdivant, R. X. (2013). *Applied Logistic Regression*, 3ª ed. Wiley. [DOI](https://doi.org/10.1002/9781118548387)
- Cox, D. R. (1958). *The Regression Analysis of Binary Sequences*. Journal of the Royal Statistical Society B, 20(2), 215–242. [DOI](https://doi.org/10.1111/j.2517-6161.1958.tb00292.x)
