# Métricas de avaliação — regressão

Treinar um modelo é só metade do trabalho; a outra metade é medir se ele é bom.
Em problemas de regressão, isso significa quantificar o quão longe as previsões
$\hat{y}$ ficam dos valores reais $y$ — e há mais de uma forma de fazer isso.

## Motivação

"O modelo está bom?" é uma pergunta vazia sem uma métrica. Pior: métricas
diferentes recompensam comportamentos diferentes. Uma métrica pode dizer que o
modelo A é melhor e outra apontar o modelo B. Escolher a métrica certa é uma
decisão de projeto, não um detalhe técnico — ela deve refletir o que **custa** um
erro no seu problema.

## Os resíduos

Tudo parte do **resíduo** de cada exemplo — a diferença entre o real e o
previsto:

$$
e_i = y_i - \hat{y}_i
$$

As métricas são apenas maneiras diferentes de resumir esses resíduos em um único
número. A questão é como penalizar erros grandes versus pequenos.

## MAE — erro absoluto médio

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

A média do tamanho dos erros, na **mesma unidade** de $y$. Se você prevê pressão
em mmHg, o MAE está em mmHg — fácil de comunicar. Trata todos os erros de forma
proporcional e é **robusta a outliers**.

## MSE e RMSE — erro quadrático

$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2, \qquad
\text{RMSE} = \sqrt{\text{MSE}}
$$

Ao elevar ao quadrado, o MSE **pune erros grandes desproporcionalmente**: um erro
de 10 pesa 100 vezes mais que um erro de 1. Isso é desejável quando erros grandes
são especialmente ruins. O RMSE tira a raiz para voltar à unidade original de
$y$, o que o torna mais interpretável que o MSE.

| Métrica | Unidade | Sensível a outliers? | Interpretação |
| --- | --- | --- | --- |
| MAE | de $y$ | pouco | erro típico |
| RMSE | de $y$ | muito | penaliza erros grandes |
| MSE | de $y^2$ | muito | usada na otimização |
| $R^2$ | adimensional | sim | fração da variância explicada |

## $R^2$ — coeficiente de determinação

O $R^2$ compara o modelo a uma linha de base ingênua: prever sempre a média
$\bar{y}$.

$$
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
$$

- $R^2 = 1$ → previsões perfeitas.
- $R^2 = 0$ → o modelo não é melhor que chutar a média.
- $R^2 < 0$ → o modelo é **pior** que chutar a média (sim, é possível!).

Por ser adimensional, o $R^2$ permite comparar problemas em escalas diferentes,
mas esconde a magnitude real dos erros — por isso costuma-se reportá-lo **junto**
com o RMSE ou MAE.

## Qual métrica escolher?

A pergunta guia é: *quanto custa cada tipo de erro no meu problema?*

- Se todos os erros custam proporcionalmente ao seu tamanho → **MAE**.
- Se erros grandes são catastróficos (dosagem de medicamento) → **RMSE/MSE**.
- Se há **outliers** que você não quer que dominem a avaliação → **MAE**.
- Para comunicar "quanto da variação o modelo explica" → **$R^2$**.

Nunca confie em uma métrica só. Reportar RMSE **e** $R^2$ dá uma visão mais
honesta.

## Em Python

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

y_true = np.array([3.0, 5.0, 2.5, 7.0])
y_pred = np.array([2.8, 5.5, 2.0, 8.0])

print("MAE :", mean_absolute_error(y_true, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_true, y_pred)))
print("R²  :", r2_score(y_true, y_pred))
```

## No notebook

O notebook `06_metricas_regressao.ipynb` ajusta um modelo de regressão, calcula
todas as métricas e mostra o efeito de um único outlier sobre RMSE versus MAE —
deixando claro por que a escolha da métrica muda a conclusão.

## Exercícios

**1.** Um modelo de previsão de custo hospitalar tem MAE de R\$ 800 e RMSE de
R\$ 3.500. O que a diferença entre os dois sugere?

<details><summary>Ver solução</summary>

O RMSE muito maior que o MAE indica a presença de **alguns erros grandes**
(outliers): como o RMSE eleva os erros ao quadrado, poucos erros enormes o
inflam bastante, enquanto o MAE, que só toma o valor absoluto, permanece baixo.
Vale investigar esses casos extremos.

</details>

**2.** É possível um modelo ter $R^2$ negativo? O que isso significa?

<details><summary>Ver solução</summary>

Sim. $R^2 < 0$ significa que o modelo prevê **pior** do que simplesmente chutar a
média $\bar{y}$ para todos os exemplos. Costuma indicar um modelo mal ajustado ou
avaliado em dados muito diferentes dos de treino.

</details>

## Referências

- James, G. et al. (2021). *An Introduction to Statistical Learning*, cap. 3.
- Documentação do scikit-learn: *Metrics and scoring*.
- Willmott, C. & Matsuura, K. (2005). *Advantages of the MAE over the RMSE*.
