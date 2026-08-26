# Exercícios — Regressão e métricas

Esta lista consolida a regressão: ajuste linear e polinomial, interpretação de
coeficientes, regularização e as métricas RMSE e $R^2$. Resolva primeiro no papel
ou no notebook e só então confira. O notebook `01_regressao.ipynb` traz a solução
executável de cada questão.

Duas métricas aparecem o tempo todo:

$$
\text{RMSE} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2},
\qquad
R^2 = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}
$$

Onde $y_i$ é o valor real, $\hat{y}_i$ a predição, $\bar{y}$ a média de $y$ e $n$ o
número de exemplos. O **RMSE** está nas unidades de $y$ (quanto menor, melhor); o
$R^2$ é a fração da variância explicada (1 é perfeito; 0 é prever a média).

## Exercício 1 — RMSE e R² na mão e no sklearn

Com o conjunto `diabetes`, ajuste uma regressão linear usando o preditor `bmi`.
Calcule o RMSE e o $R^2$ no conjunto de teste — primeiro com NumPy, depois
conferindo com `mean_squared_error` e `r2_score`.

<details><summary>Ver resposta</summary>

Os dois caminhos batem. O $R^2$ de um único preditor fica em torno de 0,2–0,35: o
`bmi` sozinho explica só parte da progressão da doença. O RMSE fica nas unidades do
alvo. Veja o cálculo completo no notebook.

</details>

## Exercício 2 — Escolher o grau do polinômio

Gere dados de uma função com curvatura mais ruído. Ajuste polinômios de graus 1 a
15 e escolha o grau pelo **erro de validação**, não pelo de treino. Qual grau
vence, e por quê não o de maior grau?

<details><summary>Ver resposta</summary>

O erro de treino cai sempre com o grau, mas o de validação faz um U — o grau ótimo
fica no fundo desse U (em geral entre 3 e 6 para uma senoide com ruído moderado, a
depender do sorteio). Graus maiores sobreajustam: ajustam o ruído do treino e pioram
em dados novos.

</details>

## Exercício 3 — Ridge, Lasso e esparsidade

Ajuste Ridge e Lasso ao `diabetes` (padronizados, dentro de um `Pipeline`) com um
$\alpha$ moderado. Quantos coeficientes cada um zera? O que isso diz sobre os dois
métodos?

<details><summary>Ver resposta</summary>

A **Ridge** não zera nenhum coeficiente (encolhe todos suavemente); o **Lasso**
zera vários, entregando um modelo esparso. Confirma a diferença $\ell_2$ × $\ell_1$:
Lasso faz seleção de variáveis, Ridge só regulariza.

</details>

## Exercício 4 — Interpretar um coeficiente

No modelo múltiplo do `diabetes` (preditores padronizados), o coeficiente de `bmi`
é o maior positivo. Escreva, em uma frase, o que ele significa — lembrando que os
preditores estão padronizados.

<details><summary>Ver resposta</summary>

Mantidos os demais preditores fixos, um aumento de **um desvio-padrão** no `bmi`
está associado a um aumento de (o valor do coeficiente) unidades na progressão
prevista da doença. Como os dados estão padronizados, a comparação entre coeficientes
indica importância relativa; a leitura é associativa, não causal.

</details>
