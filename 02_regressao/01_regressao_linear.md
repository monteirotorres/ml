# Regressão linear simples

A regressão linear simples é o modelo mais elementar de aprendizagem supervisionada: supomos que a variável resposta $y$ depende de um único preditor $x$ por uma reta, $\hat{y} = \theta_0 + \theta_1 x$. Apesar da simplicidade, ela introduz todas as ideias centrais do restante do curso — função de custo, ajuste por mínimos quadrados e interpretação de coeficientes.

Um exemplo clássico do domínio biomédico é prever a pressão arterial a partir da idade, ou o peso de um órgão a partir do peso corporal. Vamos estimar os parâmetros que minimizam a soma dos quadrados dos resíduos e discutir o que o coeficiente angular $\theta_1$ realmente significa.

> **Em construção.** Este tópico terá conteúdo completo, notebook interativo e
> slides em breve. Abaixo, o que o material cobrirá.

| O notebook cobrirá | Detalhe |
| --- | --- |
| Ajuste manual vs. `LinearRegression` | comparar mínimos quadrados na mão e no scikit-learn |
| Interpretação dos coeficientes | unidades, intercepto e resíduos |
