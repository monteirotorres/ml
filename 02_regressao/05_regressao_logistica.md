# Regressão logística

Apesar do nome, a regressão logística é um modelo de **classificação**. Ela modela a probabilidade de uma classe aplicando a função sigmoide à combinação linear dos preditores: $p = \sigma(\boldsymbol\theta^\top \mathbf{x})$, garantindo saídas entre 0 e 1.

É o modelo de referência em epidemiologia, onde os coeficientes viram razões de chance (odds ratios) interpretáveis. Vamos treinar um classificador para diagnóstico binário, interpretar os coeficientes e ligar o tema às métricas de classificação.

> **Em construção.** Este tópico terá conteúdo completo, notebook interativo e
> slides em breve. Abaixo, o que o material cobrirá.

| O notebook cobrirá | Detalhe |
| --- | --- |
| Sigmoide e odds ratio | interpretação clínica dos coeficientes |
| Fronteira de decisão | visualização em 2D |
