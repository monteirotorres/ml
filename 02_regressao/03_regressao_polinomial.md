# Regressão polinomial

Nem toda relação é uma reta. A regressão polinomial estende o modelo linear incluindo potências do preditor ($x^2, x^3, \dots$) como novas colunas, capturando curvaturas sem abandonar o arcabouço linear — o modelo continua linear nos parâmetros.

É o exemplo didático perfeito para o compromisso viés–variância: grau baixo demais gera underfitting; grau alto demais, overfitting. Vamos visualizar como a escolha do grau muda o ajuste e por que precisamos de dados de validação para escolhê-lo.

> **Em construção.** Este tópico terá conteúdo completo, notebook interativo e
> slides em breve. Abaixo, o que o material cobrirá.

| O notebook cobrirá | Detalhe |
| --- | --- |
| Expansão polinomial | `PolynomialFeatures` do scikit-learn |
| Grau × erro de validação | curva de seleção do grau ótimo |
