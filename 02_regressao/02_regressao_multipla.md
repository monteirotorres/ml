# Regressão linear múltipla

Quando há vários preditores, a reta vira um hiperplano: $\hat{y} = \theta_0 + \theta_1 x_1 + \cdots + \theta_p x_p = \boldsymbol\theta^\top \mathbf{x}$. A regressão múltipla permite controlar variáveis de confusão e medir o efeito de cada preditor mantendo os demais fixos — a base da inferência em estudos observacionais.

Discutiremos colinearidade (preditores correlacionados que inflam a variância das estimativas), a leitura de coeficientes parciais e como codificar variáveis categóricas com dummies. Usaremos um conjunto de dados clínicos com múltiplos fatores de risco.

> **Em construção.** Este tópico terá conteúdo completo, notebook interativo e
> slides em breve. Abaixo, o que o material cobrirá.

| O notebook cobrirá | Detalhe |
| --- | --- |
| Matriz de design $\mathbf{X}$ | montagem com dummies e intercepto |
| Colinearidade e VIF | diagnóstico de preditores redundantes |
