# Máquinas de vetores de suporte (SVM)

A SVM procura o hiperplano que separa as classes com a **maior margem** possível, apoiando-se apenas nos pontos mais difíceis — os vetores de suporte. É um dos modelos mais poderosos para dados de dimensão moderada.

O truque do kernel permite fronteiras não lineares sem calcular explicitamente as coordenadas em alta dimensão. Discutiremos os hiperparâmetros $C$ e $\gamma$ e a intuição geométrica por trás da margem máxima.

> **Em construção.** Este tópico terá conteúdo completo, notebook interativo e
> slides em breve. Abaixo, o que o material cobrirá.

| O notebook cobrirá | Detalhe |
| --- | --- |
| Margem e vetores de suporte | por que só alguns pontos importam |
| Kernel RBF | fronteiras não lineares e o papel de $\gamma$ |
