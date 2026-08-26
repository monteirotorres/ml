# Stacking e combinação de modelos

Até aqui combinamos muitos modelos **do mesmo tipo** (árvores). E se combinássemos
modelos **diferentes** — uma regressão logística, uma SVM, uma floresta — para que
as forças de um cobrissem as fraquezas do outro? É a ideia da combinação de
modelos, cujo ápice é o *stacking*.

## Voting: a combinação mais simples

O **voting** (votação) apenas junta as previsões de vários modelos de base:

- **hard voting** — cada modelo dá um voto de classe; vence a maioria;
- **soft voting** — faz-se a **média das probabilidades** previstas e escolhe-se a
  classe de maior média (costuma ser melhor, por usar a confiança de cada modelo).

Funciona bem quando os modelos são **bons e diferentes**: se erram em exemplos
distintos, a média cancela erros individuais. Se todos erram nos mesmos casos,
votar não ajuda.

## Stacking: aprender a combinar

O *stacking* dá um passo além: em vez de uma regra fixa (média/maioria), ele
**aprende** a melhor forma de combinar. A arquitetura tem dois níveis:

- **modelos de base** (nível 0) — os vários modelos diferentes;
- **meta-modelo** (nível 1) — um modelo simples (em geral uma regressão) que recebe
  como entrada as **previsões dos modelos de base** e aprende a pesá-las para dar a
  resposta final.

Em símbolos, a previsão final é

$$
\hat{y} = g\bigl(f_1(\mathbf{x}),\, f_2(\mathbf{x}),\, \dots,\, f_L(\mathbf{x})\bigr)
$$

Onde:

- $f_1, \dots, f_L$ são os **$L$ modelos de base**;
- $f_\ell(\mathbf{x})$ é a **previsão** (ou probabilidade) do modelo de base $\ell$;
- $g$ é o **meta-modelo**, que aprende como misturar essas previsões.

## O perigo: vazamento, e como evitá-lo

Há uma armadilha sutil. Se o meta-modelo for treinado nas previsões que os modelos
de base fizeram **sobre os próprios dados de treino deles**, essas previsões são
otimistas demais (os modelos já viram esses exemplos), e o meta-modelo aprende com
informação vazada. A solução é usar **previsões out-of-fold**: cada modelo de base
prevê um exemplo somente quando esse exemplo ficou **de fora** da dobra em que ele
treinou — a mesma lógica da validação cruzada. O `StackingClassifier` do
scikit-learn faz isso automaticamente.

O widget abaixo mostra a intuição com soft voting: dois modelos de base com forças
diferentes: um slider mistura as duas probabilidades, e a acurácia da combinação
pode superar a de cada um isolado.

## Quando compensa

Stacking costuma render um ganho **pequeno** de desempenho a um custo **grande** de
complexidade (treinar vários modelos e mais um por cima). Vale a pena quando cada
ponto percentual importa (competições, produção madura) e os modelos de base são
genuinamente diversos. Para a maioria dos casos, uma boa random forest ou um
gradient boosting bem ajustado já entrega quase tudo, com muito menos trabalho.

## No notebook

O notebook `04_stacking.ipynb` combina três modelos de base diferentes (regressão
logística, SVM e floresta) por soft voting e por stacking (com o
`StackingClassifier`), e compara a acurácia da combinação com a de cada modelo
isolado — mostrando quando misturar realmente ajuda.

## Exercícios

**1.** Um colega treinou o stacking usando, como entrada do meta-modelo, as
previsões dos modelos de base sobre o **conjunto de treino inteiro**. A acurácia
ficou ótima no treino e péssima no teste. O que aconteceu?

<details><summary>Ver solução</summary>

**Vazamento de dados.** As previsões dos modelos de base sobre os dados em que eles
treinaram são otimistas demais (eles já viram esses exemplos, às vezes os
decoraram). O meta-modelo aprendeu a confiar nessas previsões irrealisticamente
boas, que não se repetem em dados novos — daí o colapso no teste. O correto é usar
**previsões out-of-fold**, em que cada exemplo é previsto por modelos que não o
viram.

</details>

**2.** Em que situação o soft voting entre três modelos **não** supera o melhor
modelo isolado?

<details><summary>Ver solução</summary>

Quando os modelos são muito **parecidos** e erram nos **mesmos** exemplos: a média
das probabilidades não corrige nada, porque não há diversidade de erros para
cancelar. O ganho da combinação vem justamente de os modelos serem **diferentes** e
falharem em casos distintos; sem isso, votar apenas reproduz os mesmos erros.

</details>

## Referências

- Wolpert, D. H. (1992). *Stacked Generalization*. Neural Networks, 5(2), 241–259 — o artigo seminal.
- Breiman, L. (1996). *Stacked Regressions*. Machine Learning, 24, 49–64.
- Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*, cap. 8.8.
