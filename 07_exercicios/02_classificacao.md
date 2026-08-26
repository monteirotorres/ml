# Exercícios — Classificação

Problemas com k-NN, árvores, Naive Bayes, SVM e regressão logística. O foco é
escolher o modelo, ler fronteiras de decisão e avaliar com matriz de confusão,
precisão, recall e AUC. As soluções executáveis estão em `02_classificacao.ipynb`.

Lembrete das métricas, a partir de verdadeiros/falsos positivos e negativos (VP,
FP, VN, FN):

$$
\text{precisão} = \frac{\text{VP}}{\text{VP} + \text{FP}},
\qquad
\text{recall} = \frac{\text{VP}}{\text{VP} + \text{FN}}
$$

Onde a **precisão** responde "dos que classifiquei como positivos, quantos eram?" e
o **recall** responde "dos positivos reais, quantos peguei?". A **AUC** resume o
desempenho em todos os limiares de decisão (0,5 = aleatório; 1 = perfeito).

## Exercício 1 — Escolher o k do k-NN

No `wine`, use validação cruzada para escolher o melhor $k$ do k-NN (com
padronização). O que acontece com a acurácia para $k$ muito pequeno e muito grande?

<details><summary>Ver resposta</summary>

A acurácia sobe, atinge um platô num $k$ intermediário e depois cai. $k$ muito
pequeno superajusta (fronteira recortada, sensível a ruído); $k$ grande demais
suaviza excessivamente, aproximando-se de prever a classe majoritária. Sem
padronizar, o resultado piora.

</details>

## Exercício 2 — Ler uma matriz de confusão

Treine uma regressão logística no `breast cancer` (com o maligno como classe
positiva) e monte a matriz de confusão. Calcule precisão e recall à mão a partir
dela e diga qual erro é mais grave num rastreio.

<details><summary>Ver resposta</summary>

Num rastreio de câncer, o **falso negativo** (deixar passar um maligno) é mais
grave que o falso positivo (um alarme que leva a exames extras). Por isso costuma-se
priorizar o **recall** da classe maligna, mesmo à custa de precisão — baixando o
limiar de decisão abaixo de 0,5.

</details>

## Exercício 3 — Ajustar o limiar de decisão

Ainda no `breast cancer`, em vez do limiar padrão 0,5, escolha um limiar que
aumente o recall dos malignos para pelo menos 0,98. O que acontece com a precisão?

<details><summary>Ver resposta</summary>

Baixar o limiar (por exemplo para 0,2–0,3) captura mais malignos (recall sobe), mas
classifica como maligno mais casos benignos (precisão cai). É o compromisso
precisão–recall: em rastreio, aceitamos mais falsos positivos para não perder
verdadeiros positivos. A curva precisão–recall mostra o trade-off inteiro.

</details>

## Exercício 4 — Linear × não linear (o valor do kernel)

Nos dados `make_circles`, compare a acurácia de uma regressão logística, de uma SVM
linear e de uma SVM com kernel RBF. Explique o resultado.

<details><summary>Ver resposta</summary>

A logística e a SVM linear falham (acurácia perto de 50–60%): não há reta que
separe um anel de seu núcleo. A **SVM RBF** acerta quase tudo, porque o kernel
projeta os dados para um espaço onde eles ficam linearmente separáveis. É a
demonstração de quando vale sair do linear.

</details>
