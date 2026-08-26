# Métricas de avaliação — classificação

Em classificação, "acurácia" é a primeira métrica que vem à mente — e uma das
mais enganosas. Este tópico apresenta o conjunto de métricas que realmente
descreve o desempenho de um classificador, sobretudo quando as classes são
desbalanceadas.

## Motivação

Suponha um teste para uma doença rara que atinge 1% da população. Um "modelo"
que responde **sempre "saudável"** acerta 99% das vezes. A acurácia é altíssima —
e o modelo é completamente inútil, pois nunca detecta um doente. Esse paradoxo,
comum em medicina, mostra por que precisamos de métricas mais finas.

## A matriz de confusão

Todo o vocabulário de classificação binária nasce de uma tabela $2 \times 2$ que
cruza a previsão com a realidade:

| | Previu Positivo | Previu Negativo |
| --- | --- | --- |
| **Real Positivo** | VP (verdadeiro positivo) | FN (falso negativo) |
| **Real Negativo** | FP (falso positivo) | VN (verdadeiro negativo) |

- **VP** — doente identificado como doente. ✔
- **FN** — doente classificado como saudável. ✘ (o mais perigoso em triagem)
- **FP** — saudável classificado como doente. ✘ (alarme falso)
- **VN** — saudável identificado como saudável. ✔

Quase todas as métricas são razões entre essas quatro contagens.

## As métricas fundamentais

$$
\text{Acurácia} = \frac{VP + VN}{VP + VN + FP + FN}
$$

Fração de acertos totais. Boa quando as classes são equilibradas; traiçoeira
quando não são.

$$
\text{Precisão} = \frac{VP}{VP + FP}
$$

Dos que o modelo **disse** serem positivos, quantos realmente eram. Alta precisão
= poucos alarmes falsos.

$$
\text{Recall (sensibilidade)} = \frac{VP}{VP + FN}
$$

Dos que **eram** positivos, quantos o modelo pegou. Alto recall = poucos casos
perdidos.

$$
\text{Especificidade} = \frac{VN}{VN + FP}
$$

Dos que eram negativos, quantos foram corretamente descartados.

## O compromisso precisão–recall

Precisão e recall puxam em direções opostas. Um modelo que classifica tudo como
positivo tem recall perfeito (não perde ninguém) mas precisão péssima (alarme
falso constante). O oposto também vale. O ponto de equilíbrio depende do
**custo** de cada erro:

- Triagem de câncer → priorize **recall** (não deixar passar um doente).
- Filtro de spam → priorize **precisão** (não jogar e-mail bom no lixo).

O **F1-score** condensa os dois em um número, pela média harmônica:

$$
F_1 = 2 \cdot \frac{\text{precisão} \cdot \text{recall}}{\text{precisão} + \text{recall}}
$$

A média harmônica só é alta quando **ambos** são altos — pune o desequilíbrio.

## Limiar e a curva ROC

A maioria dos classificadores produz uma **probabilidade**, não um rótulo direto.
Um **limiar** (padrão 0,5) converte a probabilidade em decisão. Baixar o limiar
aumenta o recall e derruba a precisão; subi-lo faz o contrário.

A **curva ROC** traça o recall (taxa de verdadeiros positivos) contra a taxa de
falsos positivos ($1 -$ especificidade) para todos os limiares possíveis. A área
sob a curva, a **AUC**, resume o poder discriminativo do modelo num único valor:

- **AUC = 1,0** → separação perfeita.
- **AUC = 0,5** → não melhor que o acaso.

A AUC tem a vantagem de ser **independente do limiar** e do desbalanceamento,
sendo a métrica preferida para comparar modelos em muitos contextos clínicos.

## Em Python

```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))          # precisão, recall, F1
print("AUC:", roc_auc_score(y_true, y_scores))        # y_scores = probabilidades
```

## No notebook

O notebook `07_metricas_classificacao.ipynb` treina um classificador num conjunto
**desbalanceado**, mostra como a acurácia engana, calcula a matriz de confusão e
o `classification_report`, e desenha a curva ROC variando o limiar.

## Exercícios

**1.** Um teste tem sensibilidade (recall) de 95% e especificidade de 80%. Numa
população onde a doença é rara, o que esperar da precisão?

<details><summary>Ver solução</summary>

A precisão será **baixa**. Com a doença rara, há muitos negativos reais; mesmo
uma especificidade de 80% gera um grande número absoluto de **falsos positivos**,
que dominam sobre os poucos verdadeiros positivos. Isso reduz a precisão
$VP/(VP+FP)$. É o mesmo fenômeno do valor preditivo positivo baixo em rastreio de
doenças raras.

</details>

**2.** Para um filtro de spam, você prioriza precisão ou recall? E para detectar
fraude em transações?

<details><summary>Ver solução</summary>

**Spam:** priorize **precisão** — um falso positivo (e-mail legítimo marcado como
spam) é mais custoso do que deixar passar um spam ocasional.

**Fraude:** em geral priorize **recall** — deixar passar uma fraude (falso
negativo) costuma custar mais do que investigar um alarme falso. O equilíbrio
exato depende dos custos reais de cada erro.

</details>

## Referências

- James, G. et al. (2021). *An Introduction to Statistical Learning*, cap. 4.
- Fawcett, T. (2006). *An introduction to ROC analysis*.
- Documentação do scikit-learn: *Metrics and scoring*.
