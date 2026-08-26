# Exercícios — Pipeline completo

Um projeto integrador de ponta a ponta, que amarra o curso inteiro: carregar dados
com valores faltantes e variáveis de tipos diferentes, pré-processar **sem vazar**,
treinar e comparar modelos, validar corretamente e reportar. A solução em
`05_pipeline.ipynb` usa `Pipeline` e `ColumnTransformer` do scikit-learn para
garantir que nenhuma etapa toque o conjunto de teste antes da hora.

O `ColumnTransformer` aplica transformações **diferentes** a colunas diferentes
(imputação + padronização nas numéricas; imputação + *one-hot* nas categóricas), e o
`Pipeline` encadeia tudo até o modelo — de modo que cada etapa é ajustada só no
treino de cada dobra da validação. É a forma canônica de evitar vazamento.

## Exercício 1 — Montar o ColumnTransformer

Sobre um conjunto com colunas numéricas (algumas com valores faltantes) e uma
coluna categórica, monte um `ColumnTransformer` que impute e padronize as numéricas
e impute e faça *one-hot* na categórica. Por que a imputação precisa estar **dentro**
do pipeline?

<details><summary>Ver resposta</summary>

Se a imputação (por exemplo, preencher faltantes com a média) for feita **fora**,
sobre os dados inteiros, a média usada terá visto o teste — vazamento. Dentro do
`Pipeline`/`ColumnTransformer`, a média é calculada **só no treino** de cada dobra e
aplicada ao teste, replicando o que aconteceria em produção. A estimativa fica
honesta.

</details>

## Exercício 2 — Comparar modelos com o mesmo pipeline

Encaixe três modelos (regressão logística, random forest e gradient boosting) **no
mesmo** pré-processamento e compare por validação cruzada. Qual vence, e o esforço
extra do melhor compensa?

<details><summary>Ver resposta</summary>

Trocar só o passo final do pipeline mantém a comparação **justa** (mesmo
pré-processamento, mesmas dobras). Em dados tabulares, o gradient boosting e a random
forest costumam liderar; a logística serve de linha de base. Se a diferença for
pequena, o modelo mais simples e interpretável pode ser preferível — desempenho não
é o único critério.

</details>

## Exercício 3 — Estimativa final honesta

Depois de escolher o melhor modelo e ajustar hiperparâmetros com `GridSearchCV`,
reporte o desempenho em um conjunto de **teste separado desde o início**. Por que
esse número é mais confiável que a melhor acurácia do grid?

<details><summary>Ver resposta</summary>

A acurácia do grid é o **máximo** sobre muitas combinações — otimista por seleção. O
conjunto de teste, nunca tocado durante treino nem ajuste, dá uma estimativa
**imparcial** do desempenho em dados novos. Reportar o teste separado (ou usar
validação cruzada aninhada) é o que separa um resultado confiável de um número
inflado.

</details>

## Exercício 4 — Reprodutibilidade

Torne o projeto inteiro reprodutível: semente fixa, `random_state` em tudo que é
aleatório, e o pipeline salvo com `joblib`. Por que reprodutibilidade é parte da
metodologia, não um luxo?

<details><summary>Ver resposta</summary>

Sem semente fixa, dois autores (ou o mesmo autor amanhã) obtêm números diferentes, e
não se sabe se uma mudança de desempenho veio do código ou do acaso. Fixar toda a
aleatoriedade (`SEMENTE` propagada, `random_state` nos estimadores e nas divisões) e
**salvar** o pipeline treinado garantem que o resultado possa ser **verificado e
reusado** — condição para confiar em qualquer conclusão.

</details>
