# Tipos de aprendizagem

Nem todo problema de aprendizagem tem a mesma forma. Antes de escolher um
algoritmo, precisamos reconhecer que **tipo** de problema temos em mãos — isso
determina quais métodos fazem sentido e como avaliar o resultado.

## Motivação

Escolher um algoritmo antes de entender o tipo de problema é como escolher o
remédio antes do diagnóstico. Um mesmo conjunto de dados — digamos, medidas de
pacientes — pode alimentar problemas completamente diferentes: prever um valor
numérico, atribuir um rótulo, descobrir subgrupos ocultos. Cada um pertence a uma
família distinta, com métricas e algoritmos próprios.

A grande divisão é pela presença ou ausência de **rótulos** ($y$) nos dados de
treino.

## Aprendizagem supervisionada

Há um "professor": cada exemplo de treino vem acompanhado da resposta correta
$y$. O modelo aprende a mapear $\mathbf{x} \mapsto y$. Divide-se em dois casos
conforme a natureza de $y$:

- **Regressão** — $y$ é **numérico contínuo**. Ex.: prever a carga viral, a
  pressão arterial ou o preço de um imóvel.
- **Classificação** — $y$ é **categórico**. Ex.: benigno vs. maligno (binária),
  ou o tipo de célula entre várias (multiclasse).

$$
\text{supervisionada:} \quad \mathcal{D} = \{(\mathbf{x}_1, y_1), \dots, (\mathbf{x}_n, y_n)\}
$$

É de longe a família mais usada na prática, e cobre boa parte deste curso
(capítulos 2 a 4).

## Aprendizagem não supervisionada

Não há rótulos: temos apenas os $\mathbf{x}$ e queremos **descobrir estrutura**.
As duas tarefas principais são:

- **Clustering** — agrupar exemplos parecidos. Ex.: encontrar subtipos de um
  tumor a partir de perfis de expressão gênica, sem saber de antemão quantos
  subtipos existem.
- **Redução de dimensionalidade** — comprimir muitas variáveis em poucas,
  preservando o essencial. Ex.: resumir 20.000 genes em 2 eixos para visualizar.

$$
\text{não supervisionada:} \quad \mathcal{D} = \{\mathbf{x}_1, \dots, \mathbf{x}_n\}
$$

Como não há resposta certa, a avaliação é mais sutil — não dá para simplesmente
"contar acertos". É o tema do capítulo 5.

## Aprendizagem por reforço

Um **agente** aprende por tentativa e erro interagindo com um ambiente, guiado
por **recompensas**. Não há um conjunto fixo de exemplos rotulados; o agente
gera sua própria experiência e busca a sequência de ações que maximiza a
recompensa acumulada. É o paradigma por trás de agentes que jogam xadrez ou
controlam robôs. Foge do escopo deste curso introdutório, mas é bom saber que
existe.

## Um mapa das famílias

| Tipo | Tem rótulo $y$? | Tarefa típica | Exemplo biomédico |
| --- | --- | --- | --- |
| Regressão | Sim (numérico) | prever valor | estimar idade a partir de metilação do DNA |
| Classificação | Sim (categoria) | prever rótulo | diagnóstico benigno/maligno |
| Clustering | Não | agrupar | descobrir subtipos de doença |
| Redução de dim. | Não | comprimir/visualizar | mapa 2D de single-cell RNA-seq |
| Reforço | Recompensa | decidir ações | protocolo de dosagem adaptativa |

## Variações úteis na prática

O mundo real raramente é tão limpo. Alguns cenários intermediários aparecem o
tempo todo:

- **Semi-supervisionada** — poucos exemplos rotulados e muitos não rotulados.
  Comum em medicina, onde rotular exige um especialista caro.
- **Auto-supervisionada** — o rótulo é criado a partir dos próprios dados (ex.:
  prever a parte oculta de uma sequência). É a base dos grandes modelos de
  linguagem.

Essas ideias são extensões das três famílias principais, não categorias
separadas.

## Como decidir

Um roteiro rápido para classificar seu problema:

1. Você tem uma variável-alvo $y$ que quer prever? Se **não**, é não
   supervisionado (clustering ou redução de dimensionalidade).
2. Se **sim**, $y$ é um número contínuo? → **regressão**. É uma categoria? →
   **classificação**.
3. O objetivo é tomar decisões sequenciais que afetam o ambiente? → reforço.

## No notebook

O notebook `02_tipos_aprendizagem.ipynb` mostra, com o mesmo conjunto de dados,
como formular um problema de regressão, um de classificação e um de clustering —
deixando claro que o **tipo** vem da pergunta, não dos dados em si.

## Exercícios

**1.** Classifique cada problema como regressão, classificação ou clustering:
(a) prever o número de dias de internação; (b) identificar se um e-mail é spam;
(c) segmentar clientes de um plano de saúde em perfis de uso.

<details><summary>Ver solução</summary>

(a) **Regressão** — o alvo (número de dias) é contínuo.
(b) **Classificação** — o alvo é categórico (spam / não spam).
(c) **Clustering** — não há rótulo; queremos descobrir grupos.

</details>

**2.** Um pesquisador tem 10.000 imagens de pele, mas só 200 foram avaliadas por
um dermatologista. Que paradigma pode aproveitar as 9.800 imagens não rotuladas?

<details><summary>Ver solução</summary>

**Aprendizagem semi-supervisionada** (ou auto-supervisionada). Ela combina os
poucos exemplos rotulados com a grande massa de dados não rotulados — por
exemplo, aprendendo uma boa representação das imagens sem rótulo e depois
ajustando um classificador com os 200 exemplos rotulados.

</details>

## Referências

- James, G. et al. (2021). *An Introduction to Statistical Learning*, cap. 2.
- Géron, A. (2019). *Hands-On Machine Learning*, cap. 1.
- Sutton, R. & Barto, A. (2018). *Reinforcement Learning: An Introduction*.
