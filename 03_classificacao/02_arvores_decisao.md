# Árvores de decisão

Uma árvore de decisão classifica fazendo uma sequência de perguntas simples do
tipo *"a característica $x_j$ é maior que um limiar $t$?"*, partindo o espaço em
regiões retangulares. O resultado se lê como um **fluxograma clínico** — o modelo
mais interpretável do curso, que qualquer especialista consegue inspecionar.

## Como a árvore decide onde cortar

A cada nó, a árvore escolhe a divisão (a característica e o limiar) que deixa os
grupos resultantes o mais **puros** possível — idealmente, cada lado só com uma
classe. A pureza é medida pela **impureza de Gini**:

$$
G = 1 - \sum_{c=1}^{C} p_c^2
$$

Onde:

- $C$ é o **número de classes**;
- $p_c$ é a **proporção de exemplos da classe $c$** no nó;
- $G = 0$ significa nó **puro** (uma só classe); $G$ é máximo quando as classes
  estão igualmente misturadas.

A alternativa clássica é a **entropia**, $H = -\sum_c p_c \log_2 p_c$, com a mesma
ideia: medir a "bagunça" de classes no nó. A árvore testa todos os cortes
possíveis e fica com o que **mais reduz** a impureza (o maior *ganho*), e repete
recursivamente em cada lado.

O widget abaixo mostra o mecanismo com uma árvore rasa feita à mão: mova dois
limiares (um vertical, um horizontal) que particionam o plano em quatro regiões,
cada uma pintada pela classe majoritária. O cartão mostra a impureza de Gini e a
acurácia — tente encontrar os cortes que melhor separam as classes.

## Profundidade: o botão do overfitting

A **profundidade** da árvore controla sua complexidade:

- rasa demais → não captura a estrutura (subajuste);
- profunda demais → cria uma folha para quase cada exemplo, memorizando o ruído
  (sobreajuste). Uma árvore sem limite de profundidade chega a **100% de acerto no
  treino** e generaliza mal.

Controla-se isso com hiperparâmetros como `max_depth` (profundidade máxima) e
`min_samples_leaf` (mínimo de exemplos por folha), ou **podando** a árvore depois
de crescida. Justamente porque uma árvore isolada tende a overfittar, elas brilham
mesmo é **em conjunto** — a motivação direta dos ensembles do próximo capítulo.

## No notebook

O notebook `02_arvores_decisao.ipynb` treina uma `DecisionTreeClassifier` no Iris,
**desenha a árvore** aprendida (as perguntas em cada nó), mostra a curva de
acurácia de treino e de validação em função de `max_depth` — o U do overfitting —
e visualiza a fronteira de decisão retangular com Plotly.

## Exercícios

**1.** Calcule a impureza de Gini de um nó com 8 exemplos da classe A e 2 da
classe B.

<details><summary>Ver solução</summary>

$p_A = 0{,}8$, $p_B = 0{,}2$. Então
$G = 1 - (0{,}8^2 + 0{,}2^2) = 1 - (0{,}64 + 0{,}04) = 0{,}32$.
Um nó bem impuro (50/50) daria $G = 0{,}5$; um nó puro daria $G = 0$. Este está
razoavelmente puro, mais perto de zero.

</details>

**2.** Por que uma árvore muito profunda quase sempre atinge 100% de acurácia no
treino, e por que isso não é motivo de comemoração?

<details><summary>Ver solução</summary>

Sem limite de profundidade, a árvore continua dividindo até **isolar cada exemplo**
em sua própria folha — aí acerta todos os pontos de treino por construção. Mas
essas divisões finais respondem ao ruído específico do treino, que não se repete
em dados novos; a acurácia de teste despenca. É overfitting clássico: memorizou,
não generalizou.

</details>

## Referências

- Breiman, L., Friedman, J., Olshen, R. & Stone, C. (1984). *Classification and Regression Trees* (CART). Wadsworth — a referência fundadora.
- Quinlan, J. R. (1986). *Induction of Decision Trees*. Machine Learning, 1, 81–106 (algoritmo ID3).
- James, G., Witten, D., Hastie, T., Tibshirani, R. & Taylor, J. (2023). *An Introduction to Statistical Learning with Applications in Python*, cap. 8. Livro aberto: https://www.statlearning.com/
