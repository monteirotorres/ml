# Exercícios — Validação e seleção de modelos

Exercícios sobre validação cruzada, ajuste de hiperparâmetros, curvas de
aprendizado e — o mais importante — **evitar vazamento de dados**. Cada solução no
notebook `03_validacao.ipynb` destaca um erro comum de metodologia e como
corrigi-lo. Esta é a parte menos glamourosa e mais decisiva da prática de ML.

## Exercício 1 — O vazamento da seleção de variáveis

Com muitas variáveis e poucas informativas, alguém selecionou as $k$ "melhores"
usando o conjunto **inteiro** antes de validar. Explique por que isso vaza
informação e mostre a diferença ao fazer a seleção **dentro** da validação cruzada,
com `Pipeline`.

<details><summary>Ver resposta</summary>

Escolher as variáveis com treino e teste juntos deixa a seleção "ver" o teste: entre
1000 variáveis, algumas parecem boas **por acaso** no conjunto inteiro, inclusive nas
dobras de teste. O modelo é então avaliado com uma vantagem que não teria em
produção, e a estimativa fica **inflada** (no notebook, ~0,81 contra ~0,66 corretos).
O certo é encapsular o `SelectKBest` num `Pipeline`, para que a seleção use **só** o
treino de cada dobra. Vale para qualquer pré-processamento que aprende dos dados —
padronização, imputação, seleção.

</details>

## Exercício 2 — Grid search com validação cruzada

Ajuste os hiperparâmetros de uma SVM RBF ($C$ e $\gamma$) no `wine` com
`GridSearchCV`. Reporte os melhores parâmetros e a acurácia de validação cruzada —
e explique por que **não** se deve reportar a acurácia do próprio grid como
estimativa final.

<details><summary>Ver resposta</summary>

O `GridSearchCV` escolhe os hiperparâmetros que maximizam a acurácia de validação;
esse máximo é, por construção, **otimista** (escolhemos o melhor entre muitos). Para
uma estimativa honesta do desempenho final, avalia-se em um conjunto de teste
**separado** desde o início, ou usa-se validação cruzada **aninhada**.

</details>

## Exercício 3 — Curva de aprendizado

Trace a curva de aprendizado (desempenho × tamanho do treino) de um modelo no
`digits`. As curvas de treino e validação convergem? O que a distância entre elas
indica?

<details><summary>Ver resposta</summary>

Uma **lacuna grande** entre treino (alto) e validação (baixo) indica **overfitting**
(variância alta) — mais dados tendem a ajudar. Curvas que convergem para um valor
**baixo** indicam **underfitting** (viés alto) — mais dados não ajudam; é preciso um
modelo mais expressivo ou melhores atributos. A forma das curvas orienta a próxima
ação.

</details>

## Exercício 4 — Estratificar importa

Num conjunto desbalanceado, compare a variância da acurácia entre `KFold` comum e
`StratifiedKFold`. Por que estratificar reduz a variância da estimativa?

<details><summary>Ver resposta</summary>

O `StratifiedKFold` mantém a **proporção das classes** em cada dobra. Sem isso, uma
dobra pode calhar com pouquíssimos exemplos da classe rara, gerando estimativas
instáveis de uma dobra para outra. Preservar a proporção torna as dobras comparáveis
e a estimativa de desempenho mais **estável** (menor variância).

</details>
