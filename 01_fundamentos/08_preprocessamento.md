# Pré-processamento de dados

Dados reais chegam sujos: em escalas incompatíveis, com valores faltando, com
categorias em texto. O pré-processamento é a etapa — muitas vezes a mais
demorada — que transforma esses dados brutos em algo que um modelo consegue
aprender bem.

## Motivação

A frase mais repetida em ciência de dados é *"garbage in, garbage out"*. Nenhum
algoritmo compensa dados mal preparados. Além disso, vários métodos têm
**suposições** sobre a forma dos dados — k-NN e SVM assumem que todas as
características estão na mesma escala; muitos modelos não aceitam valores
faltantes. Ignorar isso não gera um erro claro: gera um modelo silenciosamente
ruim.

## Escalonamento de características

Considere prever risco cardíaco a partir de idade (20–80 anos) e colesterol
(100–400 mg/dL). Métodos baseados em **distância** somam as diferenças ao
quadrado; a variável de maior amplitude domina o cálculo, e a idade se torna
irrelevante só por estar numa escala menor. A solução é colocar tudo na mesma
escala.

**Padronização (z-score)** — centra na média 0 e desvio 1:

$$
x' = \frac{x - \mu}{\sigma}
$$

**Normalização (min-max)** — comprime para o intervalo $[0, 1]$:

$$
x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}
$$

| Método | Resultado | Quando usar |
| --- | --- | --- |
| `StandardScaler` | média 0, desvio 1 | padrão; bom com outliers moderados |
| `MinMaxScaler` | intervalo [0, 1] | quando limites são conhecidos |
| `RobustScaler` | usa mediana/IQR | quando há outliers fortes |

Modelos baseados em árvore (árvores, Random Forest, boosting) são **imunes** à
escala e dispensam esse passo — mas escalonar não os prejudica.

## Valores faltantes

Faltas são a regra em dados clínicos. As estratégias vão da mais simples à mais
sofisticada:

- **Remover** linhas ou colunas — só quando as faltas são poucas e aleatórias;
  descartar dados desperdiça informação.
- **Imputação simples** — preencher com a média, a mediana (numéricas) ou a moda
  (categóricas).
- **Imputação por modelo** — usar k-NN ou regressão para prever o valor ausente
  a partir das outras variáveis.

Um detalhe frequentemente esquecido: *o fato de um valor faltar pode ser
informativo*. Às vezes vale criar uma coluna indicadora "estava ausente".

## Codificação de categorias

Como visto no tópico 3, variáveis categóricas precisam virar números:
**one-hot** para nominais, **ordinal** para as que têm ordem. O
`OneHotEncoder` do scikit-learn faz isso de forma integrada aos pipelines.

## A regra de ouro: ajustar só no treino

Todo transformador que **aprende** parâmetros dos dados — a média e o desvio do
scaler, a mediana do imputador, as categorias do encoder — deve chamar `fit`
**apenas nos dados de treino** e depois `transform` no treino e no teste. Ajustar
no conjunto inteiro vaza informação do teste e infla o desempenho estimado
(o mesmo vazamento discutido na validação cruzada).

```python
scaler.fit(X_train)          # aprende μ e σ SÓ do treino
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)   # aplica os mesmos μ, σ
```

## Pipelines: fazendo tudo certo automaticamente

A forma robusta de encadear pré-processamento e modelo é o `Pipeline`, que
garante que cada passo seja ajustado apenas no treino de cada fold. Com o
`ColumnTransformer`, aplicamos transformações diferentes a colunas numéricas e
categóricas:

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

num = Pipeline([("imp", SimpleImputer(strategy="median")),
                ("sc",  StandardScaler())])
cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                ("oh",  OneHotEncoder(handle_unknown="ignore"))])

pre = ColumnTransformer([("num", num, colunas_num),
                         ("cat", cat, colunas_cat)])

modelo = Pipeline([("pre", pre), ("clf", LogisticRegression())])
modelo.fit(X_train, y_train)     # todo o pré-processamento acontece sem vazamento
```

Esse padrão é a espinha dorsal de qualquer projeto sério de ML, e reaparece no
exercício de pipeline completo (capítulo 7).

## No notebook

O notebook `08_preprocessamento.ipynb` parte de um conjunto de dados sujo (com
escalas diferentes, faltantes e categorias em texto), monta um `ColumnTransformer`
completo e mostra o impacto da padronização no desempenho de um modelo baseado em
distância.

## Exercícios

**1.** Você vai treinar um k-NN em dados com idade (anos) e renda (milhares de
reais). Por que padronizar é obrigatório aqui?

<details><summary>Ver solução</summary>

O k-NN decide pela **distância** entre pontos. Sem padronizar, a renda (valores
na casa dos milhares) domina completamente a distância, e a idade (dezenas)
torna-se irrelevante — não porque seja menos importante, mas por estar numa
escala menor. A padronização coloca as duas na mesma escala, dando a cada uma
peso comparável.

</details>

**2.** Qual o erro em calcular a média para imputação usando todo o conjunto de
dados antes de dividir em treino e teste?

<details><summary>Ver solução</summary>

É **vazamento de dados**: a média usada para preencher os faltantes do treino
incorpora informação dos exemplos de teste. O desempenho estimado fica
otimista. O correto é ajustar o imputador (`fit`) apenas no treino — de
preferência dentro de um `Pipeline`, que faz isso automaticamente em cada fold da
validação cruzada.

</details>

## Referências

- Géron, A. (2019). *Hands-On Machine Learning*, cap. 2. [O'Reilly](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
- Kuhn, M. & Johnson, K. (2013). *Applied Predictive Modeling*, cap. 3. [DOI](https://doi.org/10.1007/978-1-4614-6849-3)
- Documentação do scikit-learn: *Preprocessing data* e *Pipelines*. [docs](https://scikit-learn.org/stable/modules/preprocessing.html)
