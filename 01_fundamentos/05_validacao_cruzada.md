# Validação cruzada

Se o erro de teste é o que importa, como estimá-lo de forma confiável antes de
colocar o modelo em produção? A validação cruzada é a resposta padrão — e uma das
ferramentas mais úteis e mal compreendidas da prática de ML.

## Motivação

Separar os dados em treino e teste uma única vez tem um problema: a estimativa do
erro depende de **quais** exemplos calharam de cair no teste. Com pouca sorte na
divisão, você pode superestimar ou subestimar o desempenho e tomar decisões
erradas — escolher o modelo pior achando que é o melhor.

A validação cruzada resolve isso **rodando a avaliação várias vezes** com
divisões diferentes e tirando a média, produzindo uma estimativa mais estável e
usando cada exemplo tanto para treinar quanto para testar (em rodadas
distintas).

## A divisão em três

Antes da mecânica, o princípio inegociável: os dados usados para **avaliar** o
modelo final não podem ter influenciado nenhuma decisão. Na prática, dividimos em
três papéis:

- **Treino** — ajusta os parâmetros $\boldsymbol\theta$.
- **Validação** — escolhe hiperparâmetros e compara modelos.
- **Teste** — usado **uma única vez**, no fim, para reportar o desempenho.

Vazar informação do teste para as etapas anteriores (data leakage) é o pecado
capital da metodologia, e superestima o desempenho de forma silenciosa.

## k-fold cross-validation

A validação cruzada em $k$ dobras (k-fold) faz melhor uso dos dados de treino
para a etapa de validação. O procedimento:

1. Divida os dados em $k$ partes (folds) de tamanho parecido.
2. Para cada rodada $i = 1, \dots, k$: treine nas outras $k-1$ partes e avalie na
   parte $i$.
3. A estimativa final é a **média** das $k$ avaliações.

$$
\text{CV}_k = \frac{1}{k} \sum_{i=1}^{k} \mathcal{L}\big(\text{modelo treinado sem o fold } i,\ \text{fold } i\big)
$$

Cada exemplo é usado exatamente uma vez para validação e $k-1$ vezes para treino.
Valores típicos são $k = 5$ ou $k = 10$.

## Explore o esquema

O widget abaixo desenha a partição para o $k$ escolhido. Mova o slider e observe:
com $k$ maior, cada fold de teste fica **menor** (mais dados para treinar em cada
rodada), mas o número de modelos a treinar cresce — mais custo computacional. É o
compromisso central na escolha de $k$.

## Variantes importantes

O k-fold simples nem sempre basta. Conheça as adaptações:

| Variante | Quando usar |
| --- | --- |
| **Stratified k-fold** | classificação — preserva a proporção das classes em cada fold |
| **Leave-One-Out (LOO)** | $k = n$; conjuntos muito pequenos |
| **Group k-fold** | quando exemplos vêm em grupos (ex.: várias amostras do mesmo paciente) |
| **TimeSeriesSplit** | dados temporais — nunca treinar com o futuro |

A stratified k-fold é o padrão em classificação: sem ela, um fold pode ficar sem
nenhum exemplo de uma classe rara.

## Cuidado com o vazamento

Toda etapa de pré-processamento que "aprende" algo dos dados — padronizar,
selecionar características, imputar faltantes — deve ser ajustada **apenas no
treino de cada fold**, nunca no conjunto todo antes de dividir. Caso contrário, a
informação do fold de teste vaza para o treino e a estimativa fica otimista
demais. A ferramenta certa para garantir isso é o `Pipeline` do scikit-learn.

```python
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

modelo = make_pipeline(StandardScaler(), LogisticRegression())
scores = cross_val_score(modelo, X, y, cv=5, scoring="accuracy")
print(f"Acurácia: {scores.mean():.3f} ± {scores.std():.3f}")
```

Como o `StandardScaler` está **dentro** do pipeline, ele é reajustado dentro de
cada fold — sem vazamento.

## No notebook

O notebook `05_validacao_cruzada.ipynb` compara uma única divisão treino/teste
com a validação cruzada em $k$ dobras no conjunto Iris, mostrando como a
estimativa de erro varia entre divisões e por que a média do CV é mais confiável.

## Exercícios

**1.** Por que a acurácia de uma única divisão 70/30 pode enganar, e como o
k-fold ajuda?

<details><summary>Ver solução</summary>

Uma única divisão dá **uma** estimativa, que depende de quais exemplos caíram no
teste — pode ser sortuda ou azarada, especialmente com poucos dados. O k-fold
avalia em $k$ divisões diferentes e tira a média, reduzindo a variância da
estimativa e usando todos os dados para validação em algum momento.

</details>

**2.** Você padronizou todo o conjunto de dados **antes** de fazer a validação
cruzada. Qual é o problema?

<details><summary>Ver solução</summary>

Há **vazamento de dados**: a média e o desvio usados na padronização foram
calculados incluindo os folds que servirão de teste. O modelo, portanto, "viu"
estatísticas do teste. O correto é padronizar dentro de cada fold — na prática,
colocando o `StandardScaler` dentro de um `Pipeline`.

</details>

## Referências

- James, G. et al. (2021). *An Introduction to Statistical Learning*, cap. 5.
- Hastie, T. et al. (2009). *The Elements of Statistical Learning*, cap. 7.
- Documentação do scikit-learn: *Cross-validation*.
