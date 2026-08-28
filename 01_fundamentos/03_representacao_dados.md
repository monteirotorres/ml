# Representação dos dados

Antes de qualquer algoritmo, os dados precisam ganhar uma forma que o modelo
entenda. A maneira como representamos os exemplos e suas características determina
metade do sucesso de um projeto — e é onde mora boa parte do trabalho real.

## Motivação

Modelos de aprendizagem de máquina não "veem" pacientes, imagens ou textos: eles
veem **números organizados em uma matriz**. Um algoritmo brilhante alimentado com
uma representação ruim produz resultados ruins. Por isso, entender a estrutura
canônica dos dados — e o vocabulário que usamos para descrevê-la — é o primeiro
passo prático.

## A matriz de dados

A representação padrão em ML é uma tabela: **linhas são exemplos, colunas são
características**.

$$
\mathbf{X} \in \mathbb{R}^{n \times p}, \qquad \mathbf{y} \in \mathbb{R}^{n}
$$

Onde:

- $n$ é o número de **exemplos** (amostras, observações, pacientes);
- $p$ é o número de **características** (features, variáveis, atributos);
- $\mathbf{X}$ é a **matriz de características** (uma linha $\mathbf{x}_i$ por
  exemplo);
- $\mathbf{y}$ é o vetor de **alvos**, presente apenas na aprendizagem
  supervisionada.

O conjunto Iris, por exemplo, tem $n = 150$ flores e $p = 4$ características
(comprimento e largura de sépala e pétala). Sua matriz é $150 \times 4$.

## Tipos de características

Cada coluna tem uma natureza que dita como deve ser tratada:

| Tipo | Descrição | Exemplo | Tratamento comum |
| --- | --- | --- | --- |
| Numérica contínua | valores reais | idade, glicose | padronização |
| Numérica discreta | contagens | nº de gestações | usar como número |
| Categórica nominal | rótulos sem ordem | tipo sanguíneo | one-hot encoding |
| Categórica ordinal | rótulos com ordem | estágio I/II/III | codificação ordinal |
| Binária | dois valores | sexo, fumante | 0/1 |

Confundir esses tipos é um erro clássico: tratar "tipo sanguíneo" (A, B, AB, O)
como o número 1, 2, 3, 4 faz o modelo acreditar que O é "maior" que A — uma
ordem que não existe.

## Codificando categorias

Como modelos exigem números, variáveis categóricas nominais precisam ser
convertidas. A técnica padrão é o **one-hot encoding**: criar uma coluna binária
para cada categoria.

```python
import pandas as pd

df = pd.DataFrame({"tipo_sanguineo": ["A", "O", "AB", "A"]})
pd.get_dummies(df, columns=["tipo_sanguineo"])
#    tipo_sanguineo_A  tipo_sanguineo_AB  tipo_sanguineo_O
# 0                 1                  0                 0
# 1                 0                  0                 1
# 2                 0                  1                 0
# 3                 1                  0                 0
```

Para variáveis **ordinais**, em que a ordem importa, usa-se uma codificação
inteira que respeita a hierarquia (I → 1, II → 2, III → 3).

## Exemplo, característica, alvo — o vocabulário

Vale fixar os sinônimos, porque livros e bibliotecas usam palavras diferentes
para a mesma coisa:

- **Exemplo** = amostra = observação = instância = uma linha de $\mathbf{X}$.
- **Característica** = feature = variável = atributo = uma coluna de $\mathbf{X}$.
- **Alvo** = target = rótulo = variável resposta = uma entrada de $\mathbf{y}$.

## O caso $p > n$

Em muitos problemas biológicos temos **mais características do que exemplos** — um
microarray pode medir 20.000 genes ($p$) em apenas 50 pacientes ($n$). Esse
regime "gordo e curto" quebra a intuição clássica: há tantos graus de liberdade
que o modelo consegue ajustar qualquer coisa, inclusive o ruído. É a **maldição
da dimensionalidade**, e a razão pela qual regularização (capítulo 2) e redução
de dimensionalidade (capítulo 5) se tornam essenciais.

## Engenharia de características

Nem sempre as colunas mais úteis vêm prontas. **Feature engineering** é o ofício
de criar características novas e mais informativas a partir das existentes: a
razão entre dois exames, o índice de massa corporal a partir de peso e altura, o
tempo desde o último evento. Uma boa característica muitas vezes vale mais do que
um algoritmo sofisticado.

## No notebook

O notebook `03_representacao_dados.ipynb` monta uma matriz de dados a partir de
um `DataFrame` do pandas, identifica os tipos de cada coluna, aplica one-hot
encoding e discute o formato $\mathbf{X}, \mathbf{y}$ esperado pelo scikit-learn.

## Exercícios

**1.** Você tem uma coluna "escolaridade" com os valores *fundamental*, *médio*,
*superior*. Como codificá-la e por quê?

<details><summary>Ver solução</summary>

É uma variável **ordinal**: existe uma ordem natural (fundamental < médio <
superior). O ideal é uma **codificação ordinal** que preserve essa ordem, por
exemplo fundamental → 0, médio → 1, superior → 2. Usar one-hot também funciona,
mas descarta a informação de ordem, que pode ser útil.

</details>

**2.** Um conjunto de dados tem $n = 40$ pacientes e $p = 5000$ genes. Que
problema esse formato antecipa e que estratégias o mitigam?

<details><summary>Ver solução</summary>

Com $p \gg n$, o modelo tem graus de liberdade demais e tende a **overfittar**,
ajustando ruído. Estratégias: **regularização** (Ridge/Lasso), **seleção de
características** (manter apenas genes informativos) e **redução de
dimensionalidade** (PCA) antes de treinar.

</details>

## Referências

- VanderPlas, J. (2016). *Python Data Science Handbook*, cap. 5. [livro aberto](https://jakevdp.github.io/PythonDataScienceHandbook/)
- Kuhn, M. & Johnson, K. (2013). *Applied Predictive Modeling*, cap. 3. [DOI](https://doi.org/10.1007/978-1-4614-6849-3)
- Documentação do scikit-learn: *Dataset transformations*. [docs](https://scikit-learn.org/stable/data_transforms.html)
