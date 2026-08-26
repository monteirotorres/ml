# Regressão polinomial

Nem toda relação é uma reta. A regressão polinomial captura **curvaturas** sem
abandonar o arcabouço linear: basta incluir potências do preditor como novas
colunas. O truque conceitual é que o modelo continua **linear nos parâmetros** —
por isso todo o maquinário de mínimos quadrados continua valendo.

## De reta a curva

$$
\hat{y} = \theta_0 + \theta_1 x + \theta_2 x^2 + \cdots + \theta_d x^d
$$

Onde:

- $d$ é o **grau** do polinômio — o maior expoente incluído;
- $\theta_0, \dots, \theta_d$ são os **coeficientes**, ajustados exatamente como na
  regressão múltipla, tratando $x^2, x^3, \dots, x^d$ como se fossem preditores
  independentes $x_2, x_3, \dots$;
- com $d = 1$ recuperamos a reta; $d = 2$ dá uma parábola; graus maiores permitem
  curvas cada vez mais sinuosas.

A mágica: embora a **curva** em $x$ seja não-linear, a **função nos parâmetros**
$\theta_j$ é linear. Construímos a matriz de design com as potências de $x$ nas
colunas e resolvemos como uma regressão múltipla comum.

## O compromisso viés–variância, ao vivo

O grau $d$ é um **hiperparâmetro** que controla a flexibilidade do modelo, e ele é
o exemplo didático perfeito do compromisso central da ML:

- **grau baixo demais** → o modelo é rígido, não acompanha a curvatura real e erra
  por igual no treino e no teste. É o **underfitting** (viés alto);
- **grau alto demais** → o modelo tem liberdade para passar por cima de cada ponto
  de treino, inclusive o ruído, e erra feio em dados novos. É o **overfitting**
  (variância alta);
- **grau certo** → captura o sinal e ignora o ruído.

O erro de **treino** cai monotonicamente com o grau — mais flexibilidade sempre
ajusta melhor os pontos vistos. O erro de **teste** cai e depois volta a subir, e
o fundo dessa curva em U marca o grau ideal. Por isso o grau **não** pode ser
escolhido pelo erro de treino; precisamos de dados de validação (tópico de
validação cruzada, no capítulo de Fundamentos).

O widget abaixo mostra isso: mova o slider do grau e veja a curva sair de rígida
(subajustada) a excessivamente sinuosa (sobreajustada), enquanto os cartões de
erro de treino e de teste seguem caminhos opostos.

## No notebook

O notebook `03_regressao_polinomial.ipynb` usa o `PolynomialFeatures` do
scikit-learn dentro de um `Pipeline`, ajusta polinômios de vários graus a dados
com curvatura conhecida e traça a **curva de seleção do grau** — erro de treino e
de validação em função de $d$ — para escolher o grau ótimo com base em dados que o
modelo não viu.

## Exercícios

**1.** Por que o erro de treino nunca aumenta quando subimos o grau, mas o erro de
teste pode piorar?

<details><summary>Ver solução</summary>

Aumentar o grau só **acrescenta** flexibilidade: um polinômio de grau $d+1$
contém todos os de grau $d$ como caso particular (basta $\theta_{d+1}=0$), então
ele nunca ajusta os pontos de treino pior — no máximo igual. Mas essa
flexibilidade extra pode ser usada para acompanhar o **ruído** do treino, que não
se repete em dados novos; aí o erro de **teste** sobe. É a assinatura do
overfitting.

</details>

**2.** Um colega ajusta um polinômio de grau 15 a 16 pontos e obtém erro de treino
praticamente zero. Ele deveria comemorar?

<details><summary>Ver solução</summary>

Não. Com 16 pontos, um polinômio de grau 15 tem parâmetros suficientes para passar
**exatamente** por todos eles (interpolação) — erro de treino zero é esperado e não
diz nada sobre generalização. O teste em dados novos quase certamente será
péssimo. É o retrato do overfitting: memorizou, não aprendeu.

</details>

## Referências

- James, G., Witten, D., Hastie, T., Tibshirani, R. & Taylor, J. (2023). *An Introduction to Statistical Learning with Applications in Python*, cap. 7. Livro aberto: https://www.statlearning.com/
- Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*, cap. 2 e 7 (viés–variância).
