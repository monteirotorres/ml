# Regressão linear múltipla

Raramente uma resposta depende de um único fator. A regressão **múltipla**
generaliza a reta para vários preditores: a reta vira um **hiperplano**. Isso nos
permite medir o efeito de cada preditor **mantendo os demais fixos** — a ideia de
"ajustar para variáveis de confusão" que sustenta boa parte da inferência em
estudos observacionais.

## O modelo, em forma matricial

$$
\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_p x_p
= \boldsymbol\theta^\top \mathbf{x}
$$

Onde:

- $p$ é o **número de preditores** (as colunas);
- $x_j$ é o **$j$-ésimo preditor** e $\theta_j$ o seu **coeficiente parcial** — de
  quanto $\hat{y}$ muda quando $x_j$ aumenta uma unidade **e todos os outros
  preditores ficam constantes**. Essa cláusula "todos os outros fixos" é o que
  torna o coeficiente parcial diferente de uma regressão simples de $y$ contra
  $x_j$ sozinho;
- $\boldsymbol\theta = (\theta_0, \theta_1, \dots, \theta_p)$ é o **vetor de
  parâmetros** e $\mathbf{x} = (1, x_1, \dots, x_p)$ o vetor de entrada (o $1$ na
  frente absorve o intercepto $\theta_0$).

Empilhando os $n$ exemplos nas linhas de uma **matriz de design** $\mathbf{X}$ de
forma $n \times (p+1)$, todas as predições viram um único produto de matrizes,
$\hat{\mathbf{y}} = \mathbf{X}\boldsymbol\theta$, e a solução de mínimos quadrados
tem forma fechada — a **equação normal**:

$$
\boldsymbol\theta = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
$$

Onde $\mathbf{X}^\top$ é a transposta de $\mathbf{X}$ e ${}^{-1}$ é a inversa. Na
prática nunca invertemos essa matriz à mão — o scikit-learn resolve por métodos
numéricos mais estáveis —, mas a fórmula mostra que o ajuste é direto quando
$\mathbf{X}^\top\mathbf{X}$ é **inversível**. E é justamente aí que mora o próximo
problema.

## Colinearidade: o inimigo silencioso

Quando dois preditores são muito **correlacionados** (digamos, peso e índice de
massa corporal), eles carregam quase a mesma informação. A matriz
$\mathbf{X}^\top\mathbf{X}$ fica quase não-inversível, e as estimativas dos
coeficientes ficam **instáveis**: pequenas mudanças nos dados provocam grandes
saltos em $\boldsymbol\theta$. O modelo ainda prevê bem, mas os coeficientes
individuais deixam de ser confiáveis para interpretação.

O diagnóstico padrão é o **fator de inflação da variância** (VIF). Para o preditor
$x_j$:

$$
\text{VIF}_j = \frac{1}{1 - R_j^2}
$$

Onde $R_j^2$ é o **coeficiente de determinação** de uma regressão de $x_j$ contra
todos os *outros* preditores. Se $x_j$ é bem previsto pelos demais ($R_j^2$ perto
de 1), o VIF explode. A regra de bolso comum: VIF acima de 5–10 pede atenção.

O widget abaixo deixa isso concreto: um slider controla a correlação entre dois
preditores; à medida que a nuvem de pontos colapsa numa linha, o VIF dispara e a
variância dos coeficientes estimados cresce junto.

## Codificando variáveis categóricas

Preditores categóricos (sexo, grupo de tratamento) entram como **variáveis
indicadoras** (dummies): uma coluna 0/1 por categoria, deixando uma de fora como
referência. O coeficiente de cada dummy lê-se como a diferença média em relação a
essa categoria de referência.

## No notebook

O notebook `02_regressao_multipla.ipynb` usa o conjunto **diabetes** do
scikit-learn (dez fatores clínicos prevendo a progressão da doença). Montamos a
matriz de design, resolvemos pela equação normal com NumPy explícito, conferimos
contra o `LinearRegression`, e calculamos o VIF de cada preditor com um laço à
mostra para flagrar redundâncias.

## Exercícios

**1.** Em uma regressão simples, o coeficiente de "horas de exercício" contra
"colesterol" deu negativo. Ao adicionar "idade" ao modelo, ficou perto de zero.
Como interpretar isso?

<details><summary>Ver solução</summary>

O efeito aparente do exercício sozinho estava **confundido** pela idade: pessoas
que se exercitam mais também tendem a ser mais jovens, e a idade é que puxava o
colesterol. O coeficiente **parcial** (com idade no modelo) mede o efeito do
exercício *entre pessoas de mesma idade*, e ele quase some. É exatamente para isso
que serve a regressão múltipla — separar efeitos que a análise de um preditor só
mistura.

</details>

**2.** Se o VIF de um preditor é 25, qual é aproximadamente o $R_j^2$ dele contra
os demais? O que você faria?

<details><summary>Ver solução</summary>

De $\text{VIF} = 1/(1 - R_j^2) = 25$, tem-se $R_j^2 = 0{,}96$: 96% da variação
desse preditor é explicada pelos outros — ele é quase redundante. Opções: remover
um dos preditores colineares, combiná-los em um só (ex.: um índice), ou usar
regularização (próximo tópico), que lida bem com colinearidade.

</details>

## Referências

- James, G., Witten, D., Hastie, T., Tibshirani, R. & Taylor, J. (2023). *An Introduction to Statistical Learning with Applications in Python*, cap. 3. [Livro aberto](https://www.statlearning.com/)
- Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*, cap. 3. [DOI](https://doi.org/10.1007/978-0-387-84858-7)
- Belsley, D. A., Kuh, E. & Welsch, R. E. (1980). *Regression Diagnostics: Identifying Influential Data and Sources of Collinearity*. Wiley. [DOI](https://doi.org/10.1002/0471725153)
