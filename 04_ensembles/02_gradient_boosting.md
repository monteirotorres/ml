# Gradient Boosting

A random forest treina árvores **em paralelo** e tira a média. O *boosting* faz o
oposto: treina árvores **em sequência**, cada uma consertando o que a anterior
errou. Enquanto o bagging ataca a **variância**, o boosting ataca o **viés** —
construindo, passo a passo, um modelo forte a partir de modelos fracos.

## A ideia: corrigir os resíduos

Começamos com uma previsão grosseira (por exemplo, a média de $y$) e melhoramos em
$M$ estágios. A cada estágio, uma nova árvore aprende **o erro que ainda resta**:

$$
F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu \, h_m(\mathbf{x})
$$

Onde:

- $F_m$ é o **modelo acumulado** após $m$ estágios;
- $h_m$ é a **nova árvore** do estágio $m$, ajustada aos erros de $F_{m-1}$;
- $\nu$ (a letra grega *nu*) é a **taxa de aprendizado** ($0 < \nu \le 1$) — o
  tamanho do passo, o quanto de cada nova árvore incorporamos.

O que $h_m$ aprende? No **gradient** boosting, ela é ajustada ao **gradiente
negativo** da função de perda em relação à previsão atual. Para a perda quadrática,
esse gradiente negativo é simplesmente o **resíduo** $y_i - F_{m-1}(\mathbf{x}_i)$:
cada árvore aprende, literalmente, o que falta. O nome "gradient" vem daí — é
gradiente descendente, só que no espaço das funções.

O widget abaixo mostra o processo em uma curva 1D: aumente o número de estágios e
veja o modelo acumulado se aproximar dos dados, um pedacinho por vez; mude a taxa
de aprendizado e observe o passo mudar de tamanho.

## A taxa de aprendizado e o número de árvores

Esses dois hiperparâmetros andam de mãos dadas:

- $\nu$ **pequeno** (ex.: 0,1 ou menos) → cada árvore contribui pouco, o modelo
  aprende devagar e com cuidado; costuma **generalizar melhor**, mas exige **mais
  árvores**.
- $\nu$ **grande** → aprende rápido, com menos árvores, mas arrisca passar do ponto.

E, ao contrário da random forest, **mais árvores no boosting podem piorar**: como
cada árvore persegue os resíduos, um número excessivo acaba ajustando o **ruído**
do treino. O número de estágios é, ele próprio, um regulador de overfitting — e se
escolhe por validação (muitas vezes com *early stopping*, parando quando o erro de
validação para de cair).

## No notebook

O notebook `02_gradient_boosting.ipynb` constrói o boosting **à mão** num problema
de regressão 1D — ajustando árvores rasas aos resíduos num laço explícito, para ver
o modelo tomar forma — e depois usa o `GradientBoostingClassifier` do scikit-learn,
mostrando o efeito da taxa de aprendizado e do número de estágios no erro de treino
e de validação.

## Exercícios

**1.** Por que, no gradient boosting com perda quadrática, dizer que a árvore
"aprende o gradiente negativo" é o mesmo que dizer que ela "aprende os resíduos"?

<details><summary>Ver solução</summary>

A perda quadrática de um exemplo é $\ell = \tfrac{1}{2}(y - F(\mathbf{x}))^2$. Sua
derivada em relação à previsão $F$ é $\partial \ell / \partial F = -(y - F)$, então o
**gradiente negativo** é $-(\partial \ell/\partial F) = y - F$, que é exatamente o
**resíduo**. Ajustar a nova árvore ao gradiente negativo é, nesse caso, ajustá-la
ao resíduo. (Para outras perdas, o gradiente negativo é uma versão generalizada do
resíduo.)

</details>

**2.** Duas configurações dão o mesmo erro de treino: (A) $\nu = 0{,}5$ com 100
árvores; (B) $\nu = 0{,}05$ com 1000 árvores. Qual tende a generalizar melhor, e por
quê?

<details><summary>Ver solução</summary>

Em geral a **(B)**: uma taxa de aprendizado menor, compensada por mais árvores, faz
o modelo avançar em passos pequenos e cautelosos, o que costuma suavizar o ajuste e
melhorar a generalização (é uma forma de regularização, o *shrinkage*). A (A) dá
passos grandes e arrisca ajustar ruído. O custo da (B) é treinar mais árvores — o
clássico compromisso entre desempenho e tempo de treino.

</details>

## Referências

- Freund, Y. & Schapire, R. (1997). *A Decision-Theoretic Generalization of On-Line Learning and an Application to Boosting* (AdaBoost). Journal of Computer and System Sciences, 55(1), 119–139. [DOI](https://doi.org/10.1006/jcss.1997.1504)
- Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. Annals of Statistics, 29(5), 1189–1232 — o artigo seminal. [DOI](https://doi.org/10.1214/aos/1013203451)
- Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*, cap. 10. [DOI](https://doi.org/10.1007/978-0-387-84858-7)
