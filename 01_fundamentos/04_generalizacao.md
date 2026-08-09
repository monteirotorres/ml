# Generalização: overfitting e underfitting

O objetivo da aprendizagem de máquina não é acertar nos dados que já vimos, mas
prever bem nos que ainda **não** vimos. Essa capacidade se chama generalização, e
compreendê-la é o conceito mais importante de todo o curso.

▶ [Slides — Fundamentos](fundamentos_slides.html)

## Por que isso importa

Um estudante que decora as respostas da prova antiga vai mal na prova nova; um
que entende a matéria vai bem em ambas. Modelos de ML enfrentam exatamente o
mesmo dilema. Um modelo pode ter erro **zero** nos dados de treino e ainda assim
ser inútil — porque memorizou o ruído em vez de aprender o sinal.

> O erro que importa não é o erro de treino, é o erro de **generalização**: o
> desempenho esperado em dados novos.

## Erro de treino vs. erro de teste

Para medir generalização, separamos os dados: treinamos em uma parte e avaliamos
em outra, **nunca vista** durante o treino.

- **Erro de treino** — quão bem o modelo ajusta os dados que viu.
- **Erro de teste (ou de generalização)** — quão bem ele se sai em dados novos.

A distância entre os dois conta a história. Erro de treino baixo e erro de teste
alto é o sintoma clássico de que algo deu errado.

## Underfitting e overfitting

Existem dois modos de falhar, em direções opostas:

- **Underfitting** (subajuste) — o modelo é **simples demais** para capturar o
  padrão. Erra muito no treino **e** no teste. Ex.: ajustar uma reta a uma
  relação claramente curva.
- **Overfitting** (sobreajuste) — o modelo é **complexo demais** e se cola aos
  detalhes e ao ruído do treino. Erra pouco no treino, mas muito no teste.

Entre os dois extremos há um ponto ideal de complexidade — o "sweet spot".

## O compromisso viés–variância

Essa tensão é formalizada pela decomposição do erro esperado em três parcelas:

$$
\mathbb{E}\big[(y - \hat{f}(\mathbf{x}))^2\big] = \underbrace{\text{Viés}^2}_{\text{simplista}} + \underbrace{\text{Variância}}_{\text{instável}} + \underbrace{\sigma^2}_{\text{ruído irredutível}}
$$

- **Viés** — erro por suposições simplistas. Modelos rígidos (reta) têm viés
  alto → underfitting.
- **Variância** — sensibilidade do modelo a pequenas mudanças nos dados de
  treino. Modelos flexíveis têm variância alta → overfitting.
- **Ruído $\sigma^2$** — parte irredutível; nenhum modelo elimina.

Reduzir o viés (mais complexidade) tende a **aumentar** a variância, e vice-versa.
Aprender bem é equilibrar os dois.

## Visualizando o efeito

O widget abaixo ajusta um polinômio de grau ajustável a alguns pontos ruidosos
gerados a partir de uma função verdadeira (uma onda senoidal). Aumente o grau e
observe: com grau 1 o modelo é rígido demais (**underfitting**); com grau alto
ele passa por todos os pontos de treino, mas se contorce de forma absurda entre
eles (**overfitting**), e o erro de teste dispara. O botão "nova amostra" sorteia
outro conjunto de pontos — repare como o modelo de grau alto muda drasticamente,
sinal de **variância** elevada.

## Como controlar a complexidade

Diante do overfitting, temos várias alavancas:

1. **Mais dados** — a defesa mais eficaz; ruído se dilui, o sinal permanece.
2. **Modelo mais simples** — menos parâmetros, menor grau, menor profundidade.
3. **Regularização** — penalizar a complexidade durante o treino (capítulo 2).
4. **Validação cruzada** — para escolher a complexidade de forma honesta
   (próximo tópico).

Diante do underfitting, o remédio é o oposto: aumentar a capacidade do modelo,
adicionar características ou reduzir a regularização.

## No notebook

O notebook `04_generalizacao.ipynb` reproduz a experiência do widget em Python:
gera dados, ajusta polinômios de vários graus e plota as curvas de erro de treino
e de teste em função do grau, tornando o "sweet spot" visível.

## Exercícios

**1.** Um modelo tem 99% de acurácia no treino e 62% no teste. Qual é o
diagnóstico e o que fazer?

<details><summary>Ver solução</summary>

O grande fosso entre treino (99%) e teste (62%) indica **overfitting**: o modelo
memorizou o treino e não generaliza. Remédios: simplificar o modelo, adicionar
regularização, obter mais dados ou reduzir o número de características.

</details>

**2.** Outro modelo tem 68% de acurácia no treino e 66% no teste. É um caso de
overfitting?

<details><summary>Ver solução</summary>

Não. Treino e teste estão próximos, então **não** há overfitting. Mas o
desempenho é baixo nos dois — sintoma de **underfitting**. O modelo é simples
demais; convém aumentar sua capacidade (mais características, modelo mais
flexível, menos regularização).

</details>

## Referências

- James, G. et al. (2021). *An Introduction to Statistical Learning*, cap. 2.
- Hastie, T. et al. (2009). *The Elements of Statistical Learning*, cap. 7.
- Géron, A. (2019). *Hands-On Machine Learning*, cap. 4.
