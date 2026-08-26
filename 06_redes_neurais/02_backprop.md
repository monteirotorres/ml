# Backpropagation e gradiente descendente

Uma rede aprende ajustando seus pesos para reduzir o erro. Duas peças fazem isso
acontecer: o **gradiente descendente**, que diz *em que direção* mexer os pesos, e a
**backpropagation**, que calcula esse gradiente de forma eficiente. Juntas, são o
motor de aprendizado de toda rede neural.

## Gradiente descendente: descer a montanha

Imagine a perda como uma superfície montanhosa sobre o espaço dos pesos. Queremos o
ponto mais baixo. O **gradiente** aponta na direção de subida mais íngreme, então
andamos no sentido **oposto**:

$$
\mathbf{w} \leftarrow \mathbf{w} - \eta \, \nabla_{\!\mathbf{w}} \mathcal{L}
$$

Onde:

- $\mathbf{w}$ são os **pesos** da rede;
- $\mathcal{L}$ é a **perda** (o erro que queremos minimizar);
- $\nabla_{\!\mathbf{w}}\mathcal{L}$ é o **gradiente** da perda — o vetor de derivadas
  parciais $\partial\mathcal{L}/\partial w_j$, apontando para onde a perda mais cresce;
- $\eta$ é a **taxa de aprendizado** — o tamanho do passo (a mesma $\eta$ do tópico
  de ferramentas).

Repetindo esse passo, descemos a superfície até um mínimo. O widget abaixo mostra
uma bolinha descendo uma superfície de perda: ajuste a taxa de aprendizado e veja o
compromisso — pequena demais engatinha; grande demais salta o mínimo e pode
**divergir**.

## Backpropagation: a regra da cadeia, organizada

Uma rede profunda tem os pesos escondidos atrás de várias camadas. Como calcular
$\partial\mathcal{L}/\partial w_j$ de um peso lá no começo, se ele afeta a perda só
depois de passar por tudo? Pela **regra da cadeia** do cálculo. A backpropagation é
a aplicação organizada dessa regra: ela calcula o erro na saída e o **propaga de
volta**, camada por camada, reaproveitando os cálculos. Para um peso na camada
$\ell$, o gradiente é o produto encadeado das derivadas locais desde a saída até ele.

A boa notícia prática: **você não faz isso à mão**. O *autograd* do PyTorch registra
cada operação do passo para frente e, ao chamar `.backward()`, executa a
backpropagation automaticamente. Entender o mecanismo importa; implementá-lo, não.

## Full-batch, estocástico e mini-batch

Quanto dos dados usar para estimar o gradiente a cada passo?

- **Full-batch** — usa **todos** os exemplos por passo. Gradiente exato, mas caro e
  lento por passo.
- **Estocástico (SGD)** — usa **um** exemplo por passo. Barato e ruidoso; o ruído
  até ajuda a escapar de mínimos ruins, mas a trajetória zigue-zagueia.
- **Mini-batch** — usa um **pequeno lote** (32, 64, 128…) por passo. O meio-termo
  que domina a prática: aproveita a vetorização do hardware e suaviza o ruído.

Uma passagem completa por todos os dados é chamada de **época** (epoch); o treino
costuma levar muitas épocas.

## No notebook

O notebook `02_backprop.ipynb` mostra o *autograd* calculando gradientes de um
exemplo simples, treina uma rede com o laço explícito (forward → perda → backward →
passo) e compara as curvas de perda de **full-batch, mini-batch e SGD**, tornando
visível o ruído de cada estratégia.

## Exercícios

**1.** Por que o gradiente descendente anda no sentido **oposto** ao gradiente?

<details><summary>Ver solução</summary>

Por definição, o gradiente $\nabla_{\!\mathbf{w}}\mathcal{L}$ aponta na direção em que
a perda **mais aumenta**. Como queremos **reduzir** a perda, andamos no sentido
contrário, $-\nabla_{\!\mathbf{w}}\mathcal{L}$ — a direção de descida mais íngreme.
Daí o sinal de menos na regra de atualização.

</details>

**2.** O SGD tem uma trajetória bem mais ruidosa que o full-batch. Cite uma
**vantagem** desse ruído.

<details><summary>Ver solução</summary>

O ruído do SGD pode ajudar a **escapar de mínimos locais ruins** e de pontos de sela:
como cada passo usa só um exemplo, a direção varia bastante, e essa variação pode
tirar o otimizador de uma região onde o gradiente exato ficaria preso. Além disso,
passos baratos permitem **muitas** atualizações por época. O preço é a trajetória
zigue-zagueante — por isso o mini-batch, que reduz o ruído sem perder eficiência, é
o padrão.

</details>

## Referências

- Rumelhart, D., Hinton, G. & Williams, R. (1986). *Learning Representations by Back-Propagating Errors*. Nature, 323, 533–536 — o artigo que popularizou a backpropagation.
- Robbins, H. & Monro, S. (1951). *A Stochastic Approximation Method*. Annals of Mathematical Statistics, 22(3), 400–407 (origem do SGD).
- Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*, cap. 6 e 8. MIT Press. Livro aberto: https://www.deeplearningbook.org/
