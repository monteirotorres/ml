# Funções de ativação

A função de ativação é a peça que dá **não linearidade** ao neurônio — e, como
vimos, sem ela uma rede profunda colapsa em um único modelo linear. A escolha da
ativação parece um detalhe, mas afeta diretamente a velocidade do treino e até se a
rede **consegue** aprender.

## As três clássicas

**Sigmoide** — espreme qualquer número no intervalo $(0, 1)$:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma'(z) = \sigma(z)\,(1 - \sigma(z))
$$

**Tangente hiperbólica (tanh)** — parecida, mas centrada em zero, saída em
$(-1, 1)$:

$$
\tanh(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}, \qquad \tanh'(z) = 1 - \tanh^2(z)
$$

**ReLU** (unidade linear retificada) — simplesmente zera os negativos:

$$
\text{ReLU}(z) = \max(0, z), \qquad \text{ReLU}'(z) = \begin{cases} 1 & z > 0 \\ 0 & z < 0 \end{cases}
$$

Onde, em todas:

- $z$ é a **pré-ativação** (a soma ponderada de entradas);
- $f'(z)$ é a **derivada** da ativação — e é ela que a backpropagation multiplica ao
  propagar o gradiente. Guarde isso: **onde a derivada é pequena, o gradiente
  encolhe**.

O widget abaixo desenha cada ativação e a sua **derivada**. Repare no que acontece
com a derivada da sigmoide e da tanh longe do zero — ela some.

## O gradiente que desaparece

Na sigmoide e na tanh, quando $z$ é muito positivo ou muito negativo, a função
**satura** (fica quase plana) e sua derivada vai a **quase zero**. Numa rede
profunda, a backpropagation multiplica muitas dessas derivadas em cadeia; se cada uma
é menor que 1, o produto encolhe exponencialmente e o gradiente que chega às
primeiras camadas é **praticamente nulo**. Elas param de aprender. É o problema do
**gradiente que desaparece** (*vanishing gradient*), que por muito tempo travou o
treino de redes profundas.

## Por que a ReLU venceu

A ReLU resolve isso na região positiva: para $z > 0$, sua derivada é **exatamente
1**, então o gradiente passa **sem encolher**. Ela é também baratíssima de calcular
(um simples "corte no zero"). Por isso a ReLU virou o **padrão** nas camadas ocultas
das redes modernas. Seu único defeito — neurônios que "morrem" ao ficar presos na
região negativa (derivada 0) — é atenuado por variantes como a **Leaky ReLU**, que
deixa passar uma pequena inclinação para os negativos:
$\text{LeakyReLU}(z) = \max(\alpha z, z)$, com $\alpha$ pequeno (ex.: 0,01).

Uma nota: na **saída**, a ativação segue a tarefa — sigmoide para probabilidade
binária, *softmax* para várias classes, e nenhuma (linear) para regressão. O
problema do gradiente que desaparece é sobre as camadas **ocultas**.

## No notebook

O notebook `03_ativacoes.ipynb` desenha as ativações e suas derivadas com Plotly,
treina a mesma rede com sigmoide e com ReLU num problema não linear (comparando a
velocidade de convergência) e **mede** a magnitude do gradiente nas primeiras camadas
de uma rede profunda, tornando o *vanishing gradient* um número, não só uma história.

## Exercícios

**1.** A derivada máxima da sigmoide é $0{,}25$ (em $z = 0$). Numa rede de 10 camadas
com sigmoides, estime o fator pelo qual o gradiente encolhe, no melhor caso, ao
chegar à primeira camada.

<details><summary>Ver solução</summary>

No melhor caso, cada camada multiplica o gradiente por, no máximo, $0{,}25$. Em 10
camadas, o fator é $0{,}25^{10} \approx 9{,}5 \times 10^{-7}$ — o gradiente chega à
primeira camada cerca de **um milhão de vezes menor**. E isso é o *melhor* caso
(longe do zero, encolhe ainda mais). Fica claro por que redes profundas com sigmoide
quase não treinam as camadas iniciais — e por que a ReLU, com derivada 1, é tão
melhor.

</details>

**2.** Por que a ReLU **não** sofre do gradiente que desaparece na região positiva,
mas ainda assim pode ter neurônios "mortos"?

<details><summary>Ver solução</summary>

Para $z > 0$, a derivada da ReLU é exatamente **1**: o gradiente passa sem encolher,
então não há desvanecimento por saturação nessa região. Mas para $z < 0$ a derivada
é **0**; se um neurônio, por um peso ou viés infeliz, passa a receber sempre $z < 0$,
seu gradiente é sempre zero e ele **nunca mais se atualiza** — está "morto". A Leaky
ReLU evita isso dando uma pequena inclinação (derivada $\alpha > 0$) aos negativos.

</details>

## Referências

- Nair, V. & Hinton, G. (2010). *Rectified Linear Units Improve Restricted Boltzmann Machines*. ICML. [PDF](https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf)
- Glorot, X. & Bengio, Y. (2010). *Understanding the Difficulty of Training Deep Feedforward Neural Networks*. AISTATS (sobre o gradiente que desaparece). [PMLR](https://proceedings.mlr.press/v9/glorot10a.html)
- Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*, cap. 6. MIT Press. [Livro aberto](https://www.deeplearningbook.org/)
