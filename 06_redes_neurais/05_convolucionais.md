# Redes convolucionais (CNNs)

Uma rede densa trata a imagem como uma **lista de pixels soltos**: desmancha as
linhas e colunas num vetor e liga cada pixel a cada neurônio. Isso joga fora a
informação mais óbvia de uma imagem — que **pixels vizinhos formam padrões** e que um
padrão (uma borda, uma textura) é o mesmo esteja ele no canto ou no centro. As
**redes convolucionais** (CNNs) foram feitas para respeitar essa estrutura, e são a
base da visão computacional e da análise de imagens médicas — de radiografias a
lâminas de histopatologia.

## A convolução: um filtro que desliza

O ingrediente central é a **convolução**. Em vez de um peso por pixel, a rede aprende
um **filtro** pequeno — um quadradinho de pesos, tipicamente $3\times3$. Esse filtro
**desliza** por toda a imagem e, em cada posição, calcula uma soma ponderada da
janelinha de pixels que cobre:

$$
s_{ij} = \sum_{u}\sum_{v} K_{uv}\, \cdot\, I_{\,i+u,\;j+v}
$$

onde $I$ é a imagem, $K$ é o filtro (kernel) e $s_{ij}$ é a resposta na posição
$(i,j)$. O resultado, varrendo a imagem inteira, é um novo mapa chamado **mapa de
ativação** (ou mapa de características): ele acende onde o padrão do filtro aparece.

Um filtro que responde a **bordas verticais**, por exemplo, produz valores altos
exatamente nas colunas onde a imagem muda de claro para escuro. O ponto-chave: **é o
mesmo filtro em toda a imagem**. A CNN não reaprende "o que é uma borda" para cada
canto — ela aprende **uma vez** e aplica em todo lugar.

O widget abaixo deixa isso concreto: escolha uma imagem e um filtro, e arraste a
posição para ver a janela $3\times3$ percorrer a imagem e o valor que ela produz no
mapa de ativação à direita.

[[widget]]

## Duas ideias que economizam parâmetros

A convolução carrega duas propriedades que explicam por que as CNNs funcionam tão
bem em imagens:

- **Compartilhamento de pesos** — o mesmo filtro (poucos pesos) é reusado em todas as
  posições. Uma camada densa ligando uma imagem de $200\times200$ a mil neurônios
  teria dezenas de milhões de pesos; um filtro $3\times3$ tem **nove**. Menos
  parâmetros significam menos dados necessários e menos sobreajuste.
- **Invariância à translação** — como o filtro é o mesmo em todo lugar, um padrão é
  reconhecido **independentemente de onde** ele esteja na imagem. Um tumor não precisa
  aparecer sempre no mesmo canto para ser detectado.

Cada camada convolucional aprende **vários** filtros ao mesmo tempo (dezenas ou
centenas), cada um sensível a um padrão diferente — bordas em várias orientações,
manchas, texturas.

## Pooling: resumir e encolher

Entre as convoluções, as CNNs costumam intercalar uma operação de **pooling**
(agrupamento), que **reduz a resolução** do mapa de ativação. O mais comum é o
*max pooling*: divide o mapa em blocos (por exemplo $2\times2$) e mantém só o **maior**
valor de cada bloco. Isso encolhe a imagem pela metade, barateia o resto da rede e dá
uma tolerância extra a pequenos deslocamentos — não importa a posição exata do padrão
dentro do bloco, só que ele esteja lá.

## A arquitetura típica: da borda ao objeto

Empilhando esses blocos, a CNN monta a **hierarquia de representações** do tópico
anterior, agora de forma concreta:

- as **primeiras** camadas convolucionais detectam padrões simples — bordas e cantos;
- as **intermediárias** combinam bordas em **partes** (um contorno, uma textura de
  tecido);
- as **profundas** combinam partes em **estruturas** de alto nível (uma célula, uma
  lesão, um objeto inteiro).

No fim, os mapas resultantes são achatados e entregues a uma pequena **rede densa**,
que faz a classificação final. O padrão clássico é:
`convolução → ativação (ReLU) → pooling`, repetido algumas vezes, e então
`achatamento → camada densa → saída`. Tudo treina pelo mesmo motor dos tópicos
anteriores: **gradiente descendente** e **backpropagation**.

## No notebook

O notebook `05_convolucionais.ipynb` faz a convolução **na mão** com `numpy` sobre um
dígito real (para você ver o filtro de borda acender), e depois monta uma pequena CNN
em PyTorch (`Conv2d` + `MaxPool2d`) para classificar dígitos manuscritos — comparando
a acurácia com a da rede densa do tópico anterior sobre os **mesmos** dados.

## Exercícios

**1.** Uma imagem em tons de cinza tem $32\times32$ pixels. Uma camada densa a liga a
100 neurônios; uma camada convolucional usa 8 filtros $3\times3$. Quantos pesos
(sem contar vieses) cada uma tem? O que isso revela?

<details><summary>Ver solução</summary>

A densa tem $32\times32\times100 = 102\,400$ pesos. A convolucional tem
$8\times3\times3 = 72$ pesos (cada filtro reusado em toda a imagem). A convolução usa
**muito menos** parâmetros porque **compartilha** os pesos por todas as posições — daí
precisar de menos dados e sobreajustar menos, aproveitando a estrutura espacial da
imagem.

</details>

**2.** Por que aplicar o **mesmo** filtro em toda a imagem (em vez de um peso por
pixel) ajuda a reconhecer um padrão em qualquer posição?

<details><summary>Ver solução</summary>

Porque o filtro responde ao **padrão local** (por exemplo, uma borda vertical),
não à sua localização. Como ele desliza e é calculado igual em cada posição, o mesmo
padrão produz uma resposta alta **onde quer que apareça** — é a **invariância à
translação**. Um peso por pixel, ao contrário, amarraria o reconhecimento a uma
posição específica, e a rede teria de reaprender o padrão para cada canto da imagem.

</details>

## Referências

- LeCun, Y., Bottou, L., Bengio, Y. & Haffner, P. (1998). *Gradient-Based Learning Applied to Document Recognition* (LeNet). Proceedings of the IEEE, 86(11), 2278–2324. [DOI](https://doi.org/10.1109/5.726791)
- Krizhevsky, A., Sutskever, I. & Hinton, G. (2012). *ImageNet Classification with Deep Convolutional Neural Networks* (AlexNet). NeurIPS. [DOI](https://doi.org/10.1145/3065386)
- Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*, cap. 9 (Convolutional Networks). MIT Press. [Livro aberto](https://www.deeplearningbook.org/)
