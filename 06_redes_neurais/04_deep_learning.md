# Introdução ao deep learning

*Deep learning* é, em uma frase, o uso de redes neurais com **muitas camadas**. Mas
a profundidade não é só "mais do mesmo": ela muda qualitativamente o que a rede
aprende. Este tópico fecha o capítulo com um panorama — o que a profundidade traz,
as arquiteturas principais e, honestamente, **quando** o deep learning compensa e
quando não.

## O que a profundidade traz: representações hierárquicas

Em métodos clássicos, nós escolhíamos as características à mão (os descritores, os
fingerprints). A grande virada do deep learning é aprender as características
**diretamente dos dados brutos**, em uma **hierarquia**:

- as primeiras camadas aprendem padrões **simples** (bordas, em imagens; fonemas, em
  áudio);
- as camadas seguintes **combinam** os simples em padrões mais **complexos** (olhos,
  rodas; sílabas, palavras);
- as últimas montam os conceitos de **alto nível** (um rosto, um carro; uma frase).

Cada camada constrói sobre a representação da anterior. Essa **composição** de
representações é o que dá ao deep learning sua força em dados de altíssima dimensão —
e é por isso que ele dispensa boa parte da engenharia manual de características.

## As arquiteturas principais

A rede densa (MLP) deste capítulo é a base. Sobre ela, arquiteturas especializadas
exploram a **estrutura** de cada tipo de dado:

- **CNNs** (redes convolucionais) — para **imagens**. Aplicam os mesmos filtros por
  toda a imagem, aproveitando que um padrão (uma borda) é o mesmo em qualquer
  posição. São a base da visão computacional e da análise de imagens médicas.
- **RNNs** e, hoje, **Transformers** — para **sequências** (texto, séries temporais,
  proteínas). O Transformer, com seu mecanismo de **atenção**, é a arquitetura por
  trás dos grandes modelos de linguagem.

O denominador comum: todas treinam pelo mesmo motor — gradiente descendente e
backpropagation — sobre os mesmos frameworks (**PyTorch**, Keras/TensorFlow).

## Quando o deep learning compensa (e quando não)

Esta é a parte que a empolgação costuma esconder. Deep learning **brilha** quando:

- há **muitos dados** (dezenas de milhares de exemplos ou mais);
- os dados são de **alta dimensão e pouco estruturados** — imagens, texto, áudio,
  sinais brutos;
- vale investir em **computação** (GPUs) e ajuste.

E **não** compensa — ou até perde — quando:

- os dados são **tabulares** e em quantidade modesta: aqui, um **gradient boosting**
  (capítulo 4) costuma igualar ou superar uma rede, com muito menos esforço;
- há **poucos exemplos**: redes profundas têm parâmetros demais e sobreajustam;
- a **interpretabilidade** é essencial: uma rede é uma caixa mais fechada que uma
  regressão ou uma árvore.

A lição do curso inteiro vale aqui: o melhor modelo é o que **se encaixa no problema
e nos dados**, não o mais na moda. Deep learning é uma ferramenta poderosa para o
tipo certo de problema — não uma resposta universal.

## No notebook

O notebook `04_deep_learning.ipynb` treina uma pequena rede densa em PyTorch para
classificar **dígitos manuscritos** (imagens), compara seu desempenho com o de uma
regressão logística e discute, com números, o que a profundidade acrescentou — e a
que custo.

## Exercícios

**1.** Você tem uma planilha com 800 pacientes e 15 exames, e quer prever um
desfecho binário. Deep learning é a melhor escolha? Justifique.

<details><summary>Ver solução</summary>

Provavelmente **não**. São dados **tabulares** e em quantidade **modesta** (800
linhas) — exatamente o cenário em que um **gradient boosting** ou uma **random
forest** costumam igualar ou superar uma rede neural, com muito menos ajuste e risco
de overfitting, além de mais interpretabilidade. Deep learning renderia mais se
fossem imagens, texto ou dezenas de milhares de exemplos.

</details>

**2.** O que significa dizer que uma CNN aprende "representações hierárquicas" de uma
imagem?

<details><summary>Ver solução</summary>

Significa que as camadas aprendem padrões em **níveis crescentes de complexidade**,
cada um construído sobre o anterior: as primeiras camadas detectam **bordas** e
texturas simples; as intermediárias combinam bordas em **partes** (um olho, uma
roda); as finais combinam partes em **objetos** inteiros (um rosto, um carro). A rede
constrói sozinha essa escada de representações, a partir dos pixels brutos, sem que
ninguém defina à mão o que é uma borda ou um olho.

</details>

## Referências

- LeCun, Y., Bengio, Y. & Hinton, G. (2015). *Deep Learning*. Nature, 521, 436–444.
- Krizhevsky, A., Sutskever, I. & Hinton, G. (2012). *ImageNet Classification with Deep Convolutional Neural Networks* (AlexNet). NeurIPS.
- Vaswani, A. et al. (2017). *Attention Is All You Need* (Transformer). NeurIPS.
- Goodfellow, I., Bengio, Y. & Courville, A. (2016). *Deep Learning*. MIT Press. Livro aberto: https://www.deeplearningbook.org/
