# Perceptron e redes neurais simples

Toda rede neural, por maior que seja, é feita da mesma peça minúscula: o
**neurônio artificial**, ou perceptron. Entender essa peça — e por que ela sozinha
é limitada — é entender de onde vem a força das redes profundas.

## O neurônio: soma ponderada + ativação

Um neurônio recebe as entradas, faz uma **combinação linear** delas e passa o
resultado por uma **função de ativação**:

$$
z = \mathbf{w}^\top \mathbf{x} + b = \sum_{j=1}^{p} w_j x_j + b,
\qquad a = f(z)
$$

Onde:

- $\mathbf{x}$ é o **vetor de entradas** e $x_j$ a $j$-ésima entrada;
- $\mathbf{w}$ é o **vetor de pesos** — quanto cada entrada influencia o neurônio —
  e $w_j$ o peso da entrada $j$;
- $b$ é o **viés** (bias), que desloca o limiar de ativação;
- $z$ é a **pré-ativação** (a soma ponderada);
- $f$ é a **função de ativação** (uma não linearidade — próximo tópico) e $a$ a
  **saída** do neurônio.

Repare que, com $f$ sendo a sigmoide, um neurônio é **exatamente** uma regressão
logística. A novidade não é o neurônio isolado — é conectá-los.

O widget a seguir é **um único neurônio ao vivo**: mexa nas entradas $x_1, x_2$, nos
pesos $w_1, w_2$ e no viés $b$ e acompanhe a soma ponderada $z$ correr pelas
conexões até a ativação $f$ produzir a saída $a$. Troque a função de ativação e veja
o mesmo $z$ virar saídas diferentes.

[[widget:wNeuron]]

## De um neurônio a uma rede

Uma **rede neural** (multilayer perceptron, MLP) organiza neurônios em **camadas**:
a saída de uma camada é a entrada da próxima. Uma rede rasa tem três partes:

- a **camada de entrada** — os dados $\mathbf{x}$;
- uma ou mais **camadas ocultas** — neurônios que criam representações intermediárias;
- a **camada de saída** — a predição final.

Cada camada tem sua própria matriz de pesos e seu vetor de vieses. Empilhar camadas
com ativações não lineares é o que dá à rede o poder de aproximar funções
complicadas — um resultado formalizado pelo **teorema da aproximação universal**:
uma rede com uma camada oculta suficientemente larga pode aproximar qualquer função
contínua.

## Por que um neurônio não basta: o XOR

Um único neurônio traça uma **fronteira linear** (uma reta, no plano). Isso resolve
funções como o **E** (AND) e o **OU** (OR), que são linearmente separáveis. Mas
falha no **OU-exclusivo** (XOR): não existe reta que separe os casos de XOR, então
nenhum neurônio isolado o aprende — o limite chega a impossibilitar mais de 75% de
acerto. Foi essa limitação, apontada nos anos 1960, que "congelou" a área até a
chegada das redes de múltiplas camadas, que **resolvem** o XOR combinando várias
fronteiras.

O widget abaixo deixa isso na mão: ajuste os pesos e o viés de um neurônio e escolha
o problema (E, OU, XOR). Para o E e o OU você encontra uma reta que separa tudo;
para o XOR, verá que **nenhuma** reta acerta os quatro casos.

## No notebook

O notebook `01_perceptron.ipynb` implementa um neurônio à mão em PyTorch, treina-o
num problema linearmente separável, mostra que ele **falha** no XOR — e que uma rede
com **uma camada oculta** resolve o XOR, tudo com o laço de treino explícito.

## Exercícios

**1.** Escreva pesos $w_1, w_2$ e viés $b$ para um neurônio (com ativação degrau,
saída 1 se $z \ge 0$) que compute a função **E** (AND), onde as entradas são 0 ou 1.

<details><summary>Ver solução</summary>

Uma solução: $w_1 = w_2 = 1$ e $b = -1{,}5$. Então $z = x_1 + x_2 - 1{,}5$, que é
$\ge 0$ **apenas** quando $x_1 = x_2 = 1$ ($z = 0{,}5$); nos outros três casos $z <
0$. Logo a saída é 1 só para (1,1) — exatamente o AND. (Para o OU, bastaria
$b = -0{,}5$.)

</details>

**2.** Por que empilhar camadas **sem** função de ativação não adianta nada?

<details><summary>Ver solução</summary>

Porque a composição de transformações **lineares** ainda é linear. Se cada camada só
faz $\mathbf{W}\mathbf{x} + \mathbf{b}$, então duas camadas dão
$\mathbf{W}_2(\mathbf{W}_1\mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2 =
\mathbf{W}'\mathbf{x} + \mathbf{b}'$ — uma única transformação linear equivalente. A
rede inteira colapsa em um modelo linear, incapaz de resolver o XOR. É a **não
linearidade** da ativação que quebra esse colapso e dá poder à profundidade.

</details>

## Referências

- Rosenblatt, F. (1958). *The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain*. Psychological Review, 65(6), 386–408. [DOI](https://doi.org/10.1037/h0042519)
- Minsky, M. & Papert, S. (1969). *Perceptrons*. MIT Press — o livro que expôs a limitação do XOR. [DOI](https://doi.org/10.7551/mitpress/11301.001.0001)
- Cybenko, G. (1989). *Approximation by Superpositions of a Sigmoidal Function*. Mathematics of Control, Signals and Systems, 2, 303–314 (aproximação universal). [DOI](https://doi.org/10.1007/BF02551274)
