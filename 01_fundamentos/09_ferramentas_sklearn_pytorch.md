# As ferramentas: scikit-learn e PyTorch

Até aqui falamos de ideias — tarefa, experiência, generalização. Agora vamos
conhecer as duas ferramentas que usaremos para transformar essas ideias em
código no resto do curso: o **scikit-learn** e o **PyTorch**. A boa notícia é que
ambos foram projetados para que você escreva *pouco* e entenda *tudo*.

A regra de bolso deste curso: **use o scikit-learn para os modelos clássicos**
(regressão, árvores, florestas, k-means, PCA…) e **use o PyTorch quando precisar
construir e treinar uma rede neural por conta própria**, vendo cada peça do
mecanismo. Não são rivais — são camadas diferentes de abstração para propósitos
diferentes.

## scikit-learn: a mesma interface para tudo

O que torna o scikit-learn (Pedregosa et al., 2011) tão agradável é uma decisão
de projeto simples e consistente: **todo modelo é um objeto com os mesmos três
métodos**.

- `.fit(X, y)` — **ajusta** (treina) o modelo aos dados;
- `.predict(X)` — devolve as **predições** para novas entradas;
- `.score(X, y)` — devolve uma **medida de desempenho** padrão.

Isso vale para uma regressão linear, uma floresta aleatória ou um k-means: troca-se
a classe e o resto do código continua igual. Essa uniformidade é o que permite
comparar modelos de forma justa, trocando uma única linha.

Os dados entram sempre no mesmo formato:

$$
\mathbf{X} \in \mathbb{R}^{n \times p}, \qquad \mathbf{y} \in \mathbb{R}^{n}
$$

Onde:

- $n$ é o número de **exemplos** (as linhas — cada flor, cada paciente, cada
  molécula);
- $p$ é o número de **características** (as colunas — cada medida que descreve o
  exemplo);
- $\mathbf{X}$ é a **matriz de características**, com um exemplo por linha;
- $\mathbf{y}$ é o **vetor-alvo**, com o valor que queremos prever para cada
  exemplo (uma classe, na classificação; um número, na regressão).

Guardar esse formato $(\mathbf{X}, \mathbf{y})$ na cabeça resolve metade dos erros
de quem está começando.

### Qual modelo escolher? O mapa do scikit-learn

Diante de tantos algoritmos, a pergunta inevitável é *"por onde começo?"*. A
própria documentação do scikit-learn responde com um fluxograma famoso — o
[**mapa de estimadores**](https://scikit-learn.org/1.3/tutorial/machine_learning_map/) —
que parte de perguntas práticas (Tenho rótulos? Quantas amostras? Quero prever
uma categoria ou uma quantidade?) e conduz a uma família de modelos.

O widget abaixo é uma versão enxuta desse mapa: responda às perguntas nos
controles e veja para qual família o fluxograma aponta. Não existe resposta
única e mágica — o mapa é um **ponto de partida sensato**, não um oráculo —, mas
ele organiza a decisão de um jeito que evita paralisia.

## PyTorch: quando você quer abrir a caixa

O scikit-learn é ótimo enquanto o modelo que você quer já existe pronto. Quando o
assunto é rede neural, porém, o valor didático está em **montar o mecanismo peça
por peça** — e é aí que entra o PyTorch (Paszke et al., 2019). Ele oferece duas
coisas essenciais:

1. **Tensores** — arranjos numéricos parecidos com os do NumPy, mas que também
   rodam em GPU;
2. **Diferenciação automática** (*autograd*) — o PyTorch registra as operações que
   você faz e calcula sozinho as derivadas de que o treino precisa.

Treinar uma rede é, no fundo, minimizar uma função de custo ajustando parâmetros.
O método é o **gradiente descendente**: a cada passo, empurramos cada parâmetro na
direção que mais reduz o custo. A atualização de um parâmetro é

$$
\boldsymbol\theta \;\leftarrow\; \boldsymbol\theta \;-\; \eta \, \nabla_{\!\boldsymbol\theta}\, \mathcal{L}(\boldsymbol\theta)
$$

Onde:

- $\boldsymbol\theta$ são os **parâmetros** do modelo (os pesos que ele aprende);
- $\mathcal{L}(\boldsymbol\theta)$ é a **função de custo** (o quão erradas estão as
  predições; queremos o menor valor possível);
- $\nabla_{\!\boldsymbol\theta}\,\mathcal{L}$ é o **gradiente** do custo — o vetor
  que aponta na direção de maior *aumento* do custo (por isso o sinal de menos:
  andamos no sentido contrário, o de maior *redução*);
- $\eta$ (a letra grega *eta*) é a **taxa de aprendizado** — o tamanho do passo. Se
  for grande demais, o treino oscila e diverge; pequena demais, ele engatinha. É o
  hiperparâmetro que você mais vai ajustar.

O papel do *autograd* é justamente entregar $\nabla_{\!\boldsymbol\theta}\,\mathcal{L}$
sem que você precise derivar nada à mão. No notebook, todo treino de rede segue
sempre o mesmo ciclo de quatro passos, escrito de forma explícita:

$$
\underbrace{\hat{\mathbf{y}} = f(\mathbf{X};\boldsymbol\theta)}_{\text{1. forward}}
\;\rightarrow\;
\underbrace{\mathcal{L}(\hat{\mathbf{y}}, \mathbf{y})}_{\text{2. custo}}
\;\rightarrow\;
\underbrace{\nabla_{\!\boldsymbol\theta}\mathcal{L}}_{\text{3. backward}}
\;\rightarrow\;
\underbrace{\boldsymbol\theta \leftarrow \boldsymbol\theta - \eta\nabla_{\!\boldsymbol\theta}\mathcal{L}}_{\text{4. passo}}
$$

Repetir esse ciclo muitas vezes **é** treinar uma rede. Todo o resto são detalhes.

## NumPy e pandas: o chão de fábrica

Por baixo dos dois, estão as ferramentas com que já trabalhamos: o **NumPy**
(Harris et al., 2020), que dá os arranjos numéricos vetorizados, e o **pandas**
(McKinney, 2010), que dá as tabelas (`DataFrame`) para carregar e limpar dados. O
scikit-learn aceita tanto arranjos NumPy quanto `DataFrame`s do pandas — é comum
carregar e tratar os dados no pandas e passar direto para o `.fit`.

## No notebook

O notebook `09_ferramentas_sklearn_pytorch.ipynb` percorre, lado a lado, os dois
caminhos sobre o mesmo problema (classificar flores do conjunto Iris):

- o **caminho scikit-learn**: `fit` / `predict` / `score` em três linhas, e o
  mesmo código trocando o modelo;
- o **caminho PyTorch**: um tensor, um exemplo de *autograd* calculando uma
  derivada, e um laço de treino explícito (forward → custo → backward → passo)
  cujo custo cai a cada época, tudo visualizado com Plotly.

Abra-o no Colab e rode célula a célula — os dois estilos ficam lado a lado para
você sentir quando usar cada um.

## Exercícios

**1.** Um colega tem uma tabela com 200 pacientes, 8 exames por paciente (as
colunas) e uma coluna dizendo se cada um tem ou não a doença. Ele quer prever a
doença em novos pacientes. Quais são $n$, $p$ e o que é $\mathbf{y}$? Qual método
do scikit-learn ele chama primeiro?

<details><summary>Ver solução</summary>

- $n = 200$ (exemplos/pacientes, as linhas);
- $p = 8$ (características/exames, as colunas);
- $\mathbf{y}$ é o vetor com o rótulo de cada paciente (tem/não tem a doença) —
  como é uma **categoria**, trata-se de um problema de **classificação**;
- o primeiro método é `.fit(X, y)`, para ajustar o modelo aos dados de treino.
  Depois, `.predict(X_novos)` para os novos pacientes.

</details>

**2.** No gradiente descendente, o que acontece se a taxa de aprendizado $\eta$ for
grande demais? E pequena demais?

<details><summary>Ver solução</summary>

- $\eta$ **grande demais**: cada passo ultrapassa o mínimo, o custo oscila e pode
  até **divergir** (crescer sem parar) em vez de cair.
- $\eta$ **pequena demais**: cada passo reduz muito pouco o custo, e o treino fica
  **lento**, precisando de um número enorme de épocas para chegar perto do mínimo.
- Na prática, busca-se um valor intermediário — e é comum reduzir $\eta$ ao longo
  do treino. É o hiperparâmetro que mais recompensa o ajuste cuidadoso.

</details>

## Referências

- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825–2830.
- Paszke, A. et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library*. Advances in Neural Information Processing Systems (NeurIPS), 32.
- Harris, C. R. et al. (2020). *Array programming with NumPy*. Nature, 585, 357–362.
- McKinney, W. (2010). *Data Structures for Statistical Computing in Python*. Proceedings of the 9th Python in Science Conference.
- James, G., Witten, D., Hastie, T., Tibshirani, R. & Taylor, J. (2023). *An Introduction to Statistical Learning with Applications in Python*. Springer. Livro aberto: https://www.statlearning.com/
- Documentação do scikit-learn: *Choosing the right estimator* — https://scikit-learn.org/1.3/tutorial/machine_learning_map/
