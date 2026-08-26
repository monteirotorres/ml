# -*- coding: utf-8 -*-
"""Gera os notebooks .ipynb dos capítulos do curso no PADRÃO NOVO.

Convenções (decididas com o autor):
- Plotly para os gráficos dos notebooks (Colab tem tudo pré-instalado);
- código explícito e simples: SEM `def`, laços à mostra, nomes em português;
- scikit-learn para os modelos clássicos e PyTorch para redes;
- toda fórmula explicada; toda métrica com linha de base;
- os notebooks são gerados SEM saídas (para rodar no Colab).

Cada notebook tem uma função `nb_*`. Rode:
    python tools/gerar_notebooks_curso.py
"""

from pathlib import Path
import nbformat as nbf

BASE = Path(__file__).parent.parent

# Preâmbulo de estilo — abre todos os notebooks do padrão novo.
PREAMBULO = '''# bibliotecas base
import numpy as np
import pandas as pd

# Plotly para os gráficos (interativos e leves no Colab)
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
pio.templates.default = "simple_white"

# paleta do curso (a mesma do site)
AZUL, VERMELHO, VERDE = "#3266ad", "#c0392b", "#1a7a4a"
TINTA, SUAVE = "#1c1e15", "#6b7050"

# reprodutibilidade: uma única semente para tudo que é aleatório
SEMENTE = 42
np.random.seed(SEMENTE)'''


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(t):
    return nbf.v4.new_code_cell(t)


def escrever(nb, caminho):
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    (BASE / caminho).write_text(nbf.writes(nb), encoding="utf-8")
    print("OK", caminho, "|", len(nb.cells), "celulas")


# ══════════════════════════════════════════════════════════════════════════
def nb_ferramentas():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# As ferramentas: scikit-learn e PyTorch\n\n"
           "**Objetivo:** conhecer, sobre o mesmo problema (classificar flores do "
           "conjunto Iris), os dois estilos que usaremos no curso — a interface "
           "única `fit`/`predict`/`score` do **scikit-learn** e o laço de treino "
           "explícito do **PyTorch**. No fim, você sente quando usar cada um."),
        code(PREAMBULO),

        md("## 1. scikit-learn: a mesma interface para tudo\n\n"
           "Todo modelo do scikit-learn é um objeto com três métodos: `.fit(X, y)` "
           "ajusta, `.predict(X)` prevê e `.score(X, y)` mede. Os dados entram "
           "sempre como uma matriz `X` de forma `(n, p)` — `n` exemplos nas linhas, "
           "`p` características nas colunas — e um vetor-alvo `y` de tamanho `n`."),
        code('from sklearn.datasets import load_iris\n'
             'from sklearn.model_selection import train_test_split\n\n'
             'iris = load_iris(as_frame=True)\n'
             'X, y = iris.data, iris.target\n'
             'print("X tem forma (n, p) =", X.shape)\n'
             'print("classes:", list(iris.target_names))\n'
             'X.head()'),
        code('# separa treino e teste — avaliamos sempre em dados NÃO vistos\n'
             'X_treino, X_teste, y_treino, y_teste = train_test_split(\n'
             '    X, y, test_size=0.3, random_state=SEMENTE, stratify=y)\n'
             'print("treino:", X_treino.shape[0], "| teste:", X_teste.shape[0])'),

        md("### O ciclo mínimo em três linhas\n\n"
           "Ajustar, prever e medir. Repare que a acurácia é comparada a uma linha "
           "de base ingênua: chutar sempre a classe mais comum acertaria só ~1/3, "
           "já que as três espécies do Iris são igualmente frequentes."),
        code('from sklearn.ensemble import RandomForestClassifier\n\n'
             'modelo = RandomForestClassifier(n_estimators=200, random_state=SEMENTE)\n'
             'modelo.fit(X_treino, y_treino)                 # 1. ajusta\n'
             'previsto = modelo.predict(X_teste)             # 2. prevê\n'
             'acuracia = modelo.score(X_teste, y_teste)      # 3. mede\n'
             'print("acuracia da floresta:", round(acuracia, 3))\n'
             'print("linha de base (classe mais comum):", round(1/3, 3))'),

        md("### Trocar de modelo é trocar uma linha\n\n"
           "A uniformidade da interface deixa a comparação justa: o mesmo laço "
           "treina e mede modelos bem diferentes. Nada de `def` — percorremos um "
           "dicionário de modelos com um laço à mostra."),
        code('from sklearn.linear_model import LogisticRegression\n'
             'from sklearn.neighbors import KNeighborsClassifier\n'
             'from sklearn.tree import DecisionTreeClassifier\n\n'
             'modelos = {\n'
             '    "Regressao logistica": LogisticRegression(max_iter=1000),\n'
             '    "k-vizinhos (k=5)":    KNeighborsClassifier(n_neighbors=5),\n'
             '    "Arvore de decisao":   DecisionTreeClassifier(random_state=SEMENTE),\n'
             '    "Floresta aleatoria":  RandomForestClassifier(n_estimators=200, random_state=SEMENTE),\n'
             '}\n\n'
             'nomes = []\n'
             'acuracias = []\n'
             'for nome, m in modelos.items():\n'
             '    m.fit(X_treino, y_treino)\n'
             '    ac = m.score(X_teste, y_teste)\n'
             '    nomes.append(nome)\n'
             '    acuracias.append(ac)\n'
             '    print(nome.ljust(22), "acuracia =", round(ac, 3))'),
        code('# um gráfico de barras compara os quatro (Plotly)\n'
             'figura = go.Figure(go.Bar(\n'
             '    x=acuracias, y=nomes, orientation="h",\n'
             '    marker_color=AZUL, text=[round(a, 3) for a in acuracias],\n'
             '    textposition="outside"))\n'
             'figura.add_vline(x=1/3, line_dash="dash", line_color=VERMELHO,\n'
             '                 annotation_text="linha de base")\n'
             'figura.update_layout(title="Acuracia no teste — mesma interface, modelos diferentes",\n'
             '                     xaxis_title="acuracia", xaxis_range=[0, 1.05],\n'
             '                     height=340, margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()'),

        md("### Qual modelo escolher?\n\n"
           "A documentação do scikit-learn traz um fluxograma famoso, o "
           "[**mapa de estimadores**](https://scikit-learn.org/1.3/tutorial/machine_learning_map/), "
           "que parte de perguntas práticas (tenho rótulos? quantas amostras? quero "
           "prever categoria ou quantidade?) e conduz a uma família de modelos. Vale "
           "abrir e seguir os ramos — o widget da página do curso é uma versão enxuta dele."),

        md("## 2. PyTorch: abrindo a caixa\n\n"
           "Quando o assunto é rede neural, o valor está em montar o mecanismo peça "
           "por peça. O PyTorch dá **tensores** (arranjos que também rodam em GPU) e "
           "**autograd** (derivadas calculadas sozinho). Primeiro, o autograd em um "
           "exemplo mínimo: a derivada de $f(x)=x^2$ em $x=3$ vale $2x = 6$."),
        code('import torch\n\n'
             'x = torch.tensor(3.0, requires_grad=True)   # queremos a derivada em relacao a x\n'
             'f = x ** 2                                   # forward: calcula f(x)\n'
             'f.backward()                                 # backward: calcula df/dx\n'
             'print("f(3) =", f.item())\n'
             'print("df/dx em x=3 (autograd) =", x.grad.item(), "| esperado 2*x =", 6.0)'),

        md("### O laço de treino, explícito\n\n"
           "Treinar é repetir quatro passos: **forward** (prever), **custo** (medir o "
           "erro), **backward** (autograd calcula o gradiente $\\nabla_\\theta\\mathcal{L}$) "
           "e **passo** (atualizar $\\theta \\leftarrow \\theta - \\eta\\,\\nabla_\\theta\\mathcal{L}$, "
           "com $\\eta$ = taxa de aprendizado). Vamos treinar uma regressão logística "
           "de duas classes (setosa × versicolor) escrevendo o laço à mão."),
        code('# usa so as duas primeiras classes, para uma fronteira binaria simples\n'
             'mascara = y.values < 2\n'
             'X2 = X.values[mascara]\n'
             'y2 = y.values[mascara].astype("float32")\n\n'
             '# padroniza as colunas (media 0, desvio 1) — ajuda o gradiente\n'
             'X2 = (X2 - X2.mean(axis=0)) / X2.std(axis=0)\n\n'
             '# converte para tensores do PyTorch\n'
             'entradas = torch.tensor(X2, dtype=torch.float32)\n'
             'alvos = torch.tensor(y2, dtype=torch.float32).reshape(-1, 1)\n'
             'print("entradas:", entradas.shape, "| alvos:", alvos.shape)'),
        code('# uma camada linear seguida de sigmoide = regressao logistica\n'
             'torch.manual_seed(SEMENTE)\n'
             'rede = torch.nn.Sequential(\n'
             '    torch.nn.Linear(entradas.shape[1], 1),\n'
             '    torch.nn.Sigmoid())\n'
             'custo_fn = torch.nn.BCELoss()                        # entropia cruzada binaria\n'
             'otimizador = torch.optim.SGD(rede.parameters(), lr=0.5)   # lr = taxa de aprendizado (eta)\n\n'
             'historico_custo = []\n'
             'for epoca in range(200):\n'
             '    previsto = rede(entradas)                # 1. forward\n'
             '    custo = custo_fn(previsto, alvos)        # 2. custo\n'
             '    otimizador.zero_grad()\n'
             '    custo.backward()                         # 3. backward (autograd)\n'
             '    otimizador.step()                        # 4. passo\n'
             '    historico_custo.append(custo.item())\n\n'
             'print("custo inicial:", round(historico_custo[0], 4))\n'
             'print("custo final:  ", round(historico_custo[-1], 4))'),
        code('# a curva de custo deve cair a cada epoca (Plotly)\n'
             'figura = go.Figure(go.Scatter(\n'
             '    x=list(range(1, len(historico_custo) + 1)), y=historico_custo,\n'
             '    mode="lines", line=dict(color=VERDE, width=2)))\n'
             'figura.update_layout(title="Custo do treino a cada epoca (gradiente descendente)",\n'
             '                     xaxis_title="epoca", yaxis_title="custo (BCE)",\n'
             '                     height=340, margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()'),
        code('# acuracia da rede treinada: previsao > 0.5 vira classe 1\n'
             'with torch.no_grad():\n'
             '    proba = rede(entradas).numpy().ravel()\n'
             'classe_prevista = (proba > 0.5).astype(int)\n'
             'acuracia_rede = (classe_prevista == y2.astype(int)).mean()\n'
             'print("acuracia da rede (nas duas classes):", round(float(acuracia_rede), 3))'),

        md("## 3. Quando usar cada um\n\n"
           "- **scikit-learn** — o modelo que você quer já existe pronto: regressão, "
           "árvores, florestas, k-means, PCA. Interface única, comparação fácil.\n"
           "- **PyTorch** — você quer **construir** a rede e ver o treino por dentro, "
           "com controle total sobre cada passo do gradiente.\n\n"
           "No resto do curso, os modelos clássicos vêm do scikit-learn; as redes "
           "neurais (capítulo 6) vêm do PyTorch, sempre com o laço à mostra."),

        md("## Exercício\n\n"
           "No laço de treino, troque a taxa de aprendizado `lr=0.5` por `lr=0.01` e "
           "depois por `lr=5.0`, e observe a curva de custo. O que muda na **velocidade** "
           "de convergência? E por que, neste problema em particular, uma taxa alta "
           "não desestabiliza o treino?"),
        code('# @title Solução (clique para revelar)\n'
             'for taxa in [0.01, 0.5, 5.0]:\n'
             '    torch.manual_seed(SEMENTE)\n'
             '    rede_t = torch.nn.Sequential(torch.nn.Linear(entradas.shape[1], 1), torch.nn.Sigmoid())\n'
             '    oti = torch.optim.SGD(rede_t.parameters(), lr=taxa)\n'
             '    curva = []\n'
             '    for epoca in range(200):\n'
             '        c = custo_fn(rede_t(entradas), alvos)\n'
             '        oti.zero_grad(); c.backward(); oti.step()\n'
             '        curva.append(c.item())\n'
             '    print("lr =", taxa, "-> custo final =", round(curva[-1], 4))\n'
             '# lr=0.01 converge DEVAGAR (custo final ainda alto, ~0.19); lr=0.5 e\n'
             '# lr=5.0 chegam perto de 0, mais rapido quanto maior a taxa.\n'
             '# Aqui as duas classes (setosa x versicolor) sao linearmente separaveis,\n'
             '# entao a superficie de custo e bem-comportada e mesmo uma taxa alta so\n'
             '# ACELERA. A instabilidade classica de eta grande demais aparece em\n'
             '# problemas mais dificeis (vales estreitos no custo), nao neste. A licao:\n'
             '# o efeito de eta depende do problema — por isso e o hiperparametro que\n'
             '# mais se ajusta na pratica.'),
    ]
    escrever(nb, "01_fundamentos/09_ferramentas_sklearn_pytorch.ipynb")


# ══════════════════════════════════════════════════════════════════════════
def nb_reg_linear():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Regressão linear simples\n\n"
           "**Objetivo:** ajustar uma reta de duas formas — na mão, pelas fórmulas "
           "fechadas de mínimos quadrados, e com o `LinearRegression` do scikit-learn "
           "— e confirmar que dão o mesmo resultado. Depois, ler o coeficiente e "
           "olhar os resíduos."),
        code(PREAMBULO),
        md("## 1. Os dados\n\n"
           "Usamos o conjunto **diabetes** do scikit-learn e, por enquanto, um único "
           "preditor: o índice de massa corporal (`bmi`), já padronizado. O alvo `y` é "
           "uma medida de progressão da doença um ano depois."),
        code('from sklearn.datasets import load_diabetes\n\n'
             'dados = load_diabetes(as_frame=True)\n'
             'x = dados.data["bmi"].values     # um preditor\n'
             'y = dados.target.values          # alvo continuo\n'
             'print("n =", len(x), "exemplos")\n'
             'print("x[:5] =", np.round(x[:5], 3))\n'
             'print("y[:5] =", y[:5])'),
        md("## 2. Ajuste na mão\n\n"
           "As fórmulas fechadas: a inclinação é a covariância de $x$ e $y$ dividida "
           "pela variância de $x$, e o intercepto ancora a reta no ponto médio.\n\n"
           "$$\\theta_1 = \\frac{\\sum_i (x_i-\\bar{x})(y_i-\\bar{y})}{\\sum_i (x_i-\\bar{x})^2}, "
           "\\qquad \\theta_0 = \\bar{y} - \\theta_1\\bar{x}$$"),
        code('media_x = x.mean()\n'
             'media_y = y.mean()\n'
             'covariancia = ((x - media_x) * (y - media_y)).sum()\n'
             'variancia_x = ((x - media_x) ** 2).sum()\n'
             'theta1 = covariancia / variancia_x\n'
             'theta0 = media_y - theta1 * media_x\n'
             'print("theta1 (inclinacao):", round(theta1, 3))\n'
             'print("theta0 (intercepto):", round(theta0, 3))'),
        md("## 3. Ajuste com o scikit-learn\n\n"
           "A mesma conta, com a interface `fit`. O scikit-learn espera `X` em formato "
           "de matriz `(n, p)`, então damos ao vetor `x` uma segunda dimensão."),
        code('from sklearn.linear_model import LinearRegression\n\n'
             'X = x.reshape(-1, 1)                 # de (n,) para (n, 1)\n'
             'modelo = LinearRegression()\n'
             'modelo.fit(X, y)\n'
             'print("inclinacao sklearn:", round(modelo.coef_[0], 3))\n'
             'print("intercepto sklearn:", round(modelo.intercept_, 3))\n'
             'print("bate com a conta na mao?",\n'
             '      np.allclose([modelo.coef_[0], modelo.intercept_], [theta1, theta0]))'),
        md("## 4. A reta sobre os dados\n\n"
           "Os pontos e a reta ajustada. Repare que a reta passa pelo centro da nuvem."),
        code('grade_x = np.linspace(x.min(), x.max(), 100)\n'
             'reta_y = theta0 + theta1 * grade_x\n\n'
             'figura = go.Figure()\n'
             'figura.add_trace(go.Scatter(x=x, y=y, mode="markers",\n'
             '                            marker=dict(color=SUAVE, size=6, opacity=0.6), name="dados"))\n'
             'figura.add_trace(go.Scatter(x=grade_x, y=reta_y, mode="lines",\n'
             '                            line=dict(color=AZUL, width=3), name="reta ajustada"))\n'
             'figura.update_layout(title="Regressao linear simples: bmi -> progressao",\n'
             '                     xaxis_title="bmi (padronizado)", yaxis_title="progressao (y)",\n'
             '                     height=380, margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()'),
        md("## 5. Resíduos\n\n"
           "O resíduo é a distância vertical entre o ponto e a reta. Um gráfico de "
           "resíduos contra a predição deve parecer uma faixa **sem padrão** em torno "
           "do zero; qualquer curvatura sistemática seria sinal de que a relação não é "
           "bem uma reta."),
        code('previsto = modelo.predict(X)\n'
             'residuos = y - previsto\n\n'
             'figura = go.Figure(go.Scatter(x=previsto, y=residuos, mode="markers",\n'
             '                              marker=dict(color=VERMELHO, size=6, opacity=0.6)))\n'
             'figura.add_hline(y=0, line_dash="dash", line_color=TINTA)\n'
             'figura.update_layout(title="Residuos vs predicao",\n'
             '                     xaxis_title="valor previsto", yaxis_title="residuo (y - y_previsto)",\n'
             '                     height=340, margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()\n'
             'print("media dos residuos (deve ser ~0):", round(residuos.mean(), 6))'),
        md("## Exercício\n\n"
           "O `bmi` está padronizado, então o intercepto é o `y` previsto no `bmi` "
           "médio. Reajuste usando outro preditor (por exemplo `bp`, pressão) e compare "
           "a inclinação. Qual preditor sozinho explica melhor o alvo?"),
        code('# @title Solução (clique para revelar)\n'
             'for nome in ["bmi", "bp", "s5"]:\n'
             '    xi = dados.data[nome].values.reshape(-1, 1)\n'
             '    m = LinearRegression().fit(xi, y)\n'
             '    r2 = m.score(xi, y)\n'
             '    print(nome, "-> inclinacao", round(m.coef_[0], 1), "| R2 =", round(r2, 3))\n'
             '# o maior R2 indica o preditor que, sozinho, mais explica a variacao de y.'),
    ]
    escrever(nb, "02_regressao/01_regressao_linear.ipynb")


# ══════════════════════════════════════════════════════════════════════════
def nb_reg_multipla():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Regressão linear múltipla\n\n"
           "**Objetivo:** ajustar um hiperplano a vários preditores de uma vez — pela "
           "equação normal com NumPy e com o scikit-learn — e diagnosticar "
           "colinearidade com o VIF, tudo com laços à mostra."),
        code(PREAMBULO),
        md("## 1. Todos os preditores\n\n"
           "O conjunto diabetes tem 10 preditores clínicos. Vamos usá-los todos."),
        code('from sklearn.datasets import load_diabetes\n\n'
             'dados = load_diabetes(as_frame=True)\n'
             'X = dados.data.values      # (n, 10)\n'
             'y = dados.target.values\n'
             'nomes = list(dados.data.columns)\n'
             'print("X tem forma (n, p) =", X.shape)\n'
             'print("preditores:", nomes)'),
        md("## 2. A equação normal\n\n"
           "Empilhando uma coluna de $1$ para o intercepto, a solução de mínimos "
           "quadrados é $\\boldsymbol\\theta = (\\mathbf{X}^\\top\\mathbf{X})^{-1}"
           "\\mathbf{X}^\\top\\mathbf{y}$. Na prática resolvemos o sistema linear "
           "(mais estável do que inverter a matriz)."),
        code('n = X.shape[0]\n'
             'X_intercepto = np.hstack([np.ones((n, 1)), X])   # coluna de 1s + preditores\n'
             'A = X_intercepto.T @ X_intercepto\n'
             'b = X_intercepto.T @ y\n'
             'theta = np.linalg.solve(A, b)                    # resolve A theta = b\n'
             'print("theta0 (intercepto):", round(theta[0], 2))\n'
             'for nome, coef in zip(nomes, theta[1:]):\n'
             '    print(nome.ljust(5), "->", round(coef, 1))'),
        md("## 3. Conferindo com o scikit-learn"),
        code('from sklearn.linear_model import LinearRegression\n\n'
             'modelo = LinearRegression()\n'
             'modelo.fit(X, y)\n'
             'print("intercepto bate?", np.isclose(modelo.intercept_, theta[0]))\n'
             'print("coeficientes batem?", np.allclose(modelo.coef_, theta[1:]))\n'
             'print("R2 do modelo completo:", round(modelo.score(X, y), 3))'),
        md("## 4. Colinearidade: o VIF de cada preditor\n\n"
           "Para cada preditor, regredimos ele contra **todos os outros** e medimos o "
           "$R^2$; o VIF é $1/(1-R^2)$. VIF alto = preditor redundante. Fazemos num "
           "laço explícito, um preditor por vez."),
        code('vifs = []\n'
             'for j in range(X.shape[1]):\n'
             '    outros = [c for c in range(X.shape[1]) if c != j]\n'
             '    reg = LinearRegression().fit(X[:, outros], X[:, j])\n'
             '    r2_j = reg.score(X[:, outros], X[:, j])\n'
             '    vif_j = 1.0 / (1.0 - r2_j)\n'
             '    vifs.append(vif_j)\n'
             '    print(nomes[j].ljust(5), "R2 =", round(r2_j, 3), "| VIF =", round(vif_j, 2))'),
        code('figura = go.Figure(go.Bar(x=nomes, y=vifs, marker_color=VERMELHO,\n'
             '                          text=[round(v, 1) for v in vifs], textposition="outside"))\n'
             'figura.add_hline(y=5, line_dash="dash", line_color=TINTA,\n'
             '                 annotation_text="atencao acima de 5")\n'
             'figura.update_layout(title="Fator de inflacao da variancia (VIF) por preditor",\n'
             '                     yaxis_title="VIF", height=360, margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()\n'
             'print("preditores com VIF > 5 (candidatos a redundancia):",\n'
             '      [nomes[j] for j in range(len(vifs)) if vifs[j] > 5])'),
        md("## Exercício\n\n"
           "Os preditores `s1`–`s6` são medidas de sangue, algumas muito "
           "correlacionadas. Olhe os VIF: quais formam o par mais redundante? O que "
           "aconteceria com a interpretação dos coeficientes deles?"),
        code('# @title Solução (clique para revelar)\n'
             'ordem = np.argsort(vifs)[::-1]\n'
             'print("preditores por VIF (maior primeiro):")\n'
             'for j in ordem[:4]:\n'
             '    print("  ", nomes[j], "VIF =", round(vifs[j], 1))\n'
             '# Os de VIF mais alto carregam quase a mesma informacao; seus coeficientes\n'
             '# individuais ficam instaveis (mudam muito com pequenas variacoes nos dados),\n'
             '# entao nao devem ser lidos isoladamente. A regularizacao (proximo topico) ajuda.'),
    ]
    escrever(nb, "02_regressao/02_regressao_multipla.ipynb")


# ══════════════════════════════════════════════════════════════════════════
def nb_reg_polinomial():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Regressão polinomial\n\n"
           "**Objetivo:** ver o compromisso viés–variância ao vivo. Ajustamos "
           "polinômios de graus crescentes a dados com curvatura conhecida e "
           "acompanhamos o erro de treino e o de teste — um cai sempre, o outro faz "
           "um U."),
        code(PREAMBULO),
        md("## 1. Dados com curvatura\n\n"
           "Geramos pontos de uma função verdadeira suave (uma senoide) mais ruído, e "
           "separamos treino e teste. Só o treino é usado para ajustar; o teste mede "
           "generalização."),
        code('from sklearn.model_selection import train_test_split\n\n'
             'x = np.linspace(0, 1, 60)\n'
             'y_verdadeiro = np.sin(2 * np.pi * x)                 # sinal real\n'
             'y = y_verdadeiro + np.random.normal(0, 0.25, size=x.shape)  # + ruido\n\n'
             'x_treino, x_teste, y_treino, y_teste = train_test_split(\n'
             '    x, y, test_size=0.4, random_state=SEMENTE)\n'
             'print("treino:", len(x_treino), "| teste:", len(x_teste))'),
        md("## 2. Ajustar vários graus\n\n"
           "Para cada grau, um `Pipeline` cria as potências de $x$ com "
           "`PolynomialFeatures` e ajusta uma regressão linear sobre elas. Guardamos o "
           "erro de treino e de teste de cada grau, num laço à mostra."),
        code('from sklearn.preprocessing import PolynomialFeatures\n'
             'from sklearn.linear_model import LinearRegression\n'
             'from sklearn.pipeline import make_pipeline\n'
             'from sklearn.metrics import mean_squared_error\n\n'
             'graus = [1, 3, 5, 9, 15]\n'
             'erro_treino = []\n'
             'erro_teste = []\n'
             'modelos = {}\n'
             'for d in graus:\n'
             '    modelo = make_pipeline(PolynomialFeatures(d), LinearRegression())\n'
             '    modelo.fit(x_treino.reshape(-1, 1), y_treino)\n'
             '    mse_tr = mean_squared_error(y_treino, modelo.predict(x_treino.reshape(-1, 1)))\n'
             '    mse_te = mean_squared_error(y_teste, modelo.predict(x_teste.reshape(-1, 1)))\n'
             '    erro_treino.append(mse_tr)\n'
             '    erro_teste.append(mse_te)\n'
             '    modelos[d] = modelo\n'
             '    print("grau", str(d).rjust(2), "| MSE treino", round(mse_tr, 3), "| MSE teste", round(mse_te, 3))'),
        md("## 3. As curvas ajustadas\n\n"
           "Grau baixo é rígido (subajuste); grau alto serpenteia atrás do ruído "
           "(sobreajuste). A curva verdadeira está em preto."),
        code('grade = np.linspace(0, 1, 200).reshape(-1, 1)\n'
             'figura = go.Figure()\n'
             'figura.add_trace(go.Scatter(x=x_treino, y=y_treino, mode="markers",\n'
             '                            marker=dict(color=SUAVE, size=6), name="treino"))\n'
             'figura.add_trace(go.Scatter(x=grade.ravel(), y=np.sin(2*np.pi*grade.ravel()),\n'
             '                            mode="lines", line=dict(color=TINTA, dash="dash"), name="verdade"))\n'
             'for d in [1, 5, 15]:\n'
             '    figura.add_trace(go.Scatter(x=grade.ravel(), y=modelos[d].predict(grade),\n'
             '                                mode="lines", name=f"grau {d}"))\n'
             'figura.update_layout(title="Ajustes de diferentes graus", yaxis_range=[-1.8, 1.8],\n'
             '                     height=400, margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()'),
        md("## 4. Erro de treino × erro de teste\n\n"
           "O gráfico que resume tudo: o erro de treino desce sempre, o de teste "
           "desenha um U. O fundo do U marca o grau que melhor generaliza."),
        code('melhor = graus[int(np.argmin(erro_teste))]\n'
             'figura = go.Figure()\n'
             'figura.add_trace(go.Scatter(x=graus, y=erro_treino, mode="lines+markers",\n'
             '                            line=dict(color=AZUL), name="erro de treino"))\n'
             'figura.add_trace(go.Scatter(x=graus, y=erro_teste, mode="lines+markers",\n'
             '                            line=dict(color=VERMELHO), name="erro de teste"))\n'
             'figura.add_vline(x=melhor, line_dash="dash", line_color=VERDE,\n'
             '                 annotation_text=f"melhor grau = {melhor}")\n'
             'figura.update_layout(title="Selecao do grau: vies-variancia",\n'
             '                     xaxis_title="grau do polinomio", yaxis_title="MSE",\n'
             '                     height=380, margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()\n'
             'print("grau que minimiza o erro de teste:", melhor)'),
        md("## Exercício\n\n"
           "Aumente o ruído (troque `0.25` por `0.6` na geração de `y`) e rode de novo. "
           "O grau ótimo tende a subir ou a descer com mais ruído? Por quê?"),
        md("<details><summary>Ver resposta</summary>\n\n"
           "Com mais ruído, o grau ótimo tende a **descer**: como há menos sinal "
           "confiável, um modelo mais simples (mais rígido) generaliza melhor, enquanto "
           "um grau alto passa a ajustar o ruído maior e piora no teste. Mais ruído "
           "empurra a escolha para modelos menos flexíveis.\n\n</details>"),
    ]
    escrever(nb, "02_regressao/03_regressao_polinomial.ipynb")


# ══════════════════════════════════════════════════════════════════════════
def nb_regularizacao():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Regularização: Ridge e Lasso\n\n"
           "**Objetivo:** ver a penalidade em ação — traçar como os coeficientes "
           "encolhem com $\\alpha$, contrastar Ridge (encolhe) e Lasso (zera) e "
           "escolher $\\alpha$ por validação cruzada. Sempre com padronização."),
        code(PREAMBULO),
        md("## 1. Dados e padronização\n\n"
           "Conjunto diabetes de novo. Como a penalidade depende da escala dos "
           "coeficientes, **padronizamos** os preditores (média 0, desvio 1) — feito "
           "dentro de um `Pipeline` para não vazar informação do teste."),
        code('from sklearn.datasets import load_diabetes\n'
             'from sklearn.preprocessing import StandardScaler\n'
             'from sklearn.pipeline import make_pipeline\n'
             'from sklearn.linear_model import Ridge, Lasso\n\n'
             'dados = load_diabetes(as_frame=True)\n'
             'X = dados.data.values\n'
             'y = dados.target.values\n'
             'nomes = list(dados.data.columns)\n'
             'print("X:", X.shape, "| preditores:", nomes)'),
        md("## 2. O caminho de regularização\n\n"
           "Para uma faixa de $\\alpha$ (em escala logarítmica), ajustamos Ridge e "
           "Lasso e guardamos os coeficientes. Dois laços à mostra: um por método, um "
           "por valor de $\\alpha$."),
        code('alphas = np.logspace(-2, 2, 40)\n\n'
             'caminho_ridge = []\n'
             'caminho_lasso = []\n'
             'for a in alphas:\n'
             '    ridge = make_pipeline(StandardScaler(), Ridge(alpha=a)).fit(X, y)\n'
             '    lasso = make_pipeline(StandardScaler(), Lasso(alpha=a, max_iter=10000)).fit(X, y)\n'
             '    caminho_ridge.append(ridge.named_steps["ridge"].coef_)\n'
             '    caminho_lasso.append(lasso.named_steps["lasso"].coef_)\n'
             'caminho_ridge = np.array(caminho_ridge)\n'
             'caminho_lasso = np.array(caminho_lasso)\n'
             'print("formato do caminho (n_alphas, n_preditores):", caminho_ridge.shape)'),
        code('from plotly.subplots import make_subplots\n\n'
             'figura = make_subplots(rows=1, cols=2, subplot_titles=("Ridge (l2)", "Lasso (l1)"))\n'
             'for j in range(X.shape[1]):\n'
             '    figura.add_trace(go.Scatter(x=alphas, y=caminho_ridge[:, j], mode="lines",\n'
             '                                name=nomes[j], showlegend=False), row=1, col=1)\n'
             '    figura.add_trace(go.Scatter(x=alphas, y=caminho_lasso[:, j], mode="lines",\n'
             '                                name=nomes[j]), row=1, col=2)\n'
             'figura.update_xaxes(type="log", title_text="alpha (log)")\n'
             'figura.update_yaxes(title_text="coeficiente", row=1, col=1)\n'
             'figura.update_layout(title="Caminho de regularizacao: Ridge encolhe, Lasso zera",\n'
             '                     height=400, margin=dict(l=10, r=10, t=60, b=10))\n'
             'figura.show()'),
        md("## 3. Lasso zera coeficientes\n\n"
           "Contamos, para cada $\\alpha$, quantos coeficientes o Lasso mantém "
           "diferentes de zero. Quanto maior $\\alpha$, mais esparso o modelo."),
        code('n_diferentes_de_zero = (np.abs(caminho_lasso) > 1e-6).sum(axis=1)\n'
             'figura = go.Figure(go.Scatter(x=alphas, y=n_diferentes_de_zero, mode="lines+markers",\n'
             '                              line=dict(color=VERDE)))\n'
             'figura.update_xaxes(type="log", title_text="alpha (log)")\n'
             'figura.update_layout(title="Lasso: numero de coeficientes != 0 vs alpha",\n'
             '                     yaxis_title="coeficientes ativos", height=340,\n'
             '                     margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()'),
        md("## 4. Escolhendo α por validação cruzada\n\n"
           "`RidgeCV` e `LassoCV` testam vários $\\alpha$ por validação cruzada e "
           "ficam com o que minimiza o erro em dados não vistos — nada de escolher no "
           "olho."),
        code('from sklearn.linear_model import RidgeCV, LassoCV\n\n'
             'ridge_cv = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)).fit(X, y)\n'
             'lasso_cv = make_pipeline(StandardScaler(), LassoCV(alphas=alphas, max_iter=10000)).fit(X, y)\n'
             'print("melhor alpha Ridge:", round(ridge_cv.named_steps["ridgecv"].alpha_, 3))\n'
             'print("melhor alpha Lasso:", round(lasso_cv.named_steps["lassocv"].alpha_, 3))\n'
             'coef_lasso = lasso_cv.named_steps["lassocv"].coef_\n'
             'print("preditores mantidos pelo Lasso:",\n'
             '      [nomes[j] for j in range(len(coef_lasso)) if abs(coef_lasso[j]) > 1e-6])'),
        md("## Exercício\n\n"
           "No caminho do Lasso, qual preditor é o **último** a ser zerado quando "
           "$\\alpha$ cresce? O que isso sugere sobre a importância dele?"),
        code('# @title Solução (clique para revelar)\n'
             'a_grande = alphas[-8]\n'
             'lasso_forte = make_pipeline(StandardScaler(), Lasso(alpha=a_grande, max_iter=10000)).fit(X, y)\n'
             'coef = lasso_forte.named_steps["lasso"].coef_\n'
             'sobreviventes = [(nomes[j], round(coef[j], 1)) for j in range(len(coef)) if abs(coef[j]) > 1e-6]\n'
             'print("com alpha alto, sobrevivem:", sobreviventes)\n'
             '# O ultimo a resistir e o preditor mais robustamente associado ao alvo:\n'
             '# o Lasso o considera o mais informativo, o ultimo do qual abre mao.'),
    ]
    escrever(nb, "02_regressao/04_regularizacao.ipynb")


# ══════════════════════════════════════════════════════════════════════════
def nb_reg_logistica():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Regressão logística\n\n"
           "**Objetivo:** treinar um classificador binário, ler os coeficientes como "
           "razões de chance, desenhar a fronteira de decisão e ver que 'logística' e "
           "'uma camada linear + sigmoide' são a mesma coisa — o mesmo modelo, no "
           "scikit-learn e num laço explícito de PyTorch."),
        code(PREAMBULO),
        md("## 1. Dados: diagnóstico de câncer de mama\n\n"
           "Conjunto **breast cancer** do scikit-learn: 30 medidas de imagens de "
           "núcleos celulares, alvo binário (maligno = 1, benigno = 0)."),
        code('from sklearn.datasets import load_breast_cancer\n'
             'from sklearn.model_selection import train_test_split\n'
             'from sklearn.preprocessing import StandardScaler\n\n'
             'dados = load_breast_cancer(as_frame=True)\n'
             'X = dados.data.values\n'
             'y = dados.target.values\n'
             '# nesta base, 0 = maligno e 1 = benigno; invertemos para 1 = maligno (o positivo de interesse)\n'
             'y = 1 - y\n'
             'X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,\n'
             '                                          random_state=SEMENTE, stratify=y)\n'
             'escala = StandardScaler().fit(X_tr)\n'
             'X_tr = escala.transform(X_tr)\n'
             'X_te = escala.transform(X_te)\n'
             'print("treino:", X_tr.shape, "| malignos no treino:", int(y_tr.sum()))'),
        md("## 2. Ajuste com o scikit-learn e razões de chance\n\n"
           "O coeficiente $\\theta_j$ vira a razão de chances $e^{\\theta_j}$: quanto "
           "as chances de ser maligno mudam ao aumentar aquele preditor em um "
           "desvio-padrão (os dados estão padronizados)."),
        code('from sklearn.linear_model import LogisticRegression\n'
             'from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix\n\n'
             'modelo = LogisticRegression(max_iter=5000)\n'
             'modelo.fit(X_tr, y_tr)\n'
             'proba = modelo.predict_proba(X_te)[:, 1]\n'
             'previsto = (proba > 0.5).astype(int)\n'
             'print("acuracia:", round(accuracy_score(y_te, previsto), 3))\n'
             'print("AUC:", round(roc_auc_score(y_te, proba), 3))\n\n'
             'razao_chances = np.exp(modelo.coef_[0])\n'
             'ordem = np.argsort(razao_chances)[::-1]\n'
             'print("\\ntres maiores fatores de risco (razao de chances):")\n'
             'for j in ordem[:3]:\n'
             '    print("  ", dados.data.columns[j], "->", round(razao_chances[j], 2))'),
        md("## 3. A fronteira de decisão em 2D\n\n"
           "Para visualizar, treinamos de novo com só dois preditores e pintamos a "
           "região que o modelo chama de maligna."),
        code('col_a, col_b = 20, 27   # dois preditores (raio e concavidade "piores")\n'
             'X2 = X_tr[:, [col_a, col_b]]\n'
             'modelo2 = LogisticRegression(max_iter=5000).fit(X2, y_tr)\n\n'
             'passo = 0.05\n'
             'gx, gy = np.meshgrid(np.arange(X2[:, 0].min()-1, X2[:, 0].max()+1, passo),\n'
             '                     np.arange(X2[:, 1].min()-1, X2[:, 1].max()+1, passo))\n'
             'grade = np.c_[gx.ravel(), gy.ravel()]\n'
             'zz = modelo2.predict_proba(grade)[:, 1].reshape(gx.shape)\n\n'
             'figura = go.Figure()\n'
             'figura.add_trace(go.Contour(x=gx[0], y=gy[:, 0], z=zz, showscale=False,\n'
             '                            colorscale=[[0, "#dce7f4"], [1, "#f6dedb"]], opacity=0.7,\n'
             '                            contours=dict(start=0.5, end=0.5, size=1, coloring="lines")))\n'
             'figura.add_trace(go.Scatter(x=X2[y_tr==0, 0], y=X2[y_tr==0, 1], mode="markers",\n'
             '                            marker=dict(color=AZUL, size=6), name="benigno"))\n'
             'figura.add_trace(go.Scatter(x=X2[y_tr==1, 0], y=X2[y_tr==1, 1], mode="markers",\n'
             '                            marker=dict(color=VERMELHO, size=6), name="maligno"))\n'
             'figura.update_layout(title="Fronteira de decisao (dois preditores)",\n'
             '                     height=420, margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()'),
        md("## 4. A mesma logística, à mão em PyTorch\n\n"
           "Uma camada linear seguida de sigmoide **é** uma regressão logística. "
           "Treinamos com o laço explícito de sempre (forward → custo → backward → "
           "passo) e vemos a acurácia bater com a do scikit-learn."),
        code('import torch\n\n'
             'entradas = torch.tensor(X_tr, dtype=torch.float32)\n'
             'alvos = torch.tensor(y_tr, dtype=torch.float32).reshape(-1, 1)\n\n'
             'torch.manual_seed(SEMENTE)\n'
             'rede = torch.nn.Sequential(torch.nn.Linear(X_tr.shape[1], 1), torch.nn.Sigmoid())\n'
             'custo_fn = torch.nn.BCELoss()\n'
             'otimizador = torch.optim.Adam(rede.parameters(), lr=0.05)\n\n'
             'historico = []\n'
             'for epoca in range(300):\n'
             '    previsto_t = rede(entradas)          # forward\n'
             '    custo = custo_fn(previsto_t, alvos)   # custo\n'
             '    otimizador.zero_grad()\n'
             '    custo.backward()                      # backward\n'
             '    otimizador.step()                     # passo\n'
             '    historico.append(custo.item())\n\n'
             'with torch.no_grad():\n'
             '    proba_te = rede(torch.tensor(X_te, dtype=torch.float32)).numpy().ravel()\n'
             'acc_torch = ((proba_te > 0.5).astype(int) == y_te).mean()\n'
             'print("acuracia da rede (PyTorch):", round(float(acc_torch), 3))\n'
             'print("acuracia do sklearn:      ", round(accuracy_score(y_te, previsto), 3))'),
        code('figura = go.Figure(go.Scatter(y=historico, mode="lines", line=dict(color=VERDE)))\n'
             'figura.update_layout(title="Custo (log-loss) do treino em PyTorch",\n'
             '                     xaxis_title="epoca", yaxis_title="BCE", height=320,\n'
             '                     margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()'),
        md("## Exercício\n\n"
           "A matriz de confusão abaixo separa os erros em falsos positivos e falsos "
           "negativos. Em rastreio de câncer, qual dos dois é mais grave? Como o limiar "
           "de 0,5 poderia ser ajustado para reduzi-lo?"),
        code('# @title Solução (clique para revelar)\n'
             'mc = confusion_matrix(y_te, previsto)\n'
             'print("matriz de confusao [linhas=verdade, colunas=previsto]:")\n'
             'print(mc)\n'
             'print("\\nfalsos negativos (maligno dito benigno):", mc[1, 0])\n'
             '# Em rastreio, o FALSO NEGATIVO (deixar passar um maligno) costuma ser o mais\n'
             '# grave. Baixar o limiar (ex.: prever maligno se proba > 0.3) captura mais\n'
             '# malignos, ao custo de mais falsos positivos (mais biopsias de confirmacao).'),
    ]
    escrever(nb, "02_regressao/05_regressao_logistica.ipynb")


CONSTRUTORES = [
    nb_ferramentas,
    nb_reg_linear,
    nb_reg_multipla,
    nb_reg_polinomial,
    nb_regularizacao,
    nb_reg_logistica,
]

if __name__ == "__main__":
    for construir in CONSTRUTORES:
        construir()
