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


# ══════════════════════════════════════════════════════════════════════════
def nb_knn():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# k-Vizinhos mais próximos (k-NN)\n\n"
           "**Objetivo:** classificar o Iris com k-NN, ver como o número de vizinhos "
           "$k$ controla o compromisso viés–variância (a curva de acurácia por $k$) e "
           "desenhar a fronteira de decisão para dois valores de $k$."),
        code(PREAMBULO),
        md("## 1. Dados e a importância de padronizar\n\n"
           "O k-NN mede **distâncias**, então padronizamos as características (dentro "
           "de um `Pipeline`, para não vazar o teste). Usamos o Iris completo."),
        code('from sklearn.datasets import load_iris\n'
             'from sklearn.model_selection import cross_val_score\n'
             'from sklearn.preprocessing import StandardScaler\n'
             'from sklearn.neighbors import KNeighborsClassifier\n'
             'from sklearn.pipeline import make_pipeline\n\n'
             'iris = load_iris()\n'
             'X, y = iris.data, iris.target\n'
             'print("X:", X.shape, "| classes:", list(iris.target_names))'),
        md("## 2. A curva de acurácia por k\n\n"
           "Para cada $k$ (ímpar, para evitar empates), medimos a acurácia por "
           "validação cruzada de 5 dobras. Um laço explícito, um $k$ por vez."),
        code('ks = list(range(1, 40, 2))\n'
             'acuracias = []\n'
             'for k in ks:\n'
             '    modelo = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))\n'
             '    ac = cross_val_score(modelo, X, y, cv=5).mean()\n'
             '    acuracias.append(ac)\n'
             '    print("k =", str(k).rjust(2), "-> acuracia CV =", round(ac, 3))\n'
             'melhor_k = ks[int(np.argmax(acuracias))]\n'
             'print("melhor k:", melhor_k)'),
        code('figura = go.Figure(go.Scatter(x=ks, y=acuracias, mode="lines+markers",\n'
             '                              line=dict(color=AZUL)))\n'
             'figura.add_vline(x=melhor_k, line_dash="dash", line_color=VERDE,\n'
             '                 annotation_text=f"melhor k = {melhor_k}")\n'
             'figura.update_layout(title="Acuracia (validacao cruzada) vs numero de vizinhos",\n'
             '                     xaxis_title="k", yaxis_title="acuracia", height=360,\n'
             '                     margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()'),
        md("## 3. A fronteira de decisão muda com k\n\n"
           "Usando dois preditores (comprimento e largura da pétala), pintamos a "
           "região prevista para $k=1$ (recortada) e $k=25$ (suave)."),
        code('X2 = X[:, 2:4]   # petala: comprimento e largura\n'
             'passo = 0.02\n'
             'gx, gy = np.meshgrid(np.arange(X2[:, 0].min()-0.5, X2[:, 0].max()+0.5, passo),\n'
             '                     np.arange(X2[:, 1].min()-0.5, X2[:, 1].max()+0.5, passo))\n'
             'grade = np.c_[gx.ravel(), gy.ravel()]\n\n'
             'from plotly.subplots import make_subplots\n'
             'figura = make_subplots(rows=1, cols=2, subplot_titles=("k = 1", "k = 25"))\n'
             'coluna = 1\n'
             'for k in [1, 25]:\n'
             '    modelo = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k)).fit(X2, y)\n'
             '    zz = modelo.predict(grade).reshape(gx.shape)\n'
             '    figura.add_trace(go.Heatmap(x=gx[0], y=gy[:, 0], z=zz, showscale=False,\n'
             '                                colorscale="Blugrn", opacity=0.35), row=1, col=coluna)\n'
             '    figura.add_trace(go.Scatter(x=X2[:, 0], y=X2[:, 1], mode="markers",\n'
             '                                marker=dict(color=y, colorscale="Blugrn", size=6,\n'
             '                                            line=dict(width=0.5, color="white")),\n'
             '                                showlegend=False), row=1, col=coluna)\n'
             '    coluna += 1\n'
             'figura.update_layout(title="Fronteira de decisao: k=1 recortada, k=25 suave",\n'
             '                     height=380, margin=dict(l=10, r=10, t=60, b=10))\n'
             'figura.show()'),
        md("## Exercício\n\n"
           "Refaça a curva de acurácia **sem** o `StandardScaler` (troque o pipeline por "
           "um `KNeighborsClassifier` puro). No Iris o efeito é pequeno porque as escalas "
           "são parecidas — mas em que tipo de dado a padronização seria decisiva?"),
        md("<details><summary>Ver resposta</summary>\n\n"
           "Seria decisiva quando os preditores têm **escalas muito diferentes** — por "
           "exemplo, colesterol em mg/dL (centenas) misturado com uma proporção (0 a 1). "
           "Sem padronizar, a variável de valores grandes domina a distância euclidiana e "
           "as outras são praticamente ignoradas. No Iris, as quatro medidas estão todas "
           "em centímetros, então o impacto é pequeno.\n\n</details>"),
    ]
    escrever(nb, "03_classificacao/01_knn.ipynb")


# ══════════════════════════════════════════════════════════════════════════
def nb_arvores():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Árvores de decisão\n\n"
           "**Objetivo:** treinar uma árvore no Iris, ler as perguntas que ela aprendeu, "
           "ver a profundidade controlar o overfitting (a curva treino × validação) e "
           "desenhar a fronteira retangular."),
        code(PREAMBULO),
        code('from sklearn.datasets import load_iris\n'
             'from sklearn.tree import DecisionTreeClassifier, export_text\n'
             'from sklearn.model_selection import train_test_split, cross_val_score\n\n'
             'iris = load_iris()\n'
             'X, y = iris.data, iris.target\n'
             'X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,\n'
             '                                          random_state=SEMENTE, stratify=y)\n'
             'print("treino:", X_tr.shape[0], "| teste:", X_te.shape[0])'),
        md("## 1. As perguntas que a árvore aprendeu\n\n"
           "Uma árvore rasa (profundidade 3) já classifica bem o Iris. O `export_text` "
           "imprime a árvore como um fluxograma de perguntas — leitura direta, sem "
           "caixa-preta."),
        code('arvore = DecisionTreeClassifier(max_depth=3, random_state=SEMENTE)\n'
             'arvore.fit(X_tr, y_tr)\n'
             'print("acuracia no teste:", round(arvore.score(X_te, y_te), 3))\n'
             'print()\n'
             'print(export_text(arvore, feature_names=list(iris.feature_names)))'),
        md("## 2. Profundidade × overfitting\n\n"
           "Para cada profundidade, comparamos a acurácia no **treino** com a de "
           "**validação cruzada**. O treino sobe sempre em direção a 100%; a validação "
           "faz um U — sinal de que árvores fundas memorizam."),
        code('profundidades = list(range(1, 12))\n'
             'acc_treino = []\n'
             'acc_validacao = []\n'
             'for d in profundidades:\n'
             '    modelo = DecisionTreeClassifier(max_depth=d, random_state=SEMENTE)\n'
             '    modelo.fit(X_tr, y_tr)\n'
             '    acc_treino.append(modelo.score(X_tr, y_tr))\n'
             '    acc_validacao.append(cross_val_score(modelo, X_tr, y_tr, cv=5).mean())\n'
             '    print("prof", str(d).rjust(2), "| treino", round(acc_treino[-1], 3),\n'
             '          "| validacao", round(acc_validacao[-1], 3))'),
        code('figura = go.Figure()\n'
             'figura.add_trace(go.Scatter(x=profundidades, y=acc_treino, mode="lines+markers",\n'
             '                            line=dict(color=AZUL), name="treino"))\n'
             'figura.add_trace(go.Scatter(x=profundidades, y=acc_validacao, mode="lines+markers",\n'
             '                            line=dict(color=VERMELHO), name="validacao (CV)"))\n'
             'figura.update_layout(title="Arvore: profundidade x acuracia",\n'
             '                     xaxis_title="max_depth", yaxis_title="acuracia", height=360,\n'
             '                     margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()'),
        md("## 3. A fronteira retangular\n\n"
           "Com dois preditores, a fronteira da árvore é feita de **retângulos** (cortes "
           "paralelos aos eixos) — a assinatura visual do modelo."),
        code('X2 = X[:, 2:4]\n'
             'arvore2 = DecisionTreeClassifier(max_depth=4, random_state=SEMENTE).fit(X2, y)\n'
             'passo = 0.02\n'
             'gx, gy = np.meshgrid(np.arange(X2[:, 0].min()-0.5, X2[:, 0].max()+0.5, passo),\n'
             '                     np.arange(X2[:, 1].min()-0.5, X2[:, 1].max()+0.5, passo))\n'
             'zz = arvore2.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape)\n\n'
             'figura = go.Figure()\n'
             'figura.add_trace(go.Heatmap(x=gx[0], y=gy[:, 0], z=zz, showscale=False,\n'
             '                            colorscale="Blugrn", opacity=0.35))\n'
             'figura.add_trace(go.Scatter(x=X2[:, 0], y=X2[:, 1], mode="markers",\n'
             '                            marker=dict(color=y, colorscale="Blugrn", size=6,\n'
             '                                        line=dict(width=0.5, color="white"))))\n'
             'figura.update_layout(title="Fronteira retangular da arvore (2 preditores)",\n'
             '                     height=400, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)\n'
             'figura.show()'),
        md("## Exercício\n\n"
           "Pela curva do item 2, qual `max_depth` você escolheria para este problema? "
           "Justifique com base na acurácia de validação, não na de treino."),
        md("<details><summary>Ver resposta</summary>\n\n"
           "Escolhe-se a menor profundidade que já atinge o platô da acurácia de "
           "**validação** — no Iris, tipicamente `max_depth` entre 3 e 4. Ir além disso "
           "só aumenta a acurácia de treino (rumo a 100%) sem ganho na validação, o que "
           "é overfitting: mais complexidade sem mais generalização. A regra é preferir o "
           "modelo mais simples que empata no topo da validação.\n\n</details>"),
    ]
    escrever(nb, "03_classificacao/02_arvores_decisao.ipynb")


# ══════════════════════════════════════════════════════════════════════════
def nb_naive_bayes():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Naive Bayes\n\n"
           "**Objetivo:** usar o Naive Bayes gaussiano no Iris (vendo as curvas normais "
           "por classe) e o multinomial num pequeno problema de texto, notando a "
           "velocidade. Probabilidade condicional explícita, sem caixa-preta."),
        code(PREAMBULO),
        md("## 1. Naive Bayes gaussiano no Iris\n\n"
           "O `GaussianNB` estima, para cada classe, a **média** e o **desvio** de cada "
           "característica — supondo uma normal. Depois combina pela regra de Bayes."),
        code('from sklearn.datasets import load_iris\n'
             'from sklearn.naive_bayes import GaussianNB\n'
             'from sklearn.model_selection import train_test_split\n\n'
             'iris = load_iris()\n'
             'X, y = iris.data, iris.target\n'
             'X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,\n'
             '                                          random_state=SEMENTE, stratify=y)\n'
             'modelo = GaussianNB().fit(X_tr, y_tr)\n'
             'print("acuracia no teste:", round(modelo.score(X_te, y_te), 3))\n'
             'print("medias por classe (uma linha por classe):")\n'
             'print(np.round(modelo.theta_, 2))'),
        md("## 2. As curvas normais estimadas\n\n"
           "Para uma característica (comprimento da pétala), desenhamos a normal que o "
           "modelo ajustou a cada classe. Onde as curvas se cruzam fica a zona de "
           "confusão entre as espécies."),
        code('caracteristica = 2   # comprimento da petala\n'
             'grade = np.linspace(X[:, caracteristica].min()-0.5, X[:, caracteristica].max()+0.5, 300)\n\n'
             'figura = go.Figure()\n'
             'cores = [AZUL, VERDE, VERMELHO]\n'
             'for c in range(3):\n'
             '    media = modelo.theta_[c, caracteristica]\n'
             '    desvio = np.sqrt(modelo.var_[c, caracteristica])\n'
             '    densidade = np.exp(-0.5 * ((grade - media) / desvio) ** 2) / (desvio * np.sqrt(2*np.pi))\n'
             '    figura.add_trace(go.Scatter(x=grade, y=densidade, mode="lines",\n'
             '                                line=dict(color=cores[c]), name=iris.target_names[c]))\n'
             'figura.update_layout(title="Curvas normais por classe (comprimento da petala)",\n'
             '                     xaxis_title="cm", yaxis_title="densidade", height=360,\n'
             '                     margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()'),
        md("## 3. Naive Bayes multinomial para texto\n\n"
           "Em texto, cada documento vira um vetor de **contagens de palavras** "
           "(`CountVectorizer`) e o `MultinomialNB` classifica. Usamos um mini-corpus "
           "embutido (frases de esporte × tecnologia) — sem downloads."),
        code('from sklearn.feature_extraction.text import CountVectorizer\n'
             'from sklearn.naive_bayes import MultinomialNB\n\n'
             'textos = [\n'
             '    "o time venceu o jogo e marcou tres gols",\n'
             '    "o jogador foi campeao do torneio de futebol",\n'
             '    "a torcida comemorou a vitoria no estadio",\n'
             '    "o atacante marcou o gol da vitoria no jogo",\n'
             '    "o novo processador e a placa de video sao rapidos",\n'
             '    "o software roda no computador com muita memoria",\n'
             '    "o aplicativo usa inteligencia artificial e dados",\n'
             '    "a rede neural treina no processador da maquina",\n'
             ']\n'
             'rotulos = ["esporte", "esporte", "esporte", "esporte",\n'
             '           "tecnologia", "tecnologia", "tecnologia", "tecnologia"]\n\n'
             'vetorizador = CountVectorizer()\n'
             'X_texto = vetorizador.fit_transform(textos)\n'
             'classificador = MultinomialNB().fit(X_texto, rotulos)\n'
             'print("vocabulario (", len(vetorizador.get_feature_names_out()), "palavras)")\n\n'
             'novas = ["o time marcou um gol no jogo", "a maquina usa inteligencia artificial"]\n'
             'previsto = classificador.predict(vetorizador.transform(novas))\n'
             'for frase, classe in zip(novas, previsto):\n'
             '    print("->", classe.ljust(11), "|", frase)'),
        md("## Exercício\n\n"
           "O modelo classificou as duas frases novas corretamente mesmo sem nunca ter "
           "visto exatamente essas combinações de palavras. Por que a suposição de "
           "independência entre palavras não estragou a classificação?"),
        md("<details><summary>Ver resposta</summary>\n\n"
           "Porque para **decidir a classe** basta que a soma (em log) das evidências de "
           "cada palavra aponte para o lado certo — não é preciso que a probabilidade "
           "estimada seja exata. Palavras como \"gol\"/\"jogo\" empurram forte para "
           "\"esporte\" e \"inteligencia\"/\"maquina\" para \"tecnologia\"; mesmo tratando "
           "as palavras como independentes (o que ignora que \"inteligencia\" e "
           "\"artificial\" andam juntas), a **ordem** entre as duas classes se mantém e a "
           "decisão acerta.\n\n</details>"),
    ]
    escrever(nb, "03_classificacao/03_naive_bayes.ipynb")


# ══════════════════════════════════════════════════════════════════════════
def nb_svm():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Máquinas de vetores de suporte (SVM)\n\n"
           "**Objetivo:** ver a margem e os vetores de suporte de uma SVM linear e, no "
           "caso clássico dos dados 'em círculos', ver o **truque do kernel** (RBF) "
           "resolver o que nenhuma reta consegue."),
        code(PREAMBULO),
        md("## 1. Dados não separáveis por uma reta\n\n"
           "O `make_circles` gera um anel de uma classe em volta de um núcleo da outra: "
           "não há reta que os separe."),
        code('from sklearn.datasets import make_circles\n'
             'from sklearn.svm import SVC\n\n'
             'X, y = make_circles(n_samples=300, factor=0.4, noise=0.12, random_state=SEMENTE)\n'
             'print("X:", X.shape, "| classes:", np.unique(y))\n\n'
             'figura = go.Figure(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",\n'
             '                              marker=dict(color=y, colorscale="Bluered", size=6)))\n'
             'figura.update_layout(title="Dados em circulos: nenhuma reta separa",\n'
             '                     height=380, margin=dict(l=10, r=10, t=50, b=10), showlegend=False)\n'
             'figura.show()'),
        md("## 2. SVM linear × SVM com kernel RBF\n\n"
           "Treinamos as duas e comparamos a acurácia. A linear fracassa; a RBF, que "
           "projeta os dados implicitamente para onde eles ficam separáveis, acerta."),
        code('svm_linear = SVC(kernel="linear").fit(X, y)\n'
             'svm_rbf = SVC(kernel="rbf", C=1.0, gamma=1.0).fit(X, y)\n'
             'print("acuracia SVM linear:", round(svm_linear.score(X, y), 3))\n'
             'print("acuracia SVM RBF:   ", round(svm_rbf.score(X, y), 3))\n'
             'print("vetores de suporte da RBF:", svm_rbf.support_vectors_.shape[0], "de", len(X))'),
        md("## 3. As fronteiras lado a lado\n\n"
           "Pintamos a região prevista por cada modelo. A da SVM linear é um semiplano "
           "(inútil aqui); a da RBF é um anel que envolve o núcleo. Os pontos maiores são "
           "os **vetores de suporte** — os únicos que definem a fronteira."),
        code('from plotly.subplots import make_subplots\n'
             'passo = 0.03\n'
             'gx, gy = np.meshgrid(np.arange(X[:, 0].min()-0.3, X[:, 0].max()+0.3, passo),\n'
             '                     np.arange(X[:, 1].min()-0.3, X[:, 1].max()+0.3, passo))\n'
             'grade = np.c_[gx.ravel(), gy.ravel()]\n\n'
             'figura = make_subplots(rows=1, cols=2, subplot_titles=("SVM linear", "SVM RBF"))\n'
             'coluna = 1\n'
             'for modelo in [svm_linear, svm_rbf]:\n'
             '    zz = modelo.predict(grade).reshape(gx.shape)\n'
             '    figura.add_trace(go.Heatmap(x=gx[0], y=gy[:, 0], z=zz, showscale=False,\n'
             '                                colorscale="Bluered", opacity=0.3), row=1, col=coluna)\n'
             '    figura.add_trace(go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",\n'
             '                                marker=dict(color=y, colorscale="Bluered", size=5),\n'
             '                                showlegend=False), row=1, col=coluna)\n'
             '    sv = modelo.support_vectors_\n'
             '    figura.add_trace(go.Scatter(x=sv[:, 0], y=sv[:, 1], mode="markers",\n'
             '                                marker=dict(color="rgba(0,0,0,0)", size=11,\n'
             '                                            line=dict(width=1.5, color=VERDE)),\n'
             '                                showlegend=False), row=1, col=coluna)\n'
             '    coluna += 1\n'
             'figura.update_layout(title="Fronteiras: linear falha, RBF resolve (vetores de suporte em verde)",\n'
             '                     height=400, margin=dict(l=10, r=10, t=60, b=10))\n'
             'figura.show()'),
        md("## 4. O efeito de C e γ\n\n"
           "Varremos alguns valores de $\\gamma$ (mantendo $C$) e medimos a acurácia de "
           "validação cruzada. $\\gamma$ grande demais memoriza (overfitting); pequeno "
           "demais suaviza a ponto de perder o anel."),
        code('from sklearn.model_selection import cross_val_score\n\n'
             'for gamma in [0.1, 1.0, 10.0, 100.0]:\n'
             '    modelo = SVC(kernel="rbf", C=1.0, gamma=gamma)\n'
             '    ac = cross_val_score(modelo, X, y, cv=5).mean()\n'
             '    ac_treino = modelo.fit(X, y).score(X, y)\n'
             '    print("gamma =", str(gamma).rjust(5),\n'
             '          "| treino", round(ac_treino, 3), "| validacao CV", round(ac, 3))'),
        md("## Exercício\n\n"
           "Na varredura acima, para qual $\\gamma$ a acurácia de treino é altíssima mas "
           "a de validação cai? Como esse padrão se chama e como corrigi-lo?"),
        md("<details><summary>Ver resposta</summary>\n\n"
           "Para o $\\gamma$ **mais alto** (100): o treino fica quase perfeito enquanto a "
           "validação cruzada cai. É **overfitting** — cada ponto vira uma bolha da sua "
           "classe, memorizando o treino sem capturar o anel real. Corrige-se **reduzindo "
           "$\\gamma$** (e escolhendo $C$ e $\\gamma$ por validação cruzada, por exemplo "
           "com `GridSearchCV`).\n\n</details>"),
    ]
    escrever(nb, "03_classificacao/04_svm.ipynb")


# ══════════════════════════════════════════════════════════════════════════
def nb_random_forest():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Bagging e Random Forests\n\n"
           "**Objetivo:** ver a acurácia crescer e estabilizar com o número de árvores, "
           "comparar a floresta com uma árvore isolada e ler a importância das "
           "variáveis."),
        code(PREAMBULO),
        code('from sklearn.datasets import load_breast_cancer\n'
             'from sklearn.model_selection import train_test_split\n'
             'from sklearn.ensemble import RandomForestClassifier\n'
             'from sklearn.tree import DecisionTreeClassifier\n\n'
             'dados = load_breast_cancer()\n'
             'X, y = dados.data, dados.target\n'
             'X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,\n'
             '                                          random_state=SEMENTE, stratify=y)\n'
             'print("treino:", X_tr.shape, "| preditores:", X.shape[1])'),
        md("## 1. Uma árvore × uma floresta\n\n"
           "A árvore isolada tem variância alta; a floresta (bagging + aleatoriedade "
           "nas variáveis) estabiliza."),
        code('arvore = DecisionTreeClassifier(random_state=SEMENTE).fit(X_tr, y_tr)\n'
             'floresta = RandomForestClassifier(n_estimators=300, random_state=SEMENTE).fit(X_tr, y_tr)\n'
             'print("acuracia da arvore isolada:", round(arvore.score(X_te, y_te), 3))\n'
             'print("acuracia da floresta (300):", round(floresta.score(X_te, y_te), 3))'),
        md("## 2. Acurácia × número de árvores\n\n"
           "A acurácia sobe rápido com as primeiras árvores e depois estabiliza — "
           "acrescentar árvores **não** causa overfitting (só custa tempo)."),
        code('numeros = [1, 2, 5, 10, 25, 50, 100, 200, 400]\n'
             'acuracias = []\n'
             'for n in numeros:\n'
             '    modelo = RandomForestClassifier(n_estimators=n, random_state=SEMENTE)\n'
             '    modelo.fit(X_tr, y_tr)\n'
             '    acuracias.append(modelo.score(X_te, y_te))\n'
             '    print("arvores", str(n).rjust(3), "-> acuracia", round(acuracias[-1], 3))\n\n'
             'figura = go.Figure(go.Scatter(x=numeros, y=acuracias, mode="lines+markers",\n'
             '                              line=dict(color=AZUL)))\n'
             'figura.update_layout(title="Acuracia vs numero de arvores (bagging estabiliza)",\n'
             '                     xaxis_title="n_estimators", yaxis_title="acuracia", height=340,\n'
             '                     margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()'),
        md("## 3. Importância das variáveis\n\n"
           "A floresta entrega, de graça, um ranking de quais preditores mais reduzem "
           "a impureza. Mostramos os dez maiores."),
        code('importancias = floresta.feature_importances_\n'
             'ordem = np.argsort(importancias)[::-1][:10]\n'
             'nomes_top = [dados.feature_names[j] for j in ordem]\n'
             'valores_top = importancias[ordem]\n'
             'for nome, val in zip(nomes_top, valores_top):\n'
             '    print(nome.ljust(24), round(val, 3))\n\n'
             'figura = go.Figure(go.Bar(x=valores_top[::-1], y=nomes_top[::-1], orientation="h",\n'
             '                          marker_color=VERDE))\n'
             'figura.update_layout(title="Importancia das variaveis (top 10)",\n'
             '                     xaxis_title="importancia", height=380,\n'
             '                     margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()'),
        md("## Exercício\n\n"
           "Compare a acurácia da árvore isolada com a da floresta de 300 árvores. O "
           "ganho vem de reduzir viés ou variância? Justifique."),
        md("<details><summary>Ver resposta</summary>\n\n"
           "Vem de reduzir **variância**. A árvore isolada e a floresta têm o mesmo tipo "
           "de modelo de base (árvores), com viés parecido; o que a floresta faz é tirar "
           "a **média** de muitas árvores decorrelacionadas, o que reduz a variância do "
           "conjunto — daí a acurácia mais alta e estável. Não muda a natureza do viés, "
           "estabiliza a variância.\n\n</details>"),
    ]
    escrever(nb, "04_ensembles/01_random_forest.ipynb")


# ══════════════════════════════════════════════════════════════════════════
def nb_gradient_boosting():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Gradient Boosting\n\n"
           "**Objetivo:** construir o boosting **à mão** num problema 1D — árvores rasas "
           "ajustadas aos resíduos, um estágio por vez — e depois ver o efeito da taxa "
           "de aprendizado e do número de estágios com o `GradientBoostingClassifier`."),
        code(PREAMBULO),
        md("## 1. Boosting à mão: ajustar os resíduos\n\n"
           "Começamos prevendo a média de `y`. A cada estágio, uma árvore rasa (um "
           "toco, `max_depth=1`) aprende o **resíduo** que sobrou, e somamos uma fração "
           "`nu` dela ao modelo. Laço explícito, sem caixa-preta."),
        code('from sklearn.tree import DecisionTreeRegressor\n\n'
             'x = np.linspace(0, 1, 60)\n'
             'y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.15, size=x.shape)\n'
             'X = x.reshape(-1, 1)\n\n'
             'nu = 0.3\n'
             'F = np.full_like(y, y.mean())      # previsao inicial: a media\n'
             'arvores = []\n'
             'erros = []\n'
             'for estagio in range(40):\n'
             '    residuo = y - F                # o que ainda falta\n'
             '    toco = DecisionTreeRegressor(max_depth=1).fit(X, residuo)\n'
             '    F = F + nu * toco.predict(X)    # incorpora uma fracao da nova arvore\n'
             '    arvores.append(toco)\n'
             '    erros.append(np.mean((y - F) ** 2))\n'
             'print("erro (MSE) apos 1 estagio: ", round(erros[0], 3))\n'
             'print("erro (MSE) apos 40 estagios:", round(erros[-1], 3))'),
        md("## 2. O modelo tomando forma\n\n"
           "Mostramos a previsão acumulada após 1, 5 e 40 estágios: de quase uma reta a "
           "uma boa aproximação da curva verdadeira."),
        code('grade = np.linspace(0, 1, 200).reshape(-1, 1)\n'
             'figura = go.Figure()\n'
             'figura.add_trace(go.Scatter(x=x, y=y, mode="markers",\n'
             '                            marker=dict(color=SUAVE, size=5), name="dados"))\n'
             'figura.add_trace(go.Scatter(x=grade.ravel(), y=np.sin(2*np.pi*grade.ravel()),\n'
             '                            mode="lines", line=dict(color=TINTA, dash="dash"), name="verdade"))\n'
             'for n_estagios in [1, 5, 40]:\n'
             '    pred = np.full(grade.shape[0], y.mean())\n'
             '    for toco in arvores[:n_estagios]:\n'
             '        pred = pred + nu * toco.predict(grade)\n'
             '    figura.add_trace(go.Scatter(x=grade.ravel(), y=pred, mode="lines",\n'
             '                                name=f"{n_estagios} estagios"))\n'
             'figura.update_layout(title="Boosting a mao: o modelo se aproxima estagio a estagio",\n'
             '                     height=400, margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()'),
        md("## 3. Taxa de aprendizado e número de estágios\n\n"
           "Agora com o `GradientBoostingClassifier` do scikit-learn, num problema de "
           "classificação. O `staged_predict` dá a previsão após cada estágio, então "
           "vemos o erro de treino e de teste em função do número de árvores, para duas "
           "taxas de aprendizado."),
        code('from sklearn.datasets import load_breast_cancer\n'
             'from sklearn.model_selection import train_test_split\n'
             'from sklearn.ensemble import GradientBoostingClassifier\n\n'
             'dados = load_breast_cancer()\n'
             'X_tr, X_te, y_tr, y_te = train_test_split(dados.data, dados.target,\n'
             '                                          test_size=0.3, random_state=SEMENTE, stratify=dados.target)\n\n'
             'figura = go.Figure()\n'
             'for taxa, cor in [(0.1, AZUL), (1.0, VERMELHO)]:\n'
             '    modelo = GradientBoostingClassifier(n_estimators=200, learning_rate=taxa,\n'
             '                                        max_depth=2, random_state=SEMENTE)\n'
             '    modelo.fit(X_tr, y_tr)\n'
             '    erro_teste = []\n'
             '    for previsto in modelo.staged_predict(X_te):\n'
             '        erro_teste.append(1 - np.mean(previsto == y_te))\n'
             '    figura.add_trace(go.Scatter(y=erro_teste, mode="lines",\n'
             '                                line=dict(color=cor), name=f"taxa {taxa}"))\n'
             '    print("taxa", taxa, "-> menor erro de teste:", round(min(erro_teste), 3),\n'
             '          "no estagio", int(np.argmin(erro_teste)) + 1)\n'
             'figura.update_layout(title="Erro de teste vs numero de estagios, por taxa de aprendizado",\n'
             '                     xaxis_title="estagios", yaxis_title="erro de teste", height=360,\n'
             '                     margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()'),
        md("## Exercício\n\n"
           "Na figura do item 3, a taxa 1.0 atinge o menor erro mais cedo, mas costuma "
           "ficar mais irregular; a taxa 0.1 desce devagar e suave. Qual você usaria em "
           "produção e por quê?"),
        md("<details><summary>Ver resposta</summary>\n\n"
           "Em geral a **taxa menor (0,1)**, com mais estágios e *early stopping*. Passos "
           "pequenos regularizam (o *shrinkage*), dando uma curva de erro mais suave e "
           "estável e, tipicamente, melhor generalização. A taxa 1,0 chega rápido mas é "
           "sensível a ruído e pode passar do ponto. Troca-se um pouco de tempo de treino "
           "por robustez.\n\n</details>"),
    ]
    escrever(nb, "04_ensembles/02_gradient_boosting.ipynb")


# ══════════════════════════════════════════════════════════════════════════
def nb_xgboost():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# XGBoost e LightGBM\n\n"
           "**Objetivo:** usar o `HistGradientBoostingClassifier` do scikit-learn — o "
           "mesmo estilo de boosting por histogramas do LightGBM, disponível sem instalar "
           "nada — para ver *early stopping* e regularização, e comparar (opcionalmente) "
           "com o XGBoost de verdade."),
        code(PREAMBULO),
        code('from sklearn.datasets import load_breast_cancer\n'
             'from sklearn.model_selection import train_test_split\n'
             'from sklearn.ensemble import HistGradientBoostingClassifier\n'
             'import time\n\n'
             'dados = load_breast_cancer()\n'
             'X_tr, X_te, y_tr, y_te = train_test_split(dados.data, dados.target,\n'
             '                                          test_size=0.3, random_state=SEMENTE, stratify=dados.target)\n'
             'print("treino:", X_tr.shape)'),
        md("## 1. Early stopping: quantas árvores bastam?\n\n"
           "Com `early_stopping=True`, o modelo separa uma fração de validação e **para "
           "sozinho** quando o erro de validação deixa de melhorar — em vez de fixar o "
           "número de árvores no chute."),
        code('inicio = time.time()\n'
             'modelo = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=500,\n'
             '                                        early_stopping=True, validation_fraction=0.2,\n'
             '                                        n_iter_no_change=15, random_state=SEMENTE)\n'
             'modelo.fit(X_tr, y_tr)\n'
             'print("acuracia no teste:", round(modelo.score(X_te, y_te), 3))\n'
             'print("arvores efetivamente usadas (parou sozinho):", modelo.n_iter_, "de 500 possiveis")\n'
             'print("tempo de treino:", round(time.time() - inicio, 2), "s")'),
        md("## 2. Regularização via profundidade das folhas\n\n"
           "Limitar `max_leaf_nodes` regulariza (árvores menores, menos overfitting). "
           "Varremos alguns valores e comparamos treino × teste."),
        code('for folhas in [3, 7, 15, 31, 63]:\n'
             '    m = HistGradientBoostingClassifier(learning_rate=0.1, max_iter=200,\n'
             '                                       max_leaf_nodes=folhas, random_state=SEMENTE)\n'
             '    m.fit(X_tr, y_tr)\n'
             '    print("max_leaf_nodes", str(folhas).rjust(2),\n'
             '          "| treino", round(m.score(X_tr, y_tr), 3),\n'
             '          "| teste", round(m.score(X_te, y_te), 3))'),
        md("## 3. (Opcional) XGBoost de verdade\n\n"
           "Se o `xgboost` estiver disponível (no Colab, costuma estar), a célula abaixo "
           "o treina e compara. Se não estiver, ela avisa e segue — o resto do notebook "
           "não depende dele."),
        code('try:\n'
             '    from xgboost import XGBClassifier\n'
             '    xgb = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=3,\n'
             '                        reg_lambda=1.0, eval_metric="logloss", random_state=SEMENTE)\n'
             '    xgb.fit(X_tr, y_tr)\n'
             '    print("acuracia XGBoost:", round(xgb.score(X_te, y_te), 3))\n'
             'except Exception as erro:\n'
             '    print("xgboost nao disponivel neste ambiente — pulando.")\n'
             '    print("para instalar no Colab: !pip install xgboost")\n'
             '    print("detalhe:", type(erro).__name__)'),
        md("## Exercício\n\n"
           "Na varredura do item 2, o que acontece com a diferença entre acurácia de "
           "treino e de teste conforme `max_leaf_nodes` aumenta? Como isso se relaciona "
           "com a regularização?"),
        md("<details><summary>Ver resposta</summary>\n\n"
           "Neste conjunto a acurácia de **treino** já satura em ~100% mesmo com poucas "
           "folhas; o que muda é o **teste**, melhor com folhas intermediárias (7–15) e "
           "que **cai um pouco** para árvores maiores. Ou seja, capacidade extra não "
           "ajuda o teste e pode piorá-lo (overfitting): o *gap* entre treino e teste só "
           "aumenta. Limitar as folhas é **regularizar** — árvores menores generalizam "
           "melhor, exatamente o papel do termo de penalidade do XGBoost.\n\n</details>"),
    ]
    escrever(nb, "04_ensembles/03_xgboost.ipynb")


# ══════════════════════════════════════════════════════════════════════════
def nb_stacking():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Stacking e combinação de modelos\n\n"
           "**Objetivo:** combinar três modelos diferentes por soft voting e por "
           "stacking, e comparar a acurácia da combinação com a de cada modelo isolado."),
        code(PREAMBULO),
        code('from sklearn.datasets import load_breast_cancer\n'
             'from sklearn.model_selection import train_test_split, cross_val_score\n'
             'from sklearn.linear_model import LogisticRegression\n'
             'from sklearn.svm import SVC\n'
             'from sklearn.ensemble import RandomForestClassifier\n'
             'from sklearn.preprocessing import StandardScaler\n'
             'from sklearn.pipeline import make_pipeline\n\n'
             'dados = load_breast_cancer()\n'
             'X, y = dados.data, dados.target\n'
             'X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,\n'
             '                                          random_state=SEMENTE, stratify=y)'),
        md("## 1. Três modelos de base diferentes\n\n"
           "Uma regressão logística, uma SVM (com probabilidade) e uma floresta. São "
           "modelos de naturezas distintas — a diversidade é o que faz a combinação "
           "valer a pena."),
        code('base = {\n'
             '    "Regressao logistica": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),\n'
             '    "SVM (RBF)":           make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEMENTE)),\n'
             '    "Floresta":            RandomForestClassifier(n_estimators=200, random_state=SEMENTE),\n'
             '}\n'
             'for nome, m in base.items():\n'
             '    m.fit(X_tr, y_tr)\n'
             '    print(nome.ljust(22), "acuracia:", round(m.score(X_te, y_te), 3))'),
        md("## 2. Soft voting e stacking\n\n"
           "O `VotingClassifier` (soft) faz a **média das probabilidades**. O "
           "`StackingClassifier` treina um **meta-modelo** (uma regressão logística) "
           "sobre as previsões dos três — usando previsões out-of-fold para não vazar."),
        code('from sklearn.ensemble import VotingClassifier, StackingClassifier\n\n'
             'estimadores = [\n'
             '    ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))),\n'
             '    ("svm", make_pipeline(StandardScaler(), SVC(probability=True, random_state=SEMENTE))),\n'
             '    ("rf", RandomForestClassifier(n_estimators=200, random_state=SEMENTE)),\n'
             ']\n\n'
             'votacao = VotingClassifier(estimators=estimadores, voting="soft")\n'
             'votacao.fit(X_tr, y_tr)\n'
             'print("soft voting  -> acuracia:", round(votacao.score(X_te, y_te), 3))\n\n'
             'empilhado = StackingClassifier(estimators=estimadores,\n'
             '                               final_estimator=LogisticRegression(max_iter=5000), cv=5)\n'
             'empilhado.fit(X_tr, y_tr)\n'
             'print("stacking     -> acuracia:", round(empilhado.score(X_te, y_te), 3))'),
        md("## 3. Comparação final\n\n"
           "Colocamos tudo lado a lado — e o resultado é honesto: quando um modelo de "
           "base já é muito forte (aqui, a regressão logística), a combinação fica "
           "**competitiva, mas pode não superá-lo**. O ganho do stacking aparece quando "
           "**nenhum** modelo domina sozinho; misturar não é mágica."),
        code('nomes = []\n'
             'valores = []\n'
             'for nome, m in base.items():\n'
             '    nomes.append(nome); valores.append(m.score(X_te, y_te))\n'
             'nomes += ["Soft voting", "Stacking"]\n'
             'valores += [votacao.score(X_te, y_te), empilhado.score(X_te, y_te)]\n\n'
             'cores = [SUAVE, SUAVE, SUAVE, AZUL, VERDE]\n'
             'figura = go.Figure(go.Bar(x=valores, y=nomes, orientation="h", marker_color=cores,\n'
             '                          text=[round(v, 3) for v in valores], textposition="outside"))\n'
             'figura.update_layout(title="Modelos de base x combinacoes",\n'
             '                     xaxis_title="acuracia no teste", xaxis_range=[0.9, 1.0],\n'
             '                     height=360, margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()'),
        md("## Exercício\n\n"
           "Se os três modelos de base tivessem exatamente a mesma acurácia **e** "
           "errassem sempre nos mesmos exemplos, o que aconteceria com o soft voting?"),
        md("<details><summary>Ver resposta</summary>\n\n"
           "O soft voting **não ganharia nada**: a média de probabilidades de modelos que "
           "erram nos mesmos casos continua errando nesses casos. O benefício da "
           "combinação vem da **diversidade** — modelos que falham em exemplos diferentes, "
           "de modo que a média cancela erros individuais. Sem diversidade, votar apenas "
           "reproduz os mesmos acertos e erros.\n\n</details>"),
    ]
    escrever(nb, "04_ensembles/04_stacking.ipynb")


# ══════════════════════════════════════════════════════════════════════════
def nb_ex_regressao():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Exercícios — Regressão e métricas\n\n"
           "Resolva no papel antes de abrir cada solução. As células de solução vêm "
           "recolhidas (`# @title`)."),
        code(PREAMBULO),
        md("## Exercício 1 — RMSE e R² na mão e no sklearn\n\n"
           "Regressão linear de `bmi` no `diabetes`; RMSE e R² no teste, à mão e com o "
           "scikit-learn."),
        code('# @title Solução\n'
             'from sklearn.datasets import load_diabetes\n'
             'from sklearn.model_selection import train_test_split\n'
             'from sklearn.linear_model import LinearRegression\n'
             'from sklearn.metrics import mean_squared_error, r2_score\n\n'
             'd = load_diabetes()\n'
             'X = d.data[:, [2]]   # bmi\n'
             'y = d.target\n'
             'X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=SEMENTE)\n'
             'modelo = LinearRegression().fit(X_tr, y_tr)\n'
             'previsto = modelo.predict(X_te)\n'
             '# na mao\n'
             'rmse_mao = np.sqrt(np.mean((y_te - previsto) ** 2))\n'
             'r2_mao = 1 - np.sum((y_te - previsto)**2) / np.sum((y_te - y_te.mean())**2)\n'
             'print("RMSE (mao):", round(rmse_mao, 2), "| sklearn:", round(np.sqrt(mean_squared_error(y_te, previsto)), 2))\n'
             'print("R2   (mao):", round(r2_mao, 3), "| sklearn:", round(r2_score(y_te, previsto), 3))'),
        md("## Exercício 2 — Escolher o grau do polinômio"),
        code('# @title Solução\n'
             'from sklearn.preprocessing import PolynomialFeatures\n'
             'from sklearn.pipeline import make_pipeline\n\n'
             'x = np.linspace(0, 1, 60)\n'
             'y = np.sin(2*np.pi*x) + np.random.normal(0, 0.25, size=x.shape)\n'
             'xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.4, random_state=SEMENTE)\n'
             'melhor, melhor_erro = None, 1e9\n'
             'for grau in range(1, 16):\n'
             '    m = make_pipeline(PolynomialFeatures(grau), LinearRegression()).fit(xtr.reshape(-1,1), ytr)\n'
             '    erro = mean_squared_error(yte, m.predict(xte.reshape(-1,1)))\n'
             '    if erro < melhor_erro: melhor, melhor_erro = grau, erro\n'
             'print("grau que minimiza o erro de validacao:", melhor, "| MSE:", round(melhor_erro, 3))'),
        md("## Exercício 3 — Ridge, Lasso e esparsidade"),
        code('# @title Solução\n'
             'from sklearn.preprocessing import StandardScaler\n'
             'from sklearn.linear_model import Ridge, Lasso\n\n'
             'X = d.data; y = d.target\n'
             'ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, y)\n'
             'lasso = make_pipeline(StandardScaler(), Lasso(alpha=1.0)).fit(X, y)\n'
             'nz_ridge = int(np.sum(np.abs(ridge.named_steps["ridge"].coef_) > 1e-6))\n'
             'nz_lasso = int(np.sum(np.abs(lasso.named_steps["lasso"].coef_) > 1e-6))\n'
             'print("coeficientes != 0 -> Ridge:", nz_ridge, "de 10 | Lasso:", nz_lasso, "de 10")\n'
             'print("a Ridge nao zera nenhum; o Lasso zera varios (seleciona variaveis).")'),
        md("## Exercício 4 — Interpretar um coeficiente"),
        code('# @title Solução\n'
             'modelo = make_pipeline(StandardScaler(), LinearRegression()).fit(X, y)\n'
             'coefs = modelo.named_steps["linearregression"].coef_\n'
             'j = int(np.argmax(coefs))\n'
             'print("maior coeficiente positivo:", d.feature_names[j], "=", round(coefs[j], 1))\n'
             'print("Leitura: +1 desvio-padrao em", d.feature_names[j],\n'
             '      "-> +", round(coefs[j], 1), "na progressao prevista, mantidos os demais fixos.")'),
    ]
    escrever(nb, "07_exercicios/01_regressao.ipynb")


# ══════════════════════════════════════════════════════════════════════════
def nb_ex_classificacao():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Exercícios — Classificação\n\nSoluções recolhidas em `# @title`."),
        code(PREAMBULO),
        md("## Exercício 1 — Escolher o k do k-NN"),
        code('# @title Solução\n'
             'from sklearn.datasets import load_wine\n'
             'from sklearn.preprocessing import StandardScaler\n'
             'from sklearn.neighbors import KNeighborsClassifier\n'
             'from sklearn.pipeline import make_pipeline\n'
             'from sklearn.model_selection import cross_val_score\n\n'
             'w = load_wine()\n'
             'melhor, melhor_ac = None, 0\n'
             'for k in range(1, 30, 2):\n'
             '    ac = cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(k)), w.data, w.target, cv=5).mean()\n'
             '    if ac > melhor_ac: melhor, melhor_ac = k, ac\n'
             'print("melhor k:", melhor, "| acuracia CV:", round(melhor_ac, 3))'),
        md("## Exercício 2 — Ler uma matriz de confusão"),
        code('# @title Solução\n'
             'from sklearn.datasets import load_breast_cancer\n'
             'from sklearn.model_selection import train_test_split\n'
             'from sklearn.linear_model import LogisticRegression\n'
             'from sklearn.metrics import confusion_matrix\n\n'
             'bc = load_breast_cancer()\n'
             'y = 1 - bc.target   # 1 = maligno\n'
             'X_tr, X_te, y_tr, y_te = train_test_split(bc.data, y, test_size=0.3, random_state=SEMENTE, stratify=y)\n'
             'esc = StandardScaler().fit(X_tr)\n'
             'modelo = LogisticRegression(max_iter=5000).fit(esc.transform(X_tr), y_tr)\n'
             'proba = modelo.predict_proba(esc.transform(X_te))[:, 1]\n'
             'previsto = (proba > 0.5).astype(int)\n'
             'vn, fp, fn, vp = confusion_matrix(y_te, previsto).ravel()\n'
             'print("VP", vp, "FP", fp, "FN", fn, "VN", vn)\n'
             'print("precisao:", round(vp/(vp+fp), 3), "| recall:", round(vp/(vp+fn), 3))\n'
             'print("num rastreio, o falso negativo (maligno dito benigno) e o mais grave.")'),
        md("## Exercício 3 — Ajustar o limiar de decisão"),
        code('# @title Solução\n'
             'for limiar in [0.5, 0.3, 0.2, 0.1]:\n'
             '    prev = (proba > limiar).astype(int)\n'
             '    vn, fp, fn, vp = confusion_matrix(y_te, prev).ravel()\n'
             '    rec = vp/(vp+fn); prec = vp/(vp+fp) if (vp+fp) else 0\n'
             '    print("limiar", limiar, "-> recall", round(rec, 3), "| precisao", round(prec, 3))\n'
             'print("baixar o limiar sobe o recall (pega mais malignos) e baixa a precisao.")'),
        md("## Exercício 4 — Linear × não linear (o valor do kernel)"),
        code('# @title Solução\n'
             'from sklearn.datasets import make_circles\n'
             'from sklearn.svm import SVC\n\n'
             'Xc, yc = make_circles(n_samples=300, factor=0.4, noise=0.12, random_state=SEMENTE)\n'
             'for nome, m in [("logistica", LogisticRegression()),\n'
             '                ("SVM linear", SVC(kernel="linear")),\n'
             '                ("SVM RBF", SVC(kernel="rbf", gamma=1.0))]:\n'
             '    ac = cross_val_score(m, Xc, yc, cv=5).mean()\n'
             '    print(nome.ljust(11), "acuracia CV:", round(ac, 3))\n'
             'print("so a RBF resolve: o kernel torna o anel separavel.")'),
    ]
    escrever(nb, "07_exercicios/02_classificacao.ipynb")


# ══════════════════════════════════════════════════════════════════════════
def nb_ex_validacao():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Exercícios — Validação e seleção de modelos\n\nSoluções em `# @title`."),
        code(PREAMBULO),
        md("## Exercício 1 — O vazamento da seleção de variáveis\n\n"
           "Com muitas variáveis e poucas informativas, selecionar as \"melhores\" "
           "usando **todos** os dados antes de validar vaza o teste. Comparamos com a "
           "seleção feita **dentro** do pipeline."),
        code('# @title Solução\n'
             'from sklearn.datasets import make_classification\n'
             'from sklearn.feature_selection import SelectKBest, f_classif\n'
             'from sklearn.linear_model import LogisticRegression\n'
             'from sklearn.pipeline import make_pipeline\n'
             'from sklearn.model_selection import cross_val_score\n\n'
             '# 1000 variaveis, so 5 informativas\n'
             'X, y = make_classification(n_samples=200, n_features=1000, n_informative=5,\n'
             '                           n_redundant=0, random_state=SEMENTE)\n'
             '# ERRADO: escolhe as 20 melhores usando TODOS os dados, depois valida\n'
             'X_sel = SelectKBest(f_classif, k=20).fit_transform(X, y)\n'
             'ac_errado = cross_val_score(LogisticRegression(max_iter=5000), X_sel, y, cv=5).mean()\n'
             '# CERTO: selecao DENTRO do pipeline (so no treino de cada dobra)\n'
             'ac_certo = cross_val_score(make_pipeline(SelectKBest(f_classif, k=20),\n'
             '                                         LogisticRegression(max_iter=5000)), X, y, cv=5).mean()\n'
             'print("estimativa com vazamento (selecao fora):", round(ac_errado, 3))\n'
             'print("estimativa correta (selecao no pipeline):", round(ac_certo, 3))\n'
             'print("o vazamento infla: a selecao espiou o teste ao escolher as variaveis.")'),
        md("## Exercício 2 — Grid search com validação cruzada"),
        code('# @title Solução\n'
             'from sklearn.datasets import load_wine\n'
             'from sklearn.preprocessing import StandardScaler\n'
             'from sklearn.pipeline import make_pipeline\n'
             'from sklearn.svm import SVC\n'
             'from sklearn.model_selection import GridSearchCV\n\n'
             'w = load_wine()\n'
             'grade = {"svc__C": [0.1, 1, 10], "svc__gamma": [0.001, 0.01, 0.1]}\n'
             'busca = GridSearchCV(make_pipeline(StandardScaler(), SVC()), grade, cv=5)\n'
             'busca.fit(w.data, w.target)\n'
             'print("melhores parametros:", busca.best_params_)\n'
             'print("acuracia CV do melhor:", round(busca.best_score_, 3))\n'
             'print("esse numero e otimista (max sobre muitas combinacoes); use um teste separado.")'),
        md("## Exercício 3 — Curva de aprendizado"),
        code('# @title Solução\n'
             'from sklearn.datasets import load_digits\n'
             'from sklearn.preprocessing import StandardScaler\n'
             'from sklearn.pipeline import make_pipeline\n'
             'from sklearn.linear_model import LogisticRegression\n'
             'from sklearn.model_selection import learning_curve\n\n'
             'dig = load_digits()\n'
             'tam, tr, val = learning_curve(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),\n'
             '                              dig.data, dig.target, cv=5,\n'
             '                              train_sizes=np.linspace(0.1, 1.0, 6))\n'
             'figura = go.Figure()\n'
             'figura.add_trace(go.Scatter(x=tam, y=tr.mean(axis=1), mode="lines+markers",\n'
             '                            line=dict(color=AZUL), name="treino"))\n'
             'figura.add_trace(go.Scatter(x=tam, y=val.mean(axis=1), mode="lines+markers",\n'
             '                            line=dict(color=VERMELHO), name="validacao"))\n'
             'figura.update_layout(title="Curva de aprendizado (digits)", xaxis_title="tamanho do treino",\n'
             '                     yaxis_title="acuracia", height=360, margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()\n'
             'print("lacuna grande = overfitting (mais dados ajudam); convergir baixo = underfitting.")'),
        md("## Exercício 4 — Estratificar importa"),
        code('# @title Solução\n'
             'from sklearn.model_selection import KFold, StratifiedKFold\n'
             'from sklearn.tree import DecisionTreeClassifier\n\n'
             '# alvo desbalanceado: digito 3 vs resto\n'
             'y_binario = (dig.target == 3).astype(int)\n'
             'modelo = DecisionTreeClassifier(max_depth=5, random_state=SEMENTE)\n'
             'for nome, cv in [("KFold", KFold(5, shuffle=True, random_state=SEMENTE)),\n'
             '                 ("StratifiedKFold", StratifiedKFold(5, shuffle=True, random_state=SEMENTE))]:\n'
             '    scores = cross_val_score(modelo, dig.data, y_binario, cv=cv)\n'
             '    print(nome.ljust(16), "acuracia", round(scores.mean(), 3), "| desvio", round(scores.std(), 4))\n'
             'print("o estratificado costuma ter menor desvio: dobras com proporcao de classes preservada.")'),
    ]
    escrever(nb, "07_exercicios/03_validacao.ipynb")


# ══════════════════════════════════════════════════════════════════════════
def nb_ex_clustering_pca():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Exercícios — Clustering e PCA\n\nSoluções em `# @title`."),
        code(PREAMBULO),
        md("## Exercício 1 — Cotovelo e silhueta"),
        code('# @title Solução\n'
             'from sklearn.datasets import load_wine\n'
             'from sklearn.preprocessing import StandardScaler\n'
             'from sklearn.cluster import KMeans\n'
             'from sklearn.metrics import silhouette_score\n\n'
             'w = load_wine()\n'
             'X = StandardScaler().fit_transform(w.data)\n'
             'for k in range(2, 9):\n'
             '    km = KMeans(n_clusters=k, n_init=10, random_state=SEMENTE).fit(X)\n'
             '    print("k", k, "| inercia", round(km.inertia_, 1), "| silhueta", round(silhouette_score(X, km.labels_), 3))'),
        md("## Exercício 2 — Padronizar antes de agrupar"),
        code('# @title Solução\n'
             'from sklearn.metrics import adjusted_rand_score\n\n'
             'g_cru = KMeans(n_clusters=3, n_init=10, random_state=SEMENTE).fit_predict(w.data)\n'
             'g_pad = KMeans(n_clusters=3, n_init=10, random_state=SEMENTE).fit_predict(X)\n'
             'print("ARI sem padronizar:", round(adjusted_rand_score(w.target, g_cru), 3))\n'
             'print("ARI com padronizar:", round(adjusted_rand_score(w.target, g_pad), 3))\n'
             'print("variaveis de escala grande (ex.: proline) dominam a distancia sem padronizacao.")'),
        md("## Exercício 3 — Quantas componentes reter?"),
        code('# @title Solução\n'
             'from sklearn.datasets import load_breast_cancer\n'
             'from sklearn.decomposition import PCA\n\n'
             'bc = load_breast_cancer()\n'
             'Xbc = StandardScaler().fit_transform(bc.data)\n'
             'acum = np.cumsum(PCA().fit(Xbc).explained_variance_ratio_)\n'
             'n95 = int(np.argmax(acum >= 0.95)) + 1\n'
             'print("componentes para 95% da variancia:", n95, "de 30")\n'
             'print("compressao de", round(30/n95, 1), "x perdendo so 5% da variacao.")'),
        md("## Exercício 4 — PCA antes do t-SNE"),
        code('# @title Solução\n'
             'from sklearn.datasets import load_digits\n'
             'from sklearn.manifold import TSNE\n'
             'import time\n\n'
             'dig = load_digits()\n'
             'rng = np.random.RandomState(SEMENTE)\n'
             'sel = rng.choice(len(dig.data), 500, replace=False)\n'
             'Xd = dig.data[sel]\n'
             't0 = time.time()\n'
             'TSNE(n_components=2, init="random", random_state=SEMENTE).fit_transform(Xd)\n'
             'direto = time.time() - t0\n'
             't0 = time.time()\n'
             'X20 = PCA(n_components=20).fit_transform(Xd)\n'
             'TSNE(n_components=2, init="random", random_state=SEMENTE).fit_transform(X20)\n'
             'com_pca = time.time() - t0\n'
             'print("t-SNE direto (64 dims):", round(direto, 2), "s")\n'
             'print("PCA(20) + t-SNE:       ", round(com_pca, 2), "s")\n'
             'print("distancias entre grupos no mapa NAO sao confiaveis (so vizinhanca local).")'),
    ]
    escrever(nb, "07_exercicios/04_clustering_pca.ipynb")


# ══════════════════════════════════════════════════════════════════════════
def nb_ex_pipeline():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Exercícios — Pipeline completo\n\n"
           "Um projeto de ponta a ponta com dados de tipos mistos e valores faltantes, "
           "sem vazamento. Soluções em `# @title`."),
        code(PREAMBULO),
        md("## Preparação — um conjunto realista (numérico + categórico + faltantes)"),
        code('# @title Solução\n'
             'rng = np.random.RandomState(SEMENTE)\n'
             'n = 500\n'
             'idade = rng.normal(50, 12, n)\n'
             'pressao = rng.normal(125, 16, n)\n'
             'grupo = rng.choice(["A", "B", "C"], n)\n'
             'logito = 0.04*(idade-50) + 0.03*(pressao-125) + (grupo=="C")*1.0 + rng.normal(0, 0.6, n)\n'
             'y = (logito > np.median(logito)).astype(int)\n'
             '# injeta valores faltantes nas numericas\n'
             'idade[rng.rand(n) < 0.12] = np.nan\n'
             'pressao[rng.rand(n) < 0.08] = np.nan\n'
             'df = pd.DataFrame({"idade": idade, "pressao": pressao, "grupo": grupo})\n'
             'print("faltantes por coluna:\\n", df.isna().sum())'),
        md("## Exercício 1 — Montar o ColumnTransformer"),
        code('# @title Solução\n'
             'from sklearn.compose import ColumnTransformer\n'
             'from sklearn.pipeline import Pipeline\n'
             'from sklearn.impute import SimpleImputer\n'
             'from sklearn.preprocessing import StandardScaler, OneHotEncoder\n'
             'from sklearn.linear_model import LogisticRegression\n\n'
             'num = ["idade", "pressao"]\n'
             'cat = ["grupo"]\n'
             'prep = ColumnTransformer([\n'
             '    ("num", Pipeline([("imp", SimpleImputer(strategy="mean")), ("esc", StandardScaler())]), num),\n'
             '    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder())]), cat),\n'
             '])\n'
             'modelo = Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=5000))])\n'
             'print("pipeline montado. A imputacao fica DENTRO, ajustada so no treino de cada dobra.")'),
        md("## Exercício 2 — Comparar modelos com o mesmo pipeline"),
        code('# @title Solução\n'
             'from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier\n'
             'from sklearn.model_selection import cross_val_score\n\n'
             'for nome, clf in [("logistica", LogisticRegression(max_iter=5000)),\n'
             '                  ("random forest", RandomForestClassifier(n_estimators=200, random_state=SEMENTE)),\n'
             '                  ("grad. boosting", GradientBoostingClassifier(random_state=SEMENTE))]:\n'
             '    pipe = Pipeline([("prep", prep), ("clf", clf)])\n'
             '    ac = cross_val_score(pipe, df, y, cv=5).mean()\n'
             '    print(nome.ljust(15), "acuracia CV:", round(ac, 3))'),
        md("## Exercício 3 — Estimativa final honesta"),
        code('# @title Solução\n'
             'from sklearn.model_selection import train_test_split, GridSearchCV\n\n'
             'df_tr, df_te, y_tr, y_te = train_test_split(df, y, test_size=0.25, random_state=SEMENTE, stratify=y)\n'
             'pipe = Pipeline([("prep", prep), ("clf", RandomForestClassifier(random_state=SEMENTE))])\n'
             'grade = {"clf__n_estimators": [100, 300], "clf__max_depth": [None, 5]}\n'
             'busca = GridSearchCV(pipe, grade, cv=5).fit(df_tr, y_tr)\n'
             'print("melhor CV (otimista):", round(busca.best_score_, 3))\n'
             'print("acuracia no TESTE separado (honesta):", round(busca.score(df_te, y_te), 3))'),
        md("## Exercício 4 — Reprodutibilidade"),
        code('# @title Solução\n'
             'import joblib\n'
             'joblib.dump(busca.best_estimator_, "modelo_final.joblib")\n'
             'recarregado = joblib.load("modelo_final.joblib")\n'
             'print("modelo salvo e recarregado; previsoes identicas:",\n'
             '      np.array_equal(busca.best_estimator_.predict(df_te), recarregado.predict(df_te)))\n'
             'print("semente fixa + random_state em tudo + modelo salvo = resultado verificavel e reusavel.")'),
    ]
    escrever(nb, "07_exercicios/05_pipeline.ipynb")


# ══════════════════════════════════════════════════════════════════════════
def nb_perceptron():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Perceptron e redes neurais simples\n\n"
           "**Objetivo:** treinar um único neurônio (em PyTorch) num problema "
           "linearmente separável, ver que ele **falha** no XOR, e que uma rede com "
           "**uma camada oculta** resolve o XOR."),
        code(PREAMBULO + '\nimport torch'),
        md("## 1. Um neurônio num problema separável\n\n"
           "Um neurônio é `Linear(2,1)` seguido de sigmoide — literalmente uma regressão "
           "logística. Treinamos com o laço explícito de sempre."),
        code('from sklearn.datasets import make_blobs\n\n'
             'X, y = make_blobs(n_samples=200, centers=[[-2, -2], [2, 2]], cluster_std=1.2,\n'
             '                  random_state=SEMENTE)\n'
             'ent = torch.tensor(X, dtype=torch.float32)\n'
             'alvo = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)\n\n'
             'torch.manual_seed(SEMENTE)\n'
             'neuronio = torch.nn.Sequential(torch.nn.Linear(2, 1), torch.nn.Sigmoid())\n'
             'custo_fn = torch.nn.BCELoss()\n'
             'oti = torch.optim.SGD(neuronio.parameters(), lr=0.1)\n'
             'for epoca in range(300):\n'
             '    perda = custo_fn(neuronio(ent), alvo)\n'
             '    oti.zero_grad(); perda.backward(); oti.step()\n'
             'with torch.no_grad():\n'
             '    acc = (((neuronio(ent) > 0.5).float() == alvo).float().mean()).item()\n'
             'print("acuracia do neuronio (dados separaveis):", round(acc, 3))'),
        code('# fronteira de decisao do neuronio\n'
             'passo = 0.05\n'
             'gx, gy = np.meshgrid(np.arange(X[:,0].min()-1, X[:,0].max()+1, passo),\n'
             '                     np.arange(X[:,1].min()-1, X[:,1].max()+1, passo))\n'
             'with torch.no_grad():\n'
             '    zz = neuronio(torch.tensor(np.c_[gx.ravel(), gy.ravel()], dtype=torch.float32)).numpy().reshape(gx.shape)\n'
             'figura = go.Figure()\n'
             'figura.add_trace(go.Contour(x=gx[0], y=gy[:,0], z=zz, showscale=False,\n'
             '                            colorscale=[[0,"#dce7f4"],[1,"#f6dedb"]], opacity=0.6,\n'
             '                            contours=dict(start=0.5, end=0.5, size=1, coloring="lines")))\n'
             'figura.add_trace(go.Scatter(x=X[:,0], y=X[:,1], mode="markers",\n'
             '                            marker=dict(color=y, colorscale="Bluered", size=6)))\n'
             'figura.update_layout(title="Um neuronio: fronteira linear", height=380,\n'
             '                     showlegend=False, margin=dict(l=10,r=10,t=50,b=10))\n'
             'figura.show()'),
        md("## 2. O XOR quebra o neurônio\n\n"
           "No XOR, a classe é 1 quando as entradas **diferem**. Não há reta que "
           "separe — geramos uma versão ruidosa e treinamos o mesmo neurônio."),
        code('rng = np.random.RandomState(SEMENTE)\n'
             'centros = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)\n'
             'rotulos = np.array([0,1,1,0])   # XOR\n'
             'Xx = np.repeat(centros, 60, axis=0) + rng.normal(0, 0.12, size=(240,2))\n'
             'yy = np.repeat(rotulos, 60)\n'
             'entx = torch.tensor(Xx, dtype=torch.float32)\n'
             'alvox = torch.tensor(yy, dtype=torch.float32).reshape(-1,1)\n\n'
             'torch.manual_seed(SEMENTE)\n'
             'neuronio2 = torch.nn.Sequential(torch.nn.Linear(2,1), torch.nn.Sigmoid())\n'
             'oti = torch.optim.Adam(neuronio2.parameters(), lr=0.05)\n'
             'for epoca in range(400):\n'
             '    perda = custo_fn(neuronio2(entx), alvox)\n'
             '    oti.zero_grad(); perda.backward(); oti.step()\n'
             'with torch.no_grad():\n'
             '    acc1 = (((neuronio2(entx)>0.5).float()==alvox).float().mean()).item()\n'
             'print("acuracia de UM neuronio no XOR:", round(acc1, 3), "(preso perto de 0.5-0.75)")'),
        md("## 3. Uma camada oculta resolve\n\n"
           "Agora `Linear(2,8) → ReLU → Linear(8,1) → Sigmoide`: a camada oculta cria "
           "representações que tornam o XOR separável. Mesmo laço, rede um pouco maior."),
        code('torch.manual_seed(SEMENTE)\n'
             'rede = torch.nn.Sequential(\n'
             '    torch.nn.Linear(2, 8), torch.nn.ReLU(),\n'
             '    torch.nn.Linear(8, 1), torch.nn.Sigmoid())\n'
             'oti = torch.optim.Adam(rede.parameters(), lr=0.05)\n'
             'for epoca in range(400):\n'
             '    perda = custo_fn(rede(entx), alvox)\n'
             '    oti.zero_grad(); perda.backward(); oti.step()\n'
             'with torch.no_grad():\n'
             '    acc2 = (((rede(entx)>0.5).float()==alvox).float().mean()).item()\n'
             'print("acuracia de UM neuronio no XOR:", round(acc1, 3))\n'
             'print("acuracia da rede com camada oculta:", round(acc2, 3))'),
        md("## Exercício\n\n"
           "Compare `acc1` e `acc2`. Por que acrescentar uma camada oculta (com ReLU) "
           "muda tão radicalmente o resultado no XOR?"),
        md("<details><summary>Ver resposta</summary>\n\n"
           "Um neurônio só traça **uma reta**, e o XOR não é linearmente separável — daí "
           "`acc1` travar longe de 100%. A camada oculta com ReLU cria **várias** "
           "fronteiras lineares e as combina de forma **não linear**, construindo uma "
           "representação em que as classes do XOR passam a ser separáveis pela camada "
           "final. É a não linearidade da camada oculta que quebra a limitação — `acc2` "
           "chega perto de 100%.\n\n</details>"),
    ]
    escrever(nb, "06_redes_neurais/01_perceptron.ipynb")


# ══════════════════════════════════════════════════════════════════════════
def nb_backprop():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Backpropagation e gradiente descendente\n\n"
           "**Objetivo:** ver o *autograd* calcular gradientes, e comparar as curvas de "
           "perda de **full-batch, mini-batch e SGD** na mesma rede — tornando visível o "
           "ruído de cada estratégia."),
        code(PREAMBULO + '\nimport torch'),
        md("## 1. O autograd calcula o gradiente\n\n"
           "Definimos um peso, uma perda simples $\\mathcal{L} = (w \\cdot 3 - 6)^2$ e "
           "pedimos `.backward()`. A derivada é $2(3w-6)\\cdot 3$; em $w=1$ vale "
           "$2(-3)(3) = -18$. O autograd confere."),
        code('w = torch.tensor(1.0, requires_grad=True)\n'
             'perda = (w * 3 - 6) ** 2\n'
             'perda.backward()\n'
             'print("perda em w=1:", perda.item())\n'
             'print("dL/dw (autograd):", w.grad.item(), "| esperado 2*(3w-6)*3 =", 2*(3*1-6)*3)'),
        md("## 2. Full-batch × mini-batch × SGD\n\n"
           "Treinamos a **mesma** rede num problema de duas luas, mudando só **quantos "
           "exemplos** cada passo usa. Registramos a perda por época; um laço explícito "
           "fatia os mini-lotes à mão."),
        code('from sklearn.datasets import make_moons\n'
             'from sklearn.preprocessing import StandardScaler\n\n'
             'X, y = make_moons(n_samples=400, noise=0.2, random_state=SEMENTE)\n'
             'X = StandardScaler().fit_transform(X)\n'
             'ent = torch.tensor(X, dtype=torch.float32)\n'
             'alvo = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)\n'
             'custo_fn = torch.nn.BCELoss()\n\n'
             'curvas = {}\n'
             'for nome, tam_lote in [("full-batch", 400), ("mini-batch (32)", 32), ("SGD (1)", 1)]:\n'
             '    torch.manual_seed(SEMENTE)\n'
             '    rede = torch.nn.Sequential(torch.nn.Linear(2,16), torch.nn.ReLU(),\n'
             '                               torch.nn.Linear(16,1), torch.nn.Sigmoid())\n'
             '    oti = torch.optim.SGD(rede.parameters(), lr=0.1)\n'
             '    perdas = []\n'
             '    for epoca in range(60):\n'
             '        ordem = torch.randperm(len(ent))\n'
             '        for i in range(0, len(ent), tam_lote):\n'
             '            idx = ordem[i:i+tam_lote]\n'
             '            perda = custo_fn(rede(ent[idx]), alvo[idx])\n'
             '            oti.zero_grad(); perda.backward(); oti.step()\n'
             '        with torch.no_grad():\n'
             '            perdas.append(custo_fn(rede(ent), alvo).item())\n'
             '    curvas[nome] = perdas\n'
             '    print(nome.ljust(16), "perda final:", round(perdas[-1], 4))'),
        code('figura = go.Figure()\n'
             'for nome, cor in [("full-batch", AZUL), ("mini-batch (32)", VERDE), ("SGD (1)", VERMELHO)]:\n'
             '    figura.add_trace(go.Scatter(y=curvas[nome], mode="lines", line=dict(color=cor), name=nome))\n'
             'figura.update_layout(title="Perda por epoca: full-batch (suave) x mini-batch x SGD (ruidoso)",\n'
             '                     xaxis_title="epoca", yaxis_title="perda (BCE)", height=380,\n'
             '                     margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()'),
        md("## Exercício\n\n"
           "Olhando as curvas, o SGD costuma cair mais rápido nas primeiras épocas mas "
           "com mais oscilação; o full-batch é suave e mais lento. Explique os dois "
           "efeitos com base no número de atualizações por época."),
        md("<details><summary>Ver resposta</summary>\n\n"
           "O **SGD** faz **uma atualização por exemplo** — com 400 exemplos, são 400 "
           "passos por época, daí cair rápido no começo; mas cada passo usa um gradiente "
           "ruidoso (um só exemplo), o que gera a **oscilação**. O **full-batch** faz "
           "**um único** passo por época, com o gradiente exato (todos os exemplos): "
           "trajetória **suave**, porém poucos passos, logo mais lenta. O mini-batch fica "
           "no meio — vários passos por época com ruído moderado —, por isso é o padrão.\n\n</details>"),
    ]
    escrever(nb, "06_redes_neurais/02_backprop.ipynb")


# ══════════════════════════════════════════════════════════════════════════
def nb_ativacoes():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Funções de ativação\n\n"
           "**Objetivo:** desenhar as ativações e suas derivadas, treinar a mesma rede "
           "com sigmoide e com ReLU, e **medir** o gradiente que desaparece nas primeiras "
           "camadas de uma rede profunda."),
        code(PREAMBULO + '\nimport torch'),
        md("## 1. As funções e suas derivadas\n\n"
           "O que a backpropagation multiplica é a **derivada**. Repare como a da "
           "sigmoide e a da tanh somem longe do zero, enquanto a da ReLU é 1 para "
           "$z>0$."),
        code('z = np.linspace(-6, 6, 300)\n'
             'sig = 1/(1+np.exp(-z)); d_sig = sig*(1-sig)\n'
             'th = np.tanh(z); d_th = 1 - th**2\n'
             'relu = np.maximum(0, z); d_relu = (z > 0).astype(float)\n\n'
             'from plotly.subplots import make_subplots\n'
             'figura = make_subplots(rows=1, cols=2, subplot_titles=("f(z)", "derivada f\'(z)"))\n'
             'for nome, curva, cor in [("sigmoide", sig, AZUL), ("tanh", th, VERDE), ("ReLU", relu, VERMELHO)]:\n'
             '    figura.add_trace(go.Scatter(x=z, y=curva, mode="lines", line=dict(color=cor), name=nome), row=1, col=1)\n'
             'for nome, curva, cor in [("sigmoide", d_sig, AZUL), ("tanh", d_th, VERDE), ("ReLU", d_relu, VERMELHO)]:\n'
             '    figura.add_trace(go.Scatter(x=z, y=curva, mode="lines", line=dict(color=cor), name=nome, showlegend=False), row=1, col=2)\n'
             'figura.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()\n'
             'print("derivada maxima da sigmoide:", round(d_sig.max(), 3), "(em z=0)")'),
        md("## 2. Sigmoide × ReLU no mesmo problema\n\n"
           "Treinamos a mesma rede de duas luas trocando só a ativação oculta. A ReLU "
           "costuma convergir mais rápido."),
        code('from sklearn.datasets import make_moons\n'
             'from sklearn.preprocessing import StandardScaler\n\n'
             'X, y = make_moons(n_samples=400, noise=0.2, random_state=SEMENTE)\n'
             'X = StandardScaler().fit_transform(X)\n'
             'ent = torch.tensor(X, dtype=torch.float32)\n'
             'alvo = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)\n'
             'custo_fn = torch.nn.BCELoss()\n\n'
             'figura = go.Figure()\n'
             'for nome, ativacao, cor in [("sigmoide", torch.nn.Sigmoid(), AZUL), ("ReLU", torch.nn.ReLU(), VERMELHO)]:\n'
             '    torch.manual_seed(SEMENTE)\n'
             '    rede = torch.nn.Sequential(torch.nn.Linear(2,16), ativacao,\n'
             '                               torch.nn.Linear(16,1), torch.nn.Sigmoid())\n'
             '    oti = torch.optim.SGD(rede.parameters(), lr=0.1)\n'
             '    perdas = []\n'
             '    for epoca in range(200):\n'
             '        perda = custo_fn(rede(ent), alvo)\n'
             '        oti.zero_grad(); perda.backward(); oti.step()\n'
             '        perdas.append(perda.item())\n'
             '    figura.add_trace(go.Scatter(y=perdas, mode="lines", line=dict(color=cor), name=nome))\n'
             '    print(nome.ljust(9), "perda final:", round(perdas[-1], 4))\n'
             'figura.update_layout(title="Convergencia: sigmoide x ReLU", xaxis_title="epoca",\n'
             '                     yaxis_title="perda", height=360, margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()'),
        md("## 3. O gradiente que desaparece, medido\n\n"
           "Montamos uma rede **profunda** (6 camadas) e, após um único `backward`, "
           "medimos a magnitude do gradiente em cada camada. Com sigmoide, os gradientes "
           "das **primeiras** camadas são minúsculos; com ReLU, sobrevivem."),
        code('for nome, ativacao in [("sigmoide", torch.nn.Sigmoid), ("ReLU", torch.nn.ReLU)]:\n'
             '    torch.manual_seed(SEMENTE)\n'
             '    camadas = []\n'
             '    for c in range(6):\n'
             '        camadas.append(torch.nn.Linear(16 if c else 2, 16))\n'
             '        camadas.append(ativacao())\n'
             '    camadas.append(torch.nn.Linear(16, 1)); camadas.append(torch.nn.Sigmoid())\n'
             '    rede = torch.nn.Sequential(*camadas)\n'
             '    perda = custo_fn(rede(ent), alvo)\n'
             '    rede.zero_grad(); perda.backward()\n'
             '    normas = []\n'
             '    for camada in rede:\n'
             '        if isinstance(camada, torch.nn.Linear):\n'
             '            normas.append(camada.weight.grad.norm().item())\n'
             '    print(nome.ljust(9), "norma do gradiente por camada (entrada -> saida):")\n'
             '    print("   ", [round(v, 5) for v in normas])'),
        md("## Exercício\n\n"
           "No item 3, compare a norma do gradiente da **primeira** camada entre "
           "sigmoide e ReLU. O que esse número mostra sobre por que a ReLU virou padrão?"),
        md("<details><summary>Ver resposta</summary>\n\n"
           "Com **sigmoide**, a norma do gradiente na primeira camada é ordens de "
           "grandeza **menor** que nas últimas — o gradiente praticamente **desapareceu** "
           "ao atravessar as camadas, porque cada uma multiplicou por uma derivada < 1. "
           "Com **ReLU**, a derivada é 1 na região ativa, então o gradiente chega às "
           "primeiras camadas com magnitude útil e elas **conseguem aprender**. É "
           "exatamente por isso que a ReLU destravou o treino de redes profundas.\n\n</details>"),
    ]
    escrever(nb, "06_redes_neurais/03_ativacoes.ipynb")


# ══════════════════════════════════════════════════════════════════════════
def nb_deep_learning():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Introdução ao deep learning\n\n"
           "**Objetivo:** treinar uma pequena rede densa em PyTorch para classificar "
           "**dígitos manuscritos** (imagens), comparar com uma regressão logística e "
           "discutir, com números, o que a profundidade acrescentou."),
        code(PREAMBULO + '\nimport torch'),
        code('from sklearn.datasets import load_digits\n'
             'from sklearn.model_selection import train_test_split\n'
             'from sklearn.preprocessing import StandardScaler\n\n'
             'digitos = load_digits()\n'
             'X = StandardScaler().fit_transform(digitos.data)   # 64 pixels (8x8)\n'
             'y = digitos.target                                 # 0 a 9\n'
             'X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,\n'
             '                                          random_state=SEMENTE, stratify=y)\n'
             'print("treino:", X_tr.shape, "| 10 classes de digitos")'),
        md("## 1. Uma linha de base linear\n\n"
           "Antes da rede, uma regressão logística — para sabermos o que a profundidade "
           "precisa superar."),
        code('from sklearn.linear_model import LogisticRegression\n'
             'from sklearn.metrics import accuracy_score\n\n'
             'base = LogisticRegression(max_iter=5000).fit(X_tr, y_tr)\n'
             'print("acuracia da regressao logistica:", round(base.score(X_te, y_te), 3))'),
        md("## 2. A rede densa em PyTorch\n\n"
           "`64 → 64 → 32 → 10`, com ReLU nas ocultas. Para 10 classes usamos "
           "`CrossEntropyLoss` (que já embute o softmax). Laço de treino explícito, em "
           "mini-lotes."),
        code('ent_tr = torch.tensor(X_tr, dtype=torch.float32)\n'
             'alvo_tr = torch.tensor(y_tr, dtype=torch.long)\n'
             'ent_te = torch.tensor(X_te, dtype=torch.float32)\n\n'
             'torch.manual_seed(SEMENTE)\n'
             'rede = torch.nn.Sequential(\n'
             '    torch.nn.Linear(64, 64), torch.nn.ReLU(),\n'
             '    torch.nn.Linear(64, 32), torch.nn.ReLU(),\n'
             '    torch.nn.Linear(32, 10))\n'
             'custo_fn = torch.nn.CrossEntropyLoss()\n'
             'oti = torch.optim.Adam(rede.parameters(), lr=0.01)\n\n'
             'perdas = []\n'
             'for epoca in range(80):\n'
             '    ordem = torch.randperm(len(ent_tr))\n'
             '    for i in range(0, len(ent_tr), 64):\n'
             '        idx = ordem[i:i+64]\n'
             '        perda = custo_fn(rede(ent_tr[idx]), alvo_tr[idx])\n'
             '        oti.zero_grad(); perda.backward(); oti.step()\n'
             '    perdas.append(perda.item())\n\n'
             'with torch.no_grad():\n'
             '    previsto = rede(ent_te).argmax(dim=1).numpy()\n'
             'print("acuracia da rede densa:", round(accuracy_score(y_te, previsto), 3))'),
        code('figura = go.Figure(go.Scatter(y=perdas, mode="lines", line=dict(color=VERDE)))\n'
             'figura.update_layout(title="Perda do treino da rede densa (dígitos)",\n'
             '                     xaxis_title="epoca", yaxis_title="CrossEntropy", height=320,\n'
             '                     margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()'),
        md("## 3. O que a profundidade acrescentou?\n\n"
           "O resultado é honesto e instrutivo: a rede densa **empata** com a regressão "
           "logística — aqui, fica até um fio atrás. Nos dígitos 8×8, já quase linearmente "
           "separáveis, a profundidade **não** traz vantagem, exatamente o ponto do "
           "texto. O deep learning **decola** mesmo em imagens grandes e cruas (com CNNs) "
           "e em texto; numa base pequena e simples, um bom modelo clássico iguala ou "
           "supera a rede, com muito menos esforço."),
        md("## Exercício\n\n"
           "A rede **empatou** (ou perdeu por pouco) para a regressão logística. Em que "
           "cenário o deep learning teria vantagem **clara**?"),
        md("<details><summary>Ver resposta</summary>\n\n"
           "Quando os dados são **imagens grandes e cruas** (não 8×8, mas centenas de "
           "milhares de pixels), **texto** ou **áudio**, e há **muitos** exemplos. Aí a "
           "capacidade do deep learning de aprender **representações hierárquicas** "
           "(bordas → partes → objetos), tipicamente com **CNNs** ou **Transformers**, "
           "supera de longe qualquer modelo linear sobre pixels crus. Nos dígitos 8×8, "
           "quase separáveis, sobra pouco espaço para essa vantagem aparecer.\n\n</details>"),
    ]
    escrever(nb, "06_redes_neurais/04_deep_learning.ipynb")


# ══════════════════════════════════════════════════════════════════════════
def nb_kmeans():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# k-means clustering\n\n"
           "**Objetivo:** agrupar o Iris **sem** usar os rótulos, escolher $k$ pelo "
           "cotovelo e pela silhueta, comparar os grupos com as espécies verdadeiras e "
           "ver um caso em que o k-means falha."),
        code(PREAMBULO),
        code('from sklearn.datasets import load_iris\n'
             'from sklearn.preprocessing import StandardScaler\n'
             'from sklearn.cluster import KMeans\n'
             'from sklearn.metrics import silhouette_score\n\n'
             'iris = load_iris()\n'
             'X = StandardScaler().fit_transform(iris.data)   # padronizar e essencial\n'
             'y_verdade = iris.target\n'
             'print("X:", X.shape, "(rotulos escondidos do algoritmo)")'),
        md("## 1. Cotovelo e silhueta\n\n"
           "A inércia sempre cai com $k$ — procuramos o **cotovelo**. A silhueta média "
           "tem um **pico** no $k$ com grupos mais bem separados. Um laço explícito "
           "calcula os dois para cada $k$."),
        code('ks = list(range(2, 9))\n'
             'inercias = []\n'
             'silhuetas = []\n'
             'for k in ks:\n'
             '    modelo = KMeans(n_clusters=k, n_init=10, random_state=SEMENTE).fit(X)\n'
             '    inercias.append(modelo.inertia_)\n'
             '    silhuetas.append(silhouette_score(X, modelo.labels_))\n'
             '    print("k =", k, "| inercia", round(modelo.inertia_, 1), "| silhueta", round(silhuetas[-1], 3))\n\n'
             'from plotly.subplots import make_subplots\n'
             'figura = make_subplots(rows=1, cols=2, subplot_titles=("Cotovelo (inercia)", "Silhueta media"))\n'
             'figura.add_trace(go.Scatter(x=ks, y=inercias, mode="lines+markers", line=dict(color=AZUL)), row=1, col=1)\n'
             'figura.add_trace(go.Scatter(x=ks, y=silhuetas, mode="lines+markers", line=dict(color=VERDE)), row=1, col=2)\n'
             'figura.update_layout(height=340, showlegend=False, margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()\n'
             'print("k de maior silhueta:", ks[int(np.argmax(silhuetas))])'),
        md("## 2. Os grupos encontrados × as espécies reais\n\n"
           "Com $k=3$, comparamos os grupos do k-means (que nunca viu os rótulos) com as "
           "três espécies. A tabela cruzada mostra o quanto eles coincidem."),
        code('modelo = KMeans(n_clusters=3, n_init=10, random_state=SEMENTE).fit(X)\n'
             'tabela = pd.crosstab(pd.Series(iris.target_names[y_verdade], name="especie real"),\n'
             '                     pd.Series(modelo.labels_, name="grupo do k-means"))\n'
             'print(tabela)\n'
             'print("\\nsetosa costuma ficar sozinha num grupo; versicolor e virginica se misturam um pouco.")'),
        md("## 3. Onde o k-means falha\n\n"
           "O k-means supõe grupos **esféricos**. Em dados com formato de duas luas, ele "
           "corta pelo meio em vez de seguir as luas — a lição de que a suposição "
           "importa."),
        code('from sklearn.datasets import make_moons\n\n'
             'X_luas, _ = make_moons(n_samples=300, noise=0.06, random_state=SEMENTE)\n'
             'grupos_luas = KMeans(n_clusters=2, n_init=10, random_state=SEMENTE).fit_predict(X_luas)\n\n'
             'figura = go.Figure(go.Scatter(x=X_luas[:, 0], y=X_luas[:, 1], mode="markers",\n'
             '                              marker=dict(color=grupos_luas, colorscale="Bluered", size=6)))\n'
             'figura.update_layout(title="k-means em duas luas: corta reto, ignora a forma",\n'
             '                     height=380, showlegend=False, margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()'),
        md("## Exercício\n\n"
           "Pela silhueta do item 1, qual $k$ o método sugere para o Iris? Isso bate com "
           "o número de espécies? Se não, o que pode explicar a diferença?"),
        md("<details><summary>Ver resposta</summary>\n\n"
           "A silhueta costuma indicar $k=2$ para o Iris, embora existam **três** "
           "espécies. O motivo: *setosa* é muito separada das outras duas, enquanto "
           "*versicolor* e *virginica* se sobrepõem bastante — do ponto de vista de "
           "distância, elas parecem quase um único grupo. A silhueta mede separação "
           "geométrica, não conhece as espécies; por isso premia a divisão em 2 grupos "
           "bem distintos. É um lembrete de que o \"melhor\" $k$ estatístico nem sempre "
           "é o número de classes reais.\n\n</details>"),
    ]
    escrever(nb, "05_nao_supervisionado/01_kmeans.ipynb")


# ══════════════════════════════════════════════════════════════════════════
def nb_hierarquico():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Clustering hierárquico\n\n"
           "**Objetivo:** construir e ler um **dendrograma** do Iris (ligação de Ward), "
           "cortá-lo em $k$ grupos com o `AgglomerativeClustering` e comparar critérios "
           "de ligação."),
        code(PREAMBULO),
        code('from sklearn.datasets import load_iris\n'
             'from sklearn.preprocessing import StandardScaler\n\n'
             'iris = load_iris()\n'
             'X = StandardScaler().fit_transform(iris.data)\n'
             '# amostra pequena para um dendrograma legivel\n'
             'rng = np.random.RandomState(SEMENTE)\n'
             'indices = rng.choice(len(X), 30, replace=False)\n'
             'X_amostra = X[indices]\n'
             'especies = iris.target_names[iris.target[indices]]\n'
             'print("amostra:", X_amostra.shape)'),
        md("## 1. O dendrograma (ligação de Ward)\n\n"
           "Cada folha é uma flor; a **altura** de cada união mede o quão diferentes "
           "eram os grupos fundidos. Saltos grandes de altura marcam separações "
           "naturais — bons lugares para cortar."),
        code('import plotly.figure_factory as ff\n'
             'from scipy.cluster.hierarchy import linkage\n\n'
             'figura = ff.create_dendrogram(X_amostra, labels=list(especies),\n'
             '                              linkagefun=lambda d: linkage(d, "ward"))\n'
             'figura.update_layout(title="Dendrograma do Iris (ligacao de Ward)",\n'
             '                     height=420, margin=dict(l=10, r=10, t=50, b=80))\n'
             'figura.show()'),
        md("## 2. Cortando em k grupos\n\n"
           "O `AgglomerativeClustering` corta a árvore para dar exatamente $k$ grupos. "
           "Com $k=3$, comparamos com as espécies (usando a base completa)."),
        code('from sklearn.cluster import AgglomerativeClustering\n\n'
             'agrupador = AgglomerativeClustering(n_clusters=3, linkage="ward")\n'
             'grupos = agrupador.fit_predict(X)\n'
             'tabela = pd.crosstab(pd.Series(iris.target_names[iris.target], name="especie"),\n'
             '                     pd.Series(grupos, name="grupo (Ward)"))\n'
             'print(tabela)'),
        md("## 3. O critério de ligação muda tudo\n\n"
           "Comparamos quatro critérios pelo quanto os grupos batem com as espécies "
           "(índice de Rand ajustado, de 0 a 1). A ligação **simples** costuma sofrer "
           "com o encadeamento; **Ward** e **completa** vão melhor."),
        code('from sklearn.metrics import adjusted_rand_score\n\n'
             'for ligacao in ["single", "complete", "average", "ward"]:\n'
             '    g = AgglomerativeClustering(n_clusters=3, linkage=ligacao).fit_predict(X)\n'
             '    ari = adjusted_rand_score(iris.target, g)\n'
             '    print("ligacao", ligacao.ljust(9), "-> ARI com as especies:", round(ari, 3))'),
        md("## Exercício\n\n"
           "No item 3, a ligação `single` costuma dar o pior ARI. Relacione isso com o "
           "efeito de **encadeamento** discutido no texto."),
        md("<details><summary>Ver resposta</summary>\n\n"
           "A ligação simples define a distância entre grupos pelos **dois pontos mais "
           "próximos**. Como *versicolor* e *virginica* se tocam (há pontos de um bem "
           "perto do outro), esses pares próximos formam uma \"ponte\" e o algoritmo "
           "funde os dois cedo — o **encadeamento** —, produzindo grupos que não "
           "correspondem às espécies e, por isso, um ARI baixo. Ward e completa olham o "
           "grupo como um todo e resistem melhor a essa ponte.\n\n</details>"),
    ]
    escrever(nb, "05_nao_supervisionado/02_hierarquico.ipynb")


# ══════════════════════════════════════════════════════════════════════════
def nb_pca():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Análise de Componentes Principais (PCA)\n\n"
           "**Objetivo:** reduzir a dimensionalidade do conjunto **breast cancer** (30 "
           "variáveis) com PCA, ler o *scree plot* e a variância acumulada, e projetar "
           "em 2D — vendo as classes se separarem sem terem sido usadas."),
        code(PREAMBULO),
        code('from sklearn.datasets import load_breast_cancer\n'
             'from sklearn.preprocessing import StandardScaler\n'
             'from sklearn.decomposition import PCA\n\n'
             'dados = load_breast_cancer()\n'
             'X = StandardScaler().fit_transform(dados.data)   # padronizar antes da PCA\n'
             'y = dados.target\n'
             'print("X:", X.shape, "-> vamos comprimir 30 variaveis")'),
        md("## 1. Variância explicada (scree plot)\n\n"
           "Cada componente captura uma fração da variância total (seu autovalor "
           "normalizado). O *scree plot* mostra quanto cada uma guarda; a curva "
           "acumulada mostra quantas bastam."),
        code('pca = PCA().fit(X)\n'
             'variancia = pca.explained_variance_ratio_\n'
             'acumulada = np.cumsum(variancia)\n'
             'for i in range(6):\n'
             '    print("PC", i+1, "-> variancia", round(variancia[i], 3), "| acumulada", round(acumulada[i], 3))\n\n'
             'from plotly.subplots import make_subplots\n'
             'figura = make_subplots(rows=1, cols=2, subplot_titles=("Variancia por componente", "Variancia acumulada"))\n'
             'figura.add_trace(go.Bar(x=list(range(1, 11)), y=variancia[:10], marker_color=AZUL), row=1, col=1)\n'
             'figura.add_trace(go.Scatter(x=list(range(1, 11)), y=acumulada[:10], mode="lines+markers",\n'
             '                            line=dict(color=VERDE)), row=1, col=2)\n'
             'figura.add_hline(y=0.9, line_dash="dash", line_color=VERMELHO, row=1, col=2)\n'
             'figura.update_layout(height=340, showlegend=False, margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()\n'
             'n90 = int(np.argmax(acumulada >= 0.9)) + 1\n'
             'print("componentes para reter 90% da variancia:", n90, "de 30")'),
        md("## 2. Os dados em 2D\n\n"
           "Projetando nas duas primeiras componentes, os 30 números viram 2 — e as "
           "classes (que a PCA **não** viu) já aparecem quase separadas."),
        code('coords = PCA(n_components=2).fit_transform(X)\n'
             'figura = go.Figure()\n'
             'for classe, nome, cor in [(0, "maligno", VERMELHO), (1, "benigno", AZUL)]:\n'
             '    m = y == classe\n'
             '    figura.add_trace(go.Scatter(x=coords[m, 0], y=coords[m, 1], mode="markers",\n'
             '                                marker=dict(color=cor, size=6, opacity=0.6), name=nome))\n'
             'figura.update_layout(title="breast cancer projetado em 2 componentes principais",\n'
             '                     xaxis_title="PC1", yaxis_title="PC2", height=420,\n'
             '                     margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()'),
        md("## 3. O que carrega cada componente\n\n"
           "As **cargas** dizem o peso de cada variável original em cada componente. As "
           "cinco maiores da PC1 revelam o que domina a maior direção de variação."),
        code('pca2 = PCA(n_components=2).fit(X)\n'
             'cargas_pc1 = pca2.components_[0]\n'
             'ordem = np.argsort(np.abs(cargas_pc1))[::-1][:5]\n'
             'print("variaveis com maior peso na PC1:")\n'
             'for j in ordem:\n'
             '    print("  ", dados.feature_names[j].ljust(24), round(cargas_pc1[j], 3))'),
        md("## Exercício\n\n"
           "Se a PC1 sozinha explica cerca de 44% da variância e a PC2 cerca de 19%, "
           "quanto se perde ao olhar só o plano PC1–PC2? Isso invalida a visualização?"),
        md("<details><summary>Ver resposta</summary>\n\n"
           "As duas juntas retêm $\\approx 44\\% + 19\\% = 63\\%$; a projeção 2D perde os "
           "$\\approx 37\\%$ restantes, espalhados pelas outras 28 componentes. Não "
           "invalida a visualização: 63% da variação num único plano já basta para ver a "
           "separação dominante entre maligno e benigno. A ressalva é lembrar que pontos "
           "próximos no plano podem diferir nas dimensões descartadas.\n\n</details>"),
    ]
    escrever(nb, "05_nao_supervisionado/03_pca.ipynb")


# ══════════════════════════════════════════════════════════════════════════
def nb_tsne_umap():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# t-SNE e UMAP\n\n"
           "**Objetivo:** comparar a projeção **linear** da PCA com a **não linear** do "
           "t-SNE no conjunto de dígitos manuscritos, ver os dígitos se separarem em "
           "ilhas, e experimentar o efeito da perplexidade. UMAP entra como opcional."),
        code(PREAMBULO),
        code('from sklearn.datasets import load_digits\n\n'
             'digitos = load_digits()\n'
             '# subamostra para o t-SNE rodar rapido\n'
             'rng = np.random.RandomState(SEMENTE)\n'
             'sel = rng.choice(len(digitos.data), 600, replace=False)\n'
             'X = digitos.data[sel]      # 64 dimensoes (imagens 8x8)\n'
             'y = digitos.target[sel]\n'
             'print("X:", X.shape, "| digitos de 0 a 9")'),
        md("## 1. PCA linear × t-SNE não linear\n\n"
           "A PCA projeta no plano de maior variância; o t-SNE preserva a **vizinhança "
           "local**. Lado a lado, os dígitos que a PCA mistura o t-SNE separa em ilhas."),
        code('from sklearn.decomposition import PCA\n'
             'from sklearn.manifold import TSNE\n\n'
             'coords_pca = PCA(n_components=2).fit_transform(X)\n'
             'coords_tsne = TSNE(n_components=2, perplexity=30, init="pca",\n'
             '                   random_state=SEMENTE).fit_transform(X)\n\n'
             'from plotly.subplots import make_subplots\n'
             'figura = make_subplots(rows=1, cols=2, subplot_titles=("PCA (linear)", "t-SNE (nao linear)"))\n'
             'figura.add_trace(go.Scatter(x=coords_pca[:, 0], y=coords_pca[:, 1], mode="markers",\n'
             '                            marker=dict(color=y, colorscale="Rainbow", size=5, showscale=False),\n'
             '                            text=y), row=1, col=1)\n'
             'figura.add_trace(go.Scatter(x=coords_tsne[:, 0], y=coords_tsne[:, 1], mode="markers",\n'
             '                            marker=dict(color=y, colorscale="Rainbow", size=5, showscale=False),\n'
             '                            text=y), row=1, col=2)\n'
             'figura.update_layout(height=420, showlegend=False, margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()'),
        md("## 2. O efeito da perplexidade\n\n"
           "A perplexidade regula quantos vizinhos cada ponto considera. Valores muito "
           "baixos fragmentam; muito altos borram. Comparamos três."),
        code('figura = make_subplots(rows=1, cols=3, subplot_titles=("perplexidade 5", "30", "50"))\n'
             'coluna = 1\n'
             'for perp in [5, 30, 50]:\n'
             '    coords = TSNE(n_components=2, perplexity=perp, init="pca",\n'
             '                  random_state=SEMENTE).fit_transform(X)\n'
             '    figura.add_trace(go.Scatter(x=coords[:, 0], y=coords[:, 1], mode="markers",\n'
             '                                marker=dict(color=y, colorscale="Rainbow", size=4)),\n'
             '                     row=1, col=coluna)\n'
             '    coluna += 1\n'
             'figura.update_layout(height=340, showlegend=False, margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()'),
        md("## 3. (Opcional) UMAP\n\n"
           "Se a biblioteca `umap-learn` estiver instalada (no Colab, um `!pip install "
           "umap-learn` resolve), a célula abaixo roda o UMAP; senão, avisa e segue."),
        code('try:\n'
             '    import umap\n'
             '    reducao = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=SEMENTE)\n'
             '    coords_umap = reducao.fit_transform(X)\n'
             '    figura = go.Figure(go.Scatter(x=coords_umap[:, 0], y=coords_umap[:, 1], mode="markers",\n'
             '                                  marker=dict(color=y, colorscale="Rainbow", size=5)))\n'
             '    figura.update_layout(title="UMAP dos digitos", height=420, showlegend=False,\n'
             '                         margin=dict(l=10, r=10, t=50, b=10))\n'
             '    figura.show()\n'
             'except Exception as erro:\n'
             '    print("umap-learn nao disponivel — pulando.")\n'
             '    print("para instalar no Colab: !pip install umap-learn")\n'
             '    print("detalhe:", type(erro).__name__)'),
        md("## Exercício\n\n"
           "No mapa t-SNE, dois grupos de dígitos aparecem bem afastados. Você pode "
           "concluir que esses dígitos são \"mais diferentes\" entre si do que dois "
           "grupos próximos? Por quê?"),
        md("<details><summary>Ver resposta</summary>\n\n"
           "**Não.** O t-SNE preserva a estrutura **local** (quem é vizinho de quem), "
           "não as distâncias **globais**. A separação entre dois grupos no mapa é "
           "amplamente arbitrária — o algoritmo é livre para posicionar ilhas distantes "
           "sem que isso reflita a distância real no espaço de 64 dimensões. Para "
           "comparar o quão diferentes são dois grupos, é preciso medir no espaço "
           "original, não no mapa.\n\n</details>"),
    ]
    escrever(nb, "05_nao_supervisionado/04_tsne_umap.ipynb")


# ══════════════════════════════════════════════════════════════════════════
# Capítulo 1 — Fundamentos (padrão novo: Plotly, sem def, código explícito)
# ══════════════════════════════════════════════════════════════════════════
def nb_f_introducao():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# O que é aprendizagem de máquina?\n\n"
           "**Objetivo:** rodar o \"olá, mundo\" da aprendizagem supervisionada — "
           "carregar o Iris, treinar um classificador em poucas linhas e medir seu "
           "desempenho em dados que ele não viu. O ciclo mínimo: `dados → treino → "
           "avaliação`."),
        code(PREAMBULO),
        md("## 1. Carregar os dados\n\n"
           "O Iris tem 150 flores, 4 características e 3 espécies — o conjunto didático "
           "mais famoso da área."),
        code('from sklearn.datasets import load_iris\n\n'
             'iris = load_iris(as_frame=True)\n'
             'X, y = iris.data, iris.target\n'
             'print(X.shape, "->", list(iris.target_names))\n'
             'X.head()'),
        md("## 2. Separar treino e teste\n\n"
           "Avaliamos o modelo em dados **nunca vistos** durante o treino — a essência "
           "da generalização."),
        code('from sklearn.model_selection import train_test_split\n\n'
             'X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3,\n'
             '                                          random_state=SEMENTE, stratify=y)\n'
             'print("treino:", X_tr.shape[0], "| teste:", X_te.shape[0])'),
        md("## 3. Treinar e avaliar\n\n"
           "Um classificador k-NN em três linhas: `fit` aprende, `score` mede a "
           "acurácia no teste."),
        code('from sklearn.neighbors import KNeighborsClassifier\n\n'
             'modelo = KNeighborsClassifier(n_neighbors=5)\n'
             'modelo.fit(X_tr, y_tr)\n'
             'print("acuracia no teste:", round(modelo.score(X_te, y_te), 3))'),
        md("## Exercícios\n\n"
           "**1.** Troque `n_neighbors` para 1 e 50 e relacione com overfitting/"
           "underfitting.\n\n**2.** Substitua o k-NN por uma `LogisticRegression`."),
        code('# @title Solução\n'
             'for k in [1, 5, 50]:\n'
             '    m = KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr)\n'
             '    print("k =", str(k).rjust(2), "| treino", round(m.score(X_tr, y_tr), 3),\n'
             '          "| teste", round(m.score(X_te, y_te), 3))\n'
             'from sklearn.linear_model import LogisticRegression\n'
             'lr = LogisticRegression(max_iter=500).fit(X_tr, y_tr)\n'
             'print("LogReg teste:", round(lr.score(X_te, y_te), 3))'),
    ]
    escrever(nb, "01_fundamentos/01_introducao_ml.ipynb")


def nb_f_tipos():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Tipos de aprendizagem\n\n"
           "**Objetivo:** usar os mesmos dados para formular três problemas — "
           "regressão, classificação e clustering — mostrando que o **tipo** vem da "
           "pergunta, não dos dados."),
        code(PREAMBULO),
        md("## Dados simulados\n\nPacientes com idade, IMC e um marcador contínuo."),
        code('n = 200\n'
             'idade = np.random.uniform(20, 80, n)\n'
             'imc = np.random.normal(26, 4, n)\n'
             'marcador = 0.5 * idade + 1.2 * imc + np.random.normal(0, 5, n)\n'
             'doente = (marcador > np.median(marcador)).astype(int)\n'
             'df = pd.DataFrame({"idade": idade, "imc": imc, "marcador": marcador, "doente": doente})\n'
             'df.head()'),
        md("## 1. Regressão — prever o marcador (contínuo)"),
        code('from sklearn.linear_model import LinearRegression\n'
             'from sklearn.metrics import r2_score\n\n'
             'reg = LinearRegression().fit(df[["idade", "imc"]], df["marcador"])\n'
             'print("R2:", round(r2_score(df["marcador"], reg.predict(df[["idade","imc"]])), 3))'),
        md("## 2. Classificação — prever doente (categórico)"),
        code('from sklearn.linear_model import LogisticRegression\n\n'
             'clf = LogisticRegression().fit(df[["idade", "imc"]], df["doente"])\n'
             'print("acuracia:", round(clf.score(df[["idade", "imc"]], df["doente"]), 3))'),
        md("## 3. Clustering — descobrir grupos (sem rótulo)"),
        code('from sklearn.cluster import KMeans\n'
             'from sklearn.preprocessing import StandardScaler\n\n'
             'Xk = StandardScaler().fit_transform(df[["idade", "imc"]])\n'
             'grupos = KMeans(n_clusters=2, n_init=10, random_state=SEMENTE).fit_predict(Xk)\n\n'
             'figura = go.Figure(go.Scatter(x=df["idade"], y=df["imc"], mode="markers",\n'
             '                              marker=dict(color=grupos, colorscale="Bluered", size=7)))\n'
             'figura.update_layout(title="k-means (k=2) sobre idade e IMC",\n'
             '                     xaxis_title="idade", yaxis_title="imc", height=380,\n'
             '                     showlegend=False, margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()'),
        md("## Exercícios\n\n"
           "**1.** Os grupos coincidem com `doente`? Calcule a concordância.\n\n"
           "**2.** Como transformar \"prever internações\" em regressão? E em "
           "classificação?"),
        code('# @title Solução\n'
             'from sklearn.metrics import adjusted_rand_score\n'
             'print("ARI grupos vs doente:", round(adjusted_rand_score(df["doente"], grupos), 3))\n'
             '# Regressao: prever o numero exato de internacoes (contagem).\n'
             '# Classificacao: prever a categoria "0", "1-2", "3+" internacoes.'),
    ]
    escrever(nb, "01_fundamentos/02_tipos_aprendizagem.ipynb")


def nb_f_representacao():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Representação dos dados\n\n"
           "**Objetivo:** montar a matriz de características $\\mathbf{X}$, identificar "
           "tipos de coluna e aplicar one-hot encoding."),
        code(PREAMBULO),
        md("## Um DataFrame com tipos variados"),
        code('df = pd.DataFrame({\n'
             '    "idade":          [34, 51, 29, 62],\n'
             '    "glicose":        [90, 145, 88, 130],\n'
             '    "tipo_sanguineo": ["A", "O", "AB", "A"],\n'
             '    "estagio":        ["I", "III", "I", "II"],\n'
             '})\n'
             'df'),
        md("## Identificar os tipos\n\nNuméricas, categórica nominal (tipo sanguíneo) "
           "e categórica ordinal (estágio)."),
        code('print(df.dtypes)\n'
             'print("\\nnominal :", ["tipo_sanguineo"])\n'
             'print("ordinal :", ["estagio"])'),
        md("## One-hot encoding da categórica nominal"),
        code('df_oh = pd.get_dummies(df, columns=["tipo_sanguineo"], dtype=int)\n'
             'df_oh'),
        md("## Codificação ordinal (preserva a ordem)"),
        code('ordem = {"I": 1, "II": 2, "III": 3}\n'
             'df_oh["estagio"] = df_oh["estagio"].map(ordem)\n'
             'df_oh'),
        md("## Exercícios\n\n"
           "**1.** Quantas colunas a matriz final tem? Por que o one-hot criou várias "
           "para o tipo sanguíneo?\n\n**2.** Por que **não** faz sentido aplicar "
           "`map({'A':1,'O':2,'AB':3})` ao tipo sanguíneo?"),
        code('# @title Solução\n'
             'print("colunas finais:", df_oh.shape[1], list(df_oh.columns))\n'
             '# One-hot cria uma coluna por categoria porque nao ha ordem entre os tipos.\n'
             '# Codificar A=1, O=2, AB=3 imporia uma ordem falsa (AB > O > A), sem sentido biologico.'),
    ]
    escrever(nb, "01_fundamentos/03_representacao_dados.ipynb")


def nb_f_generalizacao():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Generalização: overfitting e underfitting\n\n"
           "**Objetivo:** reproduzir a experiência do widget do site — ajustar "
           "polinômios de vários graus e ver o erro de treino cair enquanto o de teste "
           "forma um 'U'."),
        code(PREAMBULO),
        md("## Dados: função verdadeira + ruído"),
        code('x_tr = np.sort(np.random.uniform(0, 1, 15))\n'
             'y_tr = np.sin(2*np.pi*x_tr) + np.random.normal(0, 0.25, x_tr.size)\n'
             'x_te = np.sort(np.random.uniform(0, 1, 200))\n'
             'y_te = np.sin(2*np.pi*x_te) + np.random.normal(0, 0.25, x_te.size)\n'
             'print("treino:", x_tr.size, "| teste:", x_te.size)'),
        md("## Ajustar polinômios de grau 1 a 12"),
        code('from sklearn.metrics import mean_squared_error\n\n'
             'graus = list(range(1, 13))\n'
             'err_tr, err_te = [], []\n'
             'for g in graus:\n'
             '    coef = np.polyfit(x_tr, y_tr, g)\n'
             '    err_tr.append(mean_squared_error(y_tr, np.polyval(coef, x_tr)))\n'
             '    err_te.append(mean_squared_error(y_te, np.polyval(coef, x_te)))\n\n'
             'figura = go.Figure()\n'
             'figura.add_trace(go.Scatter(x=graus, y=err_tr, mode="lines+markers",\n'
             '                            line=dict(color=AZUL), name="treino"))\n'
             'figura.add_trace(go.Scatter(x=graus, y=err_te, mode="lines+markers",\n'
             '                            line=dict(color=VERMELHO), name="teste"))\n'
             'figura.update_yaxes(type="log")\n'
             'figura.update_layout(title="Erro de treino x teste (escala log)",\n'
             '                     xaxis_title="grau do polinomio", yaxis_title="MSE",\n'
             '                     height=360, margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()'),
        md("## Visualizar três regimes"),
        code('from plotly.subplots import make_subplots\n'
             'xx = np.linspace(0, 1, 300)\n'
             'figura = make_subplots(rows=1, cols=3,\n'
             '                       subplot_titles=("grau 1 — underfitting", "grau 4 — equilibrio", "grau 12 — overfitting"))\n'
             'coluna = 1\n'
             'for g in [1, 4, 12]:\n'
             '    coef = np.polyfit(x_tr, y_tr, g)\n'
             '    figura.add_trace(go.Scatter(x=x_tr, y=y_tr, mode="markers",\n'
             '                                marker=dict(color=VERMELHO, size=6), showlegend=False), row=1, col=coluna)\n'
             '    figura.add_trace(go.Scatter(x=xx, y=np.sin(2*np.pi*xx), mode="lines",\n'
             '                                line=dict(color=VERDE, dash="dash"), showlegend=False), row=1, col=coluna)\n'
             '    figura.add_trace(go.Scatter(x=xx, y=np.polyval(coef, xx), mode="lines",\n'
             '                                line=dict(color=AZUL), showlegend=False), row=1, col=coluna)\n'
             '    coluna += 1\n'
             'figura.update_yaxes(range=[-2, 2])\n'
             'figura.update_layout(height=320, margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()'),
        md("## Exercícios\n\n"
           "**1.** Qual grau minimiza o erro de teste? Coincide com o do treino?\n\n"
           "**2.** Aumente o treino para 150 pontos. O overfitting do grau 12 diminui?"),
        code('# @title Solução\n'
             'print("grau otimo (teste):", graus[int(np.argmin(err_te))])\n'
             'print("grau otimo (treino):", graus[int(np.argmin(err_tr))], "-> o treino sempre melhora com mais grau")\n'
             '# Com mais dados, o polinomio de grau alto tem menos liberdade para se colar\n'
             '# ao ruido: o overfitting diminui.'),
    ]
    escrever(nb, "01_fundamentos/04_generalizacao.ipynb")


def nb_f_validacao():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Validação cruzada\n\n"
           "**Objetivo:** comparar uma única divisão treino/teste com a validação "
           "cruzada k-fold e ver por que a média do CV é mais confiável."),
        code(PREAMBULO),
        code('from sklearn.datasets import load_iris\n'
             'X, y = load_iris(return_X_y=True)'),
        md("## Uma divisão só varia bastante"),
        code('from sklearn.model_selection import train_test_split\n'
             'from sklearn.neighbors import KNeighborsClassifier\n\n'
             'for s in range(5):\n'
             '    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=s)\n'
             '    acc = KNeighborsClassifier().fit(Xtr, ytr).score(Xte, yte)\n'
             '    print("seed =", s, "-> acuracia =", round(acc, 3))'),
        md("## k-fold: estimativa estável (com pipeline, sem vazamento)"),
        code('from sklearn.model_selection import cross_val_score\n'
             'from sklearn.pipeline import make_pipeline\n'
             'from sklearn.preprocessing import StandardScaler\n\n'
             'modelo = make_pipeline(StandardScaler(), KNeighborsClassifier())\n'
             'scores = cross_val_score(modelo, X, y, cv=5)\n'
             'print("folds:", np.round(scores, 3))\n'
             'print("CV:", round(scores.mean(), 3), "+/-", round(scores.std(), 3))'),
        md("## Exercícios\n\n"
           "**1.** Rode com cv=10. A média muda muito? E o desvio?\n\n"
           "**2.** Use `StratifiedKFold` e confirme que cada fold preserva a proporção "
           "das classes."),
        code('# @title Solução\n'
             's10 = cross_val_score(modelo, X, y, cv=10)\n'
             'print("cv=10:", round(s10.mean(), 3), "+/-", round(s10.std(), 3))\n'
             'from sklearn.model_selection import StratifiedKFold\n'
             'skf = StratifiedKFold(n_splits=5)\n'
             'for i, (_, te) in enumerate(skf.split(X, y)):\n'
             '    vals, cnt = np.unique(y[te], return_counts=True)\n'
             '    print("fold", i, ":", dict(zip(vals.tolist(), cnt.tolist())))'),
    ]
    escrever(nb, "01_fundamentos/05_validacao_cruzada.ipynb")


def nb_f_metricas_reg():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Métricas de avaliação — regressão\n\n"
           "**Objetivo:** calcular MAE, RMSE e R², e ver como um único outlier afeta o "
           "RMSE muito mais que o MAE."),
        code(PREAMBULO),
        code('from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n\n'
             'y_true = np.array([3.0, 5.0, 2.5, 7.0, 4.2, 6.1])\n'
             'y_pred = np.array([2.8, 5.5, 2.0, 8.0, 4.0, 6.4])\n'
             'print("MAE :", round(mean_absolute_error(y_true, y_pred), 3))\n'
             'print("RMSE:", round(np.sqrt(mean_squared_error(y_true, y_pred)), 3))\n'
             'print("R2  :", round(r2_score(y_true, y_pred), 3))'),
        md("## O efeito de um outlier\n\nEstragamos uma única previsão."),
        code('y_pred_out = y_pred.copy()\n'
             'y_pred_out[3] = 20.0   # erro grosseiro em um ponto\n'
             'print("MAE :", round(mean_absolute_error(y_true, y_pred_out), 3))\n'
             'print("RMSE:", round(np.sqrt(mean_squared_error(y_true, y_pred_out)), 3))\n'
             '# RMSE dispara; MAE sobe pouco.'),
        md("## Exercícios\n\n"
           "**1.** Um modelo prevê sempre a média de `y_true`. Qual o R²?\n\n"
           "**2.** Quando preferir otimizar RMSE em vez de MAE?"),
        code('# @title Solução\n'
             'media = np.full_like(y_true, y_true.mean())\n'
             'print("R2 prevendo a media:", round(r2_score(y_true, media), 3))   # = 0\n'
             '# Prefira RMSE quando erros grandes sao desproporcionalmente perigosos —\n'
             '# ex.: prever a dose de um medicamento de janela terapeutica estreita.'),
    ]
    escrever(nb, "01_fundamentos/06_metricas_regressao.ipynb")


def nb_f_metricas_clf():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Métricas de avaliação — classificação\n\n"
           "**Objetivo:** num conjunto **desbalanceado**, mostrar como a acurácia "
           "engana, calcular a matriz de confusão e o `classification_report`, e "
           "desenhar a curva ROC."),
        code(PREAMBULO),
        md("## Dados desbalanceados (5% de positivos)"),
        code('from sklearn.datasets import make_classification\n'
             'X, y = make_classification(n_samples=2000, weights=[0.95, 0.05],\n'
             '                           n_informative=5, random_state=SEMENTE)\n'
             'print("proporcao de positivos:", round(y.mean(), 3))'),
        md("## O classificador trivial 'tudo negativo'"),
        code('print("acuracia prevendo sempre 0:", round((y == 0).mean(), 3), " <- alta e inutil")'),
        md("## Um modelo de verdade"),
        code('from sklearn.model_selection import train_test_split\n'
             'from sklearn.linear_model import LogisticRegression\n'
             'from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score\n\n'
             'Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=SEMENTE)\n'
             'clf = LogisticRegression(max_iter=500).fit(Xtr, ytr)\n'
             'pred = clf.predict(Xte)\n'
             'print(confusion_matrix(yte, pred))\n'
             'print(classification_report(yte, pred, digits=3))'),
        md("## Curva ROC e AUC"),
        code('from sklearn.metrics import roc_curve\n\n'
             'proba = clf.predict_proba(Xte)[:, 1]\n'
             'fpr, tpr, _ = roc_curve(yte, proba)\n'
             'print("AUC:", round(roc_auc_score(yte, proba), 3))\n\n'
             'figura = go.Figure()\n'
             'figura.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", line=dict(color=AZUL), name="LogReg"))\n'
             'figura.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",\n'
             '                            line=dict(color=SUAVE, dash="dash"), name="aleatorio"))\n'
             'figura.update_layout(title="Curva ROC", xaxis_title="taxa de falsos positivos",\n'
             '                     yaxis_title="taxa de verdadeiros positivos", height=380,\n'
             '                     margin=dict(l=10, r=10, t=50, b=10))\n'
             'figura.show()'),
        md("## Exercícios\n\n"
           "**1.** Baixe o limiar para 0,2. O que acontece com recall e precisão da "
           "classe positiva?\n\n**2.** Por que a AUC não muda ao alterar o limiar, mas "
           "a acurácia muda?"),
        code('# @title Solução\n'
             'from sklearn.metrics import precision_score, recall_score\n'
             'for thr in [0.5, 0.2]:\n'
             '    p = (proba >= thr).astype(int)\n'
             '    print("limiar", thr, ": recall", round(recall_score(yte, p), 3),\n'
             '          "| precisao", round(precision_score(yte, p, zero_division=0), 3))\n'
             '# A AUC integra TODOS os limiares, entao nao depende de um limiar especifico;\n'
             '# acuracia, precisao e recall sao medidas em UM limiar.'),
    ]
    escrever(nb, "01_fundamentos/07_metricas_classificacao.ipynb")


def nb_f_preprocessamento():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Pré-processamento de dados\n\n"
           "**Objetivo:** partir de dados sujos (escalas diferentes, faltantes, "
           "categorias em texto), montar um `ColumnTransformer` completo e medir o "
           "impacto da padronização num modelo baseado em distância."),
        code(PREAMBULO),
        md("## Um conjunto de dados sujo"),
        code('df = pd.DataFrame({\n'
             '    "idade":      [34, 51, np.nan, 62, 45, 29],\n'
             '    "colesterol": [190, 240, 210, np.nan, 260, 175],\n'
             '    "sexo":       ["F", "M", "M", "F", np.nan, "F"],\n'
             '    "risco":      [0, 1, 0, 1, 1, 0],\n'
             '})\n'
             'df'),
        md("## Pipeline: imputação + escala (num) e imputação + one-hot (cat)"),
        code('from sklearn.compose import ColumnTransformer\n'
             'from sklearn.pipeline import Pipeline\n'
             'from sklearn.preprocessing import StandardScaler, OneHotEncoder\n'
             'from sklearn.impute import SimpleImputer\n\n'
             'num = ["idade", "colesterol"]\n'
             'cat = ["sexo"]\n'
             'num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])\n'
             'cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder())])\n'
             'pre = ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)])\n'
             'Xt = pre.fit_transform(df[num + cat])\n'
             'print("matriz processada:\\n", np.round(Xt, 2))'),
        md("## Padronização importa para o k-NN?\n\n"
           "Comparamos a acurácia (validação cruzada) com e sem escala, no breast "
           "cancer."),
        code('from sklearn.neighbors import KNeighborsClassifier\n'
             'from sklearn.model_selection import cross_val_score\n'
             'from sklearn.datasets import load_breast_cancer\n\n'
             'Xbc, ybc = load_breast_cancer(return_X_y=True)\n'
             'sem = KNeighborsClassifier()\n'
             'com = Pipeline([("sc", StandardScaler()), ("knn", KNeighborsClassifier())])\n'
             'print("sem escala:", round(cross_val_score(sem, Xbc, ybc, cv=5).mean(), 3))\n'
             'print("com escala:", round(cross_val_score(com, Xbc, ybc, cv=5).mean(), 3))'),
        md("## Exercícios\n\n"
           "**1.** De quanto foi o ganho da padronização? Por que ele aparece num "
           "modelo de distância?\n\n**2.** Troque `StandardScaler` por `MinMaxScaler`. "
           "Muda muito?"),
        code('# @title Solução\n'
             'from sklearn.preprocessing import MinMaxScaler\n'
             'mm = Pipeline([("sc", MinMaxScaler()), ("knn", KNeighborsClassifier())])\n'
             'print("min-max:", round(cross_val_score(mm, Xbc, ybc, cv=5).mean(), 3))\n'
             '# O ganho aparece porque o k-NN soma distancias entre caracteristicas; sem\n'
             '# escala, as de maior amplitude dominam. Standard e MinMax dao resultados parecidos aqui.'),
    ]
    escrever(nb, "01_fundamentos/08_preprocessamento.ipynb")


CONSTRUTORES = [
    nb_ferramentas,
    nb_f_introducao,
    nb_f_tipos,
    nb_f_representacao,
    nb_f_generalizacao,
    nb_f_validacao,
    nb_f_metricas_reg,
    nb_f_metricas_clf,
    nb_f_preprocessamento,
    nb_reg_linear,
    nb_reg_multipla,
    nb_reg_polinomial,
    nb_regularizacao,
    nb_reg_logistica,
    nb_knn,
    nb_arvores,
    nb_naive_bayes,
    nb_svm,
    nb_random_forest,
    nb_gradient_boosting,
    nb_xgboost,
    nb_stacking,
    nb_kmeans,
    nb_hierarquico,
    nb_pca,
    nb_tsne_umap,
    nb_perceptron,
    nb_backprop,
    nb_ativacoes,
    nb_deep_learning,
    nb_ex_regressao,
    nb_ex_classificacao,
    nb_ex_validacao,
    nb_ex_clustering_pca,
    nb_ex_pipeline,
]

if __name__ == "__main__":
    for construir in CONSTRUTORES:
        construir()
