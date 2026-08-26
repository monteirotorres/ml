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


CONSTRUTORES = [
    nb_ferramentas,
]

if __name__ == "__main__":
    for construir in CONSTRUTORES:
        construir()
