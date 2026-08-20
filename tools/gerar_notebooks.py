# -*- coding: utf-8 -*-
"""Gera os notebooks .ipynb do capítulo 1 (Fundamentos).

Uma função por notebook. Todos seguem as convenções do curso:
- célula Markdown de objetivo no topo;
- bibliotecas padrão (numpy, pandas, matplotlib, scikit-learn);
- dados reprodutíveis com numpy.random.default_rng(42);
- paleta do curso nas visualizações;
- 2 exercícios ao final, com solução escondida em célula própria.

Uso:
    python tools/gerar_notebooks.py
"""

from pathlib import Path
import nbformat as nbf

BASE = Path(__file__).parent.parent

# Preâmbulo de estilo — importado no início de cada notebook.
PREAMBLE = '''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paleta do curso (idêntica ao site)
INK, PAPER = "#1c1e15", "#fbfdf3"
BLUE, RED, GREEN, MUTED = "#3266ad", "#c0392b", "#1a7a4a", "#6b7050"

plt.rcParams.update({
    "font.family": "serif", "font.size": 12,
    "figure.facecolor": PAPER, "axes.facecolor": PAPER,
    "axes.edgecolor": "#bcc79a", "axes.grid": True,
    "grid.color": "#e4ead0", "grid.linewidth": 0.7,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
})
rng = np.random.default_rng(42)'''


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(t):
    return nbf.v4.new_code_cell(t)


def write(nb, path):
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    (BASE / path).write_text(nbf.writes(nb), encoding="utf-8")
    print("OK", path)


def solution(text):
    """Célula de código com a solução, precedida por um cabeçalho oculto."""
    return code("# @title Solução (clique para revelar)\n" + text)


# ══════════════════════════════════════════════════════════════════════════
def nb_introducao():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# O que é aprendizagem de máquina?\n\n"
           "**Objetivo:** rodar o \"olá, mundo\" da aprendizagem supervisionada. "
           "Vamos carregar o conjunto Iris, treinar um classificador em poucas "
           "linhas e medir seu desempenho em dados que ele não viu — ilustrando o "
           "ciclo mínimo de um projeto de ML: `dados → treino → avaliação`."),
        code(PREAMBLE),
        md("## 1. Carregar os dados\n\n"
           "O Iris tem 150 flores, 4 características e 3 espécies. É o conjunto "
           "didático mais famoso da área."),
        code("from sklearn.datasets import load_iris\n\n"
             "iris = load_iris(as_frame=True)\n"
             "X, y = iris.data, iris.target\n"
             "print(X.shape, '->', iris.target_names)\n"
             "X.head()"),
        md("## 2. Separar treino e teste\n\n"
           "Avaliamos o modelo em dados **nunca vistos** durante o treino — a "
           "essência da generalização."),
        code("from sklearn.model_selection import train_test_split\n\n"
             "X_tr, X_te, y_tr, y_te = train_test_split(\n"
             "    X, y, test_size=0.3, random_state=42, stratify=y)\n"
             "print('treino:', X_tr.shape[0], ' teste:', X_te.shape[0])"),
        md("## 3. Treinar e avaliar\n\n"
           "Um classificador k-NN em três linhas. Repare que `fit` aprende e "
           "`score` mede a acurácia no teste."),
        code("from sklearn.neighbors import KNeighborsClassifier\n\n"
             "modelo = KNeighborsClassifier(n_neighbors=5)\n"
             "modelo.fit(X_tr, y_tr)\n"
             "print(f'Acurácia no teste: {modelo.score(X_te, y_te):.3f}')"),
        md("## Exercícios\n\n"
           "**1.** Troque `n_neighbors` para 1 e para 50. O que acontece com a "
           "acurácia de treino e de teste? Relacione com overfitting/underfitting.\n\n"
           "**2.** Substitua o k-NN por uma `LogisticRegression`. O desempenho "
           "muda muito?"),
        solution("for k in [1, 5, 50]:\n"
                 "    m = KNeighborsClassifier(n_neighbors=k).fit(X_tr, y_tr)\n"
                 "    print(f'k={k:2d} | treino={m.score(X_tr,y_tr):.3f} | teste={m.score(X_te,y_te):.3f}')\n\n"
                 "from sklearn.linear_model import LogisticRegression\n"
                 "lr = LogisticRegression(max_iter=200).fit(X_tr, y_tr)\n"
                 "print('LogReg teste:', round(lr.score(X_te, y_te), 3))"),
    ]
    write(nb, "01_fundamentos/01_introducao_ml.ipynb")


def nb_tipos():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Tipos de aprendizagem\n\n"
           "**Objetivo:** usar um mesmo conjunto de dados para formular três "
           "problemas diferentes — regressão, classificação e clustering — "
           "mostrando que o **tipo** vem da pergunta, não dos dados."),
        code(PREAMBLE),
        md("## Dados simulados\n\n"
           "Geramos pacientes com idade, IMC e um marcador contínuo."),
        code("n = 200\n"
             "idade = rng.uniform(20, 80, n)\n"
             "imc = rng.normal(26, 4, n)\n"
             "marcador = 0.5 * idade + 1.2 * imc + rng.normal(0, 5, n)\n"
             "doente = (marcador > np.median(marcador)).astype(int)\n"
             "df = pd.DataFrame({'idade': idade, 'imc': imc,\n"
             "                   'marcador': marcador, 'doente': doente})\n"
             "df.head()"),
        md("## 1. Regressão — prever o marcador (contínuo)"),
        code("from sklearn.linear_model import LinearRegression\n"
             "from sklearn.metrics import r2_score\n\n"
             "Xr, yr = df[['idade', 'imc']], df['marcador']\n"
             "reg = LinearRegression().fit(Xr, yr)\n"
             "print('R²:', round(r2_score(yr, reg.predict(Xr)), 3))"),
        md("## 2. Classificação — prever doente (categórico)"),
        code("from sklearn.linear_model import LogisticRegression\n\n"
             "Xc, yc = df[['idade', 'imc']], df['doente']\n"
             "clf = LogisticRegression().fit(Xc, yc)\n"
             "print('Acurácia:', round(clf.score(Xc, yc), 3))"),
        md("## 3. Clustering — descobrir grupos (sem rótulo)"),
        code("from sklearn.cluster import KMeans\n"
             "from sklearn.preprocessing import StandardScaler\n\n"
             "Xk = StandardScaler().fit_transform(df[['idade', 'imc']])\n"
             "grupos = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(Xk)\n"
             "plt.scatter(df['idade'], df['imc'], c=grupos, cmap='coolwarm', s=18)\n"
             "plt.xlabel('idade'); plt.ylabel('imc'); plt.title('k-means (k=2)')\n"
             "plt.show()"),
        md("## Exercícios\n\n"
           "**1.** No clustering, os grupos encontrados coincidem com a coluna "
           "`doente`? Calcule a concordância.\n\n"
           "**2.** Reformule: como transformar \"prever o número de internações\" "
           "num problema de regressão? E de classificação?"),
        solution("from sklearn.metrics import adjusted_rand_score\n"
                 "print('ARI grupos vs doente:', round(adjusted_rand_score(df['doente'], grupos), 3))\n"
                 "# Regressão: prever o número exato de internações (contagem).\n"
                 "# Classificação: prever a categoria '0, 1-2, 3+' internações."),
    ]
    write(nb, "01_fundamentos/02_tipos_aprendizagem.ipynb")


def nb_representacao():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Representação dos dados\n\n"
           "**Objetivo:** montar a matriz de características $\\mathbf{X}$, "
           "identificar tipos de coluna e aplicar one-hot encoding."),
        code(PREAMBLE),
        md("## Um DataFrame com tipos variados"),
        code("df = pd.DataFrame({\n"
             "    'idade':          [34, 51, 29, 62],\n"
             "    'glicose':        [90, 145, 88, 130],\n"
             "    'tipo_sanguineo': ['A', 'O', 'AB', 'A'],\n"
             "    'estagio':        ['I', 'III', 'I', 'II'],\n"
             "})\n"
             "df"),
        md("## Identificar os tipos\n\n"
           "Numéricas contínuas, categórica nominal (tipo sanguíneo) e categórica "
           "ordinal (estágio)."),
        code("print(df.dtypes)\n"
             "print('\\nnominal :', ['tipo_sanguineo'])\n"
             "print('ordinal :', ['estagio'])"),
        md("## One-hot encoding da categórica nominal"),
        code("df_oh = pd.get_dummies(df, columns=['tipo_sanguineo'], dtype=int)\n"
             "df_oh"),
        md("## Codificação ordinal (preserva a ordem)"),
        code("ordem = {'I': 1, 'II': 2, 'III': 3}\n"
             "df_oh['estagio'] = df_oh['estagio'].map(ordem)\n"
             "df_oh"),
        md("## Exercícios\n\n"
           "**1.** Quantas colunas a matriz final tem? Por que o one-hot criou "
           "várias colunas para tipo sanguíneo?\n\n"
           "**2.** Por que **não** faz sentido aplicar `map({'A':1,'O':2,'AB':3})` "
           "ao tipo sanguíneo?"),
        solution("print('colunas finais:', df_oh.shape[1], list(df_oh.columns))\n"
                 "# One-hot cria uma coluna por categoria porque não há ordem entre\n"
                 "# os tipos. Codificar A=1, O=2, AB=3 imporia uma ordem falsa, fazendo\n"
                 "# o modelo achar que AB > O > A — o que não tem sentido biológico."),
    ]
    write(nb, "01_fundamentos/03_representacao_dados.ipynb")


def nb_generalizacao():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Generalização: overfitting e underfitting\n\n"
           "**Objetivo:** reproduzir em Python a experiência do widget do site — "
           "ajustar polinômios de vários graus e ver o erro de treino cair "
           "enquanto o erro de teste forma um 'U'."),
        code(PREAMBLE),
        md("## Dados: função verdadeira + ruído"),
        code("f_true = lambda x: np.sin(2 * np.pi * x)\n"
             "x_tr = np.sort(rng.uniform(0, 1, 15))\n"
             "y_tr = f_true(x_tr) + rng.normal(0, 0.25, x_tr.size)\n"
             "x_te = np.sort(rng.uniform(0, 1, 200))\n"
             "y_te = f_true(x_te) + rng.normal(0, 0.25, x_te.size)"),
        md("## Ajustar polinômios de grau 1 a 12"),
        code("from numpy.polynomial import polynomial as P\n"
             "from sklearn.metrics import mean_squared_error as mse\n\n"
             "graus = range(1, 13)\n"
             "err_tr, err_te = [], []\n"
             "for g in graus:\n"
             "    coef = np.polyfit(x_tr, y_tr, g)\n"
             "    err_tr.append(mse(y_tr, np.polyval(coef, x_tr)))\n"
             "    err_te.append(mse(y_te, np.polyval(coef, x_te)))\n\n"
             "plt.plot(list(graus), err_tr, 'o-', color=BLUE, label='treino')\n"
             "plt.plot(list(graus), err_te, 's-', color=RED, label='teste')\n"
             "plt.yscale('log'); plt.xlabel('grau do polinômio'); plt.ylabel('MSE (log)')\n"
             "plt.legend(); plt.title('Erro de treino x teste'); plt.show()"),
        md("## Visualizar três regimes"),
        code("xx = np.linspace(0, 1, 300)\n"
             "fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))\n"
             "for ax, g, t in zip(axes, [1, 4, 12],\n"
             "                    ['underfitting', 'equilíbrio', 'overfitting']):\n"
             "    coef = np.polyfit(x_tr, y_tr, g)\n"
             "    ax.scatter(x_tr, y_tr, color=RED, s=20)\n"
             "    ax.plot(xx, f_true(xx), '--', color=GREEN)\n"
             "    ax.plot(xx, np.polyval(coef, xx), color=BLUE)\n"
             "    ax.set_ylim(-2, 2); ax.set_title(f'grau {g} — {t}')\n"
             "plt.tight_layout(); plt.show()"),
        md("## Exercícios\n\n"
           "**1.** Qual grau minimiza o erro de teste? Ele coincide com o que "
           "minimiza o erro de treino?\n\n"
           "**2.** Aumente o tamanho do treino para 150 pontos. O overfitting do "
           "grau 12 diminui?"),
        solution("melhor = list(graus)[int(np.argmin(err_te))]\n"
                 "print('grau ótimo (teste):', melhor)\n"
                 "print('grau ótimo (treino):', list(graus)[int(np.argmin(err_tr))],\n"
                 "      '-> o treino sempre melhora com mais grau')\n"
                 "# Com mais dados, o modelo de grau alto tem menos liberdade para\n"
                 "# se colar ao ruído: o overfitting diminui."),
    ]
    write(nb, "01_fundamentos/04_generalizacao.ipynb")


def nb_validacao():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Validação cruzada\n\n"
           "**Objetivo:** comparar uma única divisão treino/teste com a validação "
           "cruzada k-fold e ver por que a média do CV é mais confiável."),
        code(PREAMBLE),
        code("from sklearn.datasets import load_iris\n"
             "X, y = load_iris(return_X_y=True)"),
        md("## Uma divisão só varia bastante"),
        code("from sklearn.model_selection import train_test_split\n"
             "from sklearn.neighbors import KNeighborsClassifier\n\n"
             "for s in range(5):\n"
             "    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=s)\n"
             "    acc = KNeighborsClassifier().fit(Xtr, ytr).score(Xte, yte)\n"
             "    print(f'seed={s} -> acurácia={acc:.3f}')"),
        md("## k-fold: estimativa estável (com pipeline, sem vazamento)"),
        code("from sklearn.model_selection import cross_val_score\n"
             "from sklearn.pipeline import make_pipeline\n"
             "from sklearn.preprocessing import StandardScaler\n\n"
             "modelo = make_pipeline(StandardScaler(), KNeighborsClassifier())\n"
             "scores = cross_val_score(modelo, X, y, cv=5)\n"
             "print('folds:', np.round(scores, 3))\n"
             "print(f'CV: {scores.mean():.3f} ± {scores.std():.3f}')"),
        md("## Exercícios\n\n"
           "**1.** Rode a validação cruzada com cv=10. A média muda muito? E o "
           "desvio?\n\n"
           "**2.** Use `StratifiedKFold` e confirme que cada fold preserva a "
           "proporção das classes."),
        solution("s10 = cross_val_score(modelo, X, y, cv=10)\n"
                 "print(f'cv=10: {s10.mean():.3f} ± {s10.std():.3f}')\n\n"
                 "from sklearn.model_selection import StratifiedKFold\n"
                 "skf = StratifiedKFold(n_splits=5)\n"
                 "for i, (_, te) in enumerate(skf.split(X, y)):\n"
                 "    vals, cnt = np.unique(y[te], return_counts=True)\n"
                 "    print(f'fold {i}: {dict(zip(vals, cnt))}')"),
    ]
    write(nb, "01_fundamentos/05_validacao_cruzada.ipynb")


def nb_metricas_reg():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Métricas de avaliação — regressão\n\n"
           "**Objetivo:** calcular MAE, RMSE e R², e ver como um único outlier "
           "afeta RMSE muito mais que MAE."),
        code(PREAMBLE),
        code("from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score\n\n"
             "y_true = np.array([3.0, 5.0, 2.5, 7.0, 4.2, 6.1])\n"
             "y_pred = np.array([2.8, 5.5, 2.0, 8.0, 4.0, 6.4])\n\n"
             "print('MAE :', round(mean_absolute_error(y_true, y_pred), 3))\n"
             "print('RMSE:', round(np.sqrt(mean_squared_error(y_true, y_pred)), 3))\n"
             "print('R²  :', round(r2_score(y_true, y_pred), 3))"),
        md("## O efeito de um outlier\n\n"
           "Estragamos uma única previsão e observamos as métricas."),
        code("y_pred_out = y_pred.copy()\n"
             "y_pred_out[3] = 20.0   # erro grosseiro em um ponto\n\n"
             "print('MAE :', round(mean_absolute_error(y_true, y_pred_out), 3))\n"
             "print('RMSE:', round(np.sqrt(mean_squared_error(y_true, y_pred_out)), 3))\n"
             "# RMSE dispara; MAE sobe pouco."),
        md("## Exercícios\n\n"
           "**1.** Um modelo prevê sempre a média de `y_true`. Qual o R²? Confirme "
           "no código.\n\n"
           "**2.** Em que situação clínica você prefere otimizar RMSE em vez de "
           "MAE?"),
        solution("media = np.full_like(y_true, y_true.mean())\n"
                 "print('R² prevendo a média:', round(r2_score(y_true, media), 3))  # = 0\n"
                 "# Prefira RMSE quando erros grandes são desproporcionalmente\n"
                 "# perigosos — ex.: prever a dose de um medicamento de janela estreita."),
    ]
    write(nb, "01_fundamentos/06_metricas_regressao.ipynb")


def nb_metricas_clf():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Métricas de avaliação — classificação\n\n"
           "**Objetivo:** num conjunto **desbalanceado**, mostrar como a acurácia "
           "engana, calcular a matriz de confusão e o `classification_report`, e "
           "desenhar a curva ROC."),
        code(PREAMBLE),
        md("## Dados desbalanceados (5% de positivos)"),
        code("from sklearn.datasets import make_classification\n"
             "X, y = make_classification(n_samples=2000, weights=[0.95, 0.05],\n"
             "                           n_informative=5, random_state=42)\n"
             "print('proporção de positivos:', round(y.mean(), 3))"),
        md("## O classificador trivial 'tudo negativo'"),
        code("acc_trivial = (y == 0).mean()\n"
             "print(f'Acurácia prevendo sempre 0: {acc_trivial:.3f}  <- alta e inútil')"),
        md("## Um modelo de verdade"),
        code("from sklearn.model_selection import train_test_split\n"
             "from sklearn.linear_model import LogisticRegression\n"
             "from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score\n\n"
             "Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3,\n"
             "                                      stratify=y, random_state=42)\n"
             "clf = LogisticRegression(max_iter=500).fit(Xtr, ytr)\n"
             "pred = clf.predict(Xte)\n"
             "print(confusion_matrix(yte, pred))\n"
             "print(classification_report(yte, pred, digits=3))"),
        md("## Curva ROC e AUC"),
        code("from sklearn.metrics import RocCurveDisplay\n\n"
             "proba = clf.predict_proba(Xte)[:, 1]\n"
             "print('AUC:', round(roc_auc_score(yte, proba), 3))\n"
             "RocCurveDisplay.from_predictions(yte, proba, name='LogReg')\n"
             "plt.plot([0, 1], [0, 1], '--', color=MUTED)\n"
             "plt.title('Curva ROC'); plt.show()"),
        md("## Exercícios\n\n"
           "**1.** Baixe o limiar de decisão para 0,2 (em vez de 0,5). O que "
           "acontece com recall e precisão da classe positiva?\n\n"
           "**2.** Por que a AUC não muda ao alterar o limiar, mas a acurácia "
           "muda?"),
        solution("from sklearn.metrics import precision_score, recall_score\n"
                 "for thr in [0.5, 0.2]:\n"
                 "    p = (proba >= thr).astype(int)\n"
                 "    print(f'limiar={thr}: recall={recall_score(yte, p):.3f} '\n"
                 "          f'precisão={precision_score(yte, p):.3f}')\n"
                 "# A AUC integra TODOS os limiares, então não depende de um limiar\n"
                 "# específico; acurácia, precisão e recall são medidas em um limiar."),
    ]
    write(nb, "01_fundamentos/07_metricas_classificacao.ipynb")


def nb_preprocessamento():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Pré-processamento de dados\n\n"
           "**Objetivo:** partir de dados sujos (escalas diferentes, faltantes, "
           "categorias em texto), montar um `ColumnTransformer` completo e medir o "
           "impacto da padronização num modelo baseado em distância."),
        code(PREAMBLE),
        md("## Um conjunto de dados sujo"),
        code("df = pd.DataFrame({\n"
             "    'idade':     [34, 51, np.nan, 62, 45, 29],\n"
             "    'colesterol':[190, 240, 210, np.nan, 260, 175],\n"
             "    'sexo':      ['F', 'M', 'M', 'F', np.nan, 'F'],\n"
             "    'risco':     [0, 1, 0, 1, 1, 0],\n"
             "})\n"
             "df"),
        md("## Pipeline: imputação + escala (num) e imputação + one-hot (cat)"),
        code("from sklearn.compose import ColumnTransformer\n"
             "from sklearn.pipeline import Pipeline\n"
             "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n"
             "from sklearn.impute import SimpleImputer\n\n"
             "num = ['idade', 'colesterol']\n"
             "cat = ['sexo']\n"
             "num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')),\n"
             "                     ('sc', StandardScaler())])\n"
             "cat_pipe = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),\n"
             "                     ('oh', OneHotEncoder())])\n"
             "pre = ColumnTransformer([('num', num_pipe, num), ('cat', cat_pipe, cat)])\n"
             "Xt = pre.fit_transform(df[num + cat])\n"
             "print('matriz processada:\\n', np.round(Xt, 2))"),
        md("## Padronização importa para k-NN?\n\n"
           "Comparamos a acurácia (validação cruzada) com e sem escala."),
        code("from sklearn.neighbors import KNeighborsClassifier\n"
             "from sklearn.model_selection import cross_val_score\n"
             "from sklearn.datasets import load_breast_cancer\n\n"
             "Xbc, ybc = load_breast_cancer(return_X_y=True)\n"
             "sem = KNeighborsClassifier()\n"
             "com = Pipeline([('sc', StandardScaler()), ('knn', KNeighborsClassifier())])\n"
             "print('sem escala:', round(cross_val_score(sem, Xbc, ybc, cv=5).mean(), 3))\n"
             "print('com escala:', round(cross_val_score(com, Xbc, ybc, cv=5).mean(), 3))"),
        md("## Exercícios\n\n"
           "**1.** De quanto foi o ganho da padronização no k-NN? Por que ele "
           "aparece justamente num modelo de distância?\n\n"
           "**2.** Troque `StandardScaler` por `MinMaxScaler`. O resultado muda "
           "muito?"),
        solution("from sklearn.preprocessing import MinMaxScaler\n"
                 "mm = Pipeline([('sc', MinMaxScaler()), ('knn', KNeighborsClassifier())])\n"
                 "print('min-max:', round(cross_val_score(mm, Xbc, ybc, cv=5).mean(), 3))\n"
                 "# O ganho aparece porque o k-NN soma distâncias entre características;\n"
                 "# sem escala, as de maior amplitude dominam. Standard e MinMax costumam\n"
                 "# dar resultados parecidos aqui."),
    ]
    write(nb, "01_fundamentos/08_preprocessamento.ipynb")


if __name__ == "__main__":
    nb_introducao()
    nb_tipos()
    nb_representacao()
    nb_generalizacao()
    nb_validacao()
    nb_metricas_reg()
    nb_metricas_clf()
    nb_preprocessamento()
    print("\nNotebooks do capítulo 1 gerados.")
