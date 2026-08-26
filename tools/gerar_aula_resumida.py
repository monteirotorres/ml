"""Gera a versao RESUMIDA da aula pratica de bioinformatica (Inibidores de AChE).

Uma trilha enxuta, do inicio ao fim em ~1h30: no maximo 30 celulas, mais curtas,
mais visuais, sem definir funcoes. Cobre as partes essenciais do estudo de caso
completo (08_aula_bioinfo/aula.ipynb): dado bruto do ChEMBL, curadoria, descritores
e fingerprint, particao por esqueleto, comparacao de modelos, interpretabilidade e
o classificador em uso.

Uso:
    python tools/gerar_aula_resumida.py
Gera 08_aula_bioinfo/aula_resumida.ipynb
"""
import json
import os

CELULAS = []


def md(texto):
    CELULAS.append({"cell_type": "markdown", "metadata": {},
                    "source": texto.strip("\n").splitlines(keepends=True)})


def code(texto):
    CELULAS.append({"cell_type": "code", "metadata": {}, "outputs": [],
                    "execution_count": None,
                    "source": texto.strip("\n").splitlines(keepends=True)})


# ---------------------------------------------------------------------------
md(r"""
# Aula prática (versão resumida) — Inibidores de AChE

**Aprendizado de máquina aplicado à bioinformática, do início ao fim em ~1h30.**

Esta é a **trilha enxuta** do estudo de caso: partimos de dados brutos do
[ChEMBL](https://www.ebi.ac.uk/chembl/) sobre a **acetilcolinesterase (AChE)** —
alvo clássico em Alzheimer — e chegamos a um **classificador** que rotula uma
molécula como **FORTE** ou **FRACO**, ou se **abstém** quando não tem base para
opinar. No caminho vemos o que realmente importa em um projeto de ML aplicado:
curadoria, como representar a molécula em números, a armadilha da partição
otimista, a comparação honesta de modelos e a interpretação do resultado.

> A **versão completa** (`aula.ipynb`) aprofunda cada etapa — proveniência dos
> dados, PyTorch e TensorBoard, SHAP, domínio de aplicabilidade, predição
> conformal e triagem virtual. Use esta versão para ter a visão de conjunto;
> volte à completa para os detalhes.
""")

# ----------------------------- Ambiente ------------------------------------
code(r"""
# Célula 01 — Ambiente: instala o que falta, importa e fixa a semente
import importlib.util, subprocess, sys
for nome_import, nome_pip in [("rdkit", "rdkit"), ("plotly", "plotly")]:
    if importlib.util.find_spec(nome_import) is None:
        print("instalando", nome_pip, "...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", nome_pip], check=True)

import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs, RDLogger
RDLogger.DisableLog("rdApp.*")          # silencia avisos do RDKit

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import matthews_corrcoef, roc_auc_score, roc_curve

import plotly.graph_objects as go
from plotly.subplots import make_subplots

SEMENTE = 42
np.random.seed(SEMENTE)
print("ambiente pronto")
""")

# ------------------------- A pergunta e os dados ---------------------------
md(r"""
## 1. A pergunta e os dados

**Pergunta:** dada a estrutura de uma molécula, ela é um **inibidor forte** da
AChE? Para responder com ML precisamos de **exemplos rotulados** — moléculas cuja
potência já foi medida em laboratório. É o que o ChEMBL nos dá: milhares de
medidas de atividade contra esse alvo.

Cada linha do arquivo bruto é **uma medida** (uma molécula pode ter várias). As
colunas que vão nos importar: `canonical_smiles` (a estrutura em texto),
`standard_value`/`standard_units` (a potência e sua unidade) e
`standard_relation` (se a medida é exata `=` ou apenas um limite `>`).
""")

code(r"""
# Célula 02 — Carrega os dados brutos do ChEMBL (alvo: AChE)
URL = ("https://raw.githubusercontent.com/monteirotorres/ml/"
       "main/data/dados_alvo_bruto.csv")
try:
    brutos = pd.read_csv(URL)
    print("lidos da URL do repositorio")
except Exception:
    brutos = pd.read_csv("dados_alvo_bruto.csv")   # fallback local
    print("lidos de arquivo local")

print("medidas (linhas):", len(brutos),
      "| moleculas unicas:", brutos["molecule_chembl_id"].nunique())
brutos[["molecule_chembl_id", "canonical_smiles", "standard_relation",
        "standard_value", "standard_units", "pchembl_value"]].head()
""")

# ------------------------------ Curadoria ----------------------------------
md(r"""
## 2. Curadoria: de medidas cruas a moléculas confiáveis

Dado bruto não se treina. Precisamos de um **funil** que descarta o que não é
comparável e resume cada molécula em um único número de potência. Fazemos tudo em
uma célula, guardando a contagem a cada passo — é o funil que aparece no gráfico.

Convertendo tudo para **nM** e usando o **pIC50** $= -\log_{10}(\text{IC}_{50}\,\text{em mol/L})$:
quanto **maior** o pIC50, **mais potente** a molécula. Medidas do tipo "só sei que
é maior que X" (`>`) não viram pIC50, mas não são lixo: dizem que a molécula é
**fraca**, e nós as aproveitamos como tal.
""")

code(r"""
# Célula 03 — Curadoria em um funil (sem funcoes, passo a passo)
funil = [("bruto", len(brutos))]
curados = brutos.copy()

# (a) SMILES presente e legivel pelo RDKit
curados = curados[curados["canonical_smiles"].notna()].copy()
valido = []
for smiles in curados["canonical_smiles"]:
    valido.append(Chem.MolFromSmiles(smiles) is not None)
curados = curados[pd.Series(valido, index=curados.index)].copy()
funil.append(("SMILES valido", len(curados)))

# (b) sem alerta de validade e ensaio de ligacao (tipo B, comparavel)
curados = curados[curados["data_validity_comment"].isna()].copy()
curados = curados[curados["assay_type"] == "B"].copy()
funil.append(("validade + ensaio B", len(curados)))

# (c) converte a unidade para nM e mantem valores positivos
fator = {"nM": 1.0, "uM": 1e3, "µM": 1e3, "mM": 1e6, "M": 1e9, "pM": 1e-3}
valor_nM = []
for v, u in zip(curados["standard_value"], curados["standard_units"]):
    if u in fator:
        try:
            valor_nM.append(float(v) * fator[u])
        except (TypeError, ValueError):
            valor_nM.append(np.nan)            # valor malformado
    else:
        valor_nM.append(np.nan)                # unidade nao-molar (ex. ug/mL)
curados["valor_nM"] = valor_nM
curados = curados[(curados["valor_nM"].notna()) & (curados["valor_nM"] > 0)].copy()
funil.append(("nM, valor > 0", len(curados)))

# (d) descarta relacao '<' (ambigua); marca '>' como "so limite" (sera FRACO)
curados = curados[curados["standard_relation"] != "<"].copy()
curados["apenas_limite"] = curados["standard_relation"].isin([">", ">>"])
funil.append(("descarta '<'", len(curados)))

# (e) pIC50 das medidas EXATAS; medidas '>' nao entram no pIC50
curados["pic50"] = 9.0 - np.log10(curados["valor_nM"])
curados.loc[curados["apenas_limite"], "pic50"] = np.nan

# (f) agrega por molecula: mediana do pIC50 exato (ou "so limite" -> FRACO)
linhas = []
for id_mol, bloco in curados.groupby("molecule_chembl_id"):
    exatas = bloco["pic50"].dropna()
    linhas.append({
        "molecule_chembl_id": id_mol,
        "canonical_smiles": bloco["canonical_smiles"].iloc[0],
        "pic50": np.nan if len(exatas) == 0 else exatas.median(),
        "dispersao": 0.0 if len(exatas) == 0 else (exatas.max() - exatas.min()),
        "sem_medida_exata": len(exatas) == 0,
    })
agregados = pd.DataFrame(linhas)
# descarta moleculas cujas medidas discordam mais de 10x (1 log): pIC50 nao confiavel
agregados = agregados[agregados["dispersao"] <= 1.0].reset_index(drop=True)
funil.append(("moleculas curadas", len(agregados)))

for etapa, n in funil:
    print(f"{etapa:22s} -> {n}")

figura_funil = go.Figure(go.Funnel(
    y=[e for e, _ in funil], x=[n for _, n in funil],
    marker={"color": "#3266ad"}, textinfo="value+percent initial"))
figura_funil.update_layout(title="Funil de curadoria (medidas brutas -> moleculas)",
                           height=340, margin=dict(l=10, r=10, t=40, b=10))
figura_funil.show()
""")

# --------------------------- Descritores + FP ------------------------------
md(r"""
## 3. Como transformar uma molécula em números

Um modelo não lê SMILES; lê **vetores**. Usamos duas representações
complementares:

- **9 descritores** interpretáveis (propriedades físico-químicas): peso molecular
  (MW), lipofilicidade (LogP), área polar (TPSA), doadores/aceptores de ligação de
  hidrogênio, ligações rotáveis, anéis aromáticos, fração sp³ e nº de átomos
  pesados.
- **Fingerprint de Morgan** (2048 bits): cada bit registra a **presença de uma
  subestrutura** ao redor de cada átomo (raio 2). É a "impressão digital" da
  molécula — esparsa e de alta dimensão.

Concatenando os dois, cada molécula vira uma linha da matriz $X$.
""")

code(r"""
# Célula 04 — 9 descritores + fingerprint de Morgan -> matriz X
RAIO_MORGAN, N_BITS = 2, 2048

moleculas = []
for smiles in agregados["canonical_smiles"]:
    moleculas.append(Chem.MolFromSmiles(smiles))
agregados["molecula"] = moleculas

descritores, fingerprints = [], []
for molecula in moleculas:
    descritores.append([
        Descriptors.MolWt(molecula), Descriptors.MolLogP(molecula),
        Descriptors.TPSA(molecula), Descriptors.NumHDonors(molecula),
        Descriptors.NumHAcceptors(molecula), Descriptors.NumRotatableBonds(molecula),
        Descriptors.NumAromaticRings(molecula), Descriptors.FractionCSP3(molecula),
        molecula.GetNumHeavyAtoms(),
    ])
    vetor = np.zeros((N_BITS,), dtype=np.int8)
    fp = AllChem.GetMorganFingerprintAsBitVect(molecula, RAIO_MORGAN, nBits=N_BITS)
    DataStructs.ConvertToNumpyArray(fp, vetor)
    fingerprints.append(vetor)

nomes_descritores = ["MW", "LogP", "TPSA", "HBD", "HBA",
                     "lig_rotaveis", "aneis_arom", "fracao_sp3", "atomos_pesados"]
matriz_descritores = np.array(descritores, dtype=float)
matriz_fingerprint = np.array(fingerprints, dtype=float)
X = np.hstack([matriz_descritores, matriz_fingerprint])
print("X (moleculas x features):", X.shape,
      "= 9 descritores +", N_BITS, "bits")
""")

code(r"""
# Célula 05 — Um fingerprint de Morgan, visto na molecula mais potente
indice_exemplo = agregados["pic50"].idxmax()
molecula_exemplo = agregados.loc[indice_exemplo, "molecula"]
info_bits = {}
AllChem.GetMorganFingerprintAsBitVect(molecula_exemplo, RAIO_MORGAN,
                                      nBits=N_BITS, bitInfo=info_bits)
print("molecula de exemplo:", agregados.loc[indice_exemplo, "molecule_chembl_id"],
      "| pIC50 =", round(agregados.loc[indice_exemplo, "pic50"], 2))
print("bits ligados nesta molecula:", len(info_bits), "de", N_BITS,
      "(o fingerprint e esparso)")
Draw.MolToImage(molecula_exemplo, size=(380, 280))
""")

# ------------------------------ Rotulo + balanco ---------------------------
md(r"""
## 4. O rótulo: FORTE ou FRACO

Transformamos o pIC50 contínuo em duas classes usando um limiar clássico:
**pIC50 ≥ 6** (ou seja, IC₅₀ ≤ 1 µM) é **FORTE**; abaixo disso, **FRACO**. As
moléculas que só têm medida `>` viram FRACO por regra.

Antes de treinar, olhamos o **balanço das classes**. Ele define a **linha de
base**: a acurácia de simplesmente chutar sempre a classe maioritária. Todo
modelo tem que superar esse número para valer a pena.
""")

code(r"""
# Célula 06 — Rotulo FORTE (1) / FRACO (0) e balanco das classes
LIMIAR = 6.0
rotulo = []
for i in range(len(agregados)):
    if agregados.loc[i, "sem_medida_exata"]:
        rotulo.append(0)                                  # so tem '>': FRACO
    elif agregados.loc[i, "pic50"] >= LIMIAR:
        rotulo.append(1)                                  # FORTE
    else:
        rotulo.append(0)                                  # FRACO
y = np.array(rotulo)

n_forte, n_fraco = int((y == 1).sum()), int((y == 0).sum())
base = max(n_forte, n_fraco) / len(y)
print("FORTE:", n_forte, "| FRACO:", n_fraco,
      "| linha de base (chutar a maioria):", round(base, 3))

figura_balanco = go.Figure(go.Bar(
    x=["FRACO (0)", "FORTE (1)"], y=[n_fraco, n_forte],
    marker_color=["#c0392b", "#3266ad"],
    text=[n_fraco, n_forte], textposition="outside"))
figura_balanco.update_layout(title="Balanco das classes",
                             height=330, yaxis_title="nº de moléculas")
figura_balanco.show()
""")

# ------------------------- Particao por esqueleto --------------------------
md(r"""
## 5. A partição que separa quem sabe de quem finge

Aqui está a lição central da aula. A tentação é dividir treino/teste **ao acaso**.
O problema: moléculas parecidas (mesmo **esqueleto** químico) caem dos dois lados,
e o modelo "acerta" o teste porque já viu quase a mesma molécula no treino —
**otimismo enganoso**.

A alternativa honesta é a **partição por esqueleto** (*scaffold split*): cada
núcleo químico fica **inteiro** em um só conjunto. Assim o teste tem moléculas
*estruturalmente novas*, que é o cenário real de quem quer prever algo inédito.
Fazemos as duas divisões para poder **compará-las**.
""")

code(r"""
# Célula 07 — Esqueleto de cada molecula e as duas particoes
esqueletos = []
for molecula in agregados["molecula"]:
    esqueletos.append(MurckoScaffold.MurckoScaffoldSmiles(mol=molecula))
agregados["esqueleto"] = esqueletos

PROP_TREINO, PROP_CALIB = 0.70, 0.15
n_total = len(agregados)

# --- POR ESQUELETO: enche treino, depois teste, mantendo cada nucleo inteiro ---
grupos = {}
for i in range(n_total):
    grupos.setdefault(agregados.loc[i, "esqueleto"], []).append(i)
grupos_ordenados = sorted(grupos.values(), key=lambda g: -len(g))
idx_treino_esq, idx_teste_esq = [], []
for grupo in grupos_ordenados:
    if len(idx_treino_esq) < PROP_TREINO * n_total:
        idx_treino_esq.extend(grupo)
    else:
        idx_teste_esq.extend(grupo)

# --- ALEATORIA: sorteia moleculas ignorando o esqueleto ---
ordem = np.random.RandomState(SEMENTE).permutation(n_total)
corte = int(PROP_TREINO * n_total)
idx_treino_ale, idx_teste_ale = list(ordem[:corte]), list(ordem[corte:])

print("por esqueleto -> treino", len(idx_treino_esq), "| teste", len(idx_teste_esq))
print("aleatoria     -> treino", len(idx_treino_ale), "| teste", len(idx_teste_ale))
""")

code(r"""
# Célula 08 — O espaco quimico das duas divisoes, lado a lado (PCA dos esqueletos)
fps_esqueleto = []
for molecula in agregados["molecula"]:
    mol_esq = MurckoScaffold.GetScaffoldForMol(molecula)
    vetor = np.zeros((N_BITS,), dtype=np.int8)
    if mol_esq is not None and mol_esq.GetNumAtoms() > 0:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol_esq, RAIO_MORGAN, nBits=N_BITS)
        DataStructs.ConvertToNumpyArray(fp, vetor)
    fps_esqueleto.append(vetor)
coords = PCA(n_components=2, random_state=SEMENTE).fit_transform(
    np.array(fps_esqueleto, dtype=float))

conjunto_ale = np.array(["treino"] * n_total, dtype=object)
conjunto_ale[idx_teste_ale] = "teste"
conjunto_esq = np.array(["treino"] * n_total, dtype=object)
conjunto_esq[idx_teste_esq] = "teste"

figura_espaco = make_subplots(rows=1, cols=2, subplot_titles=(
    "Aleatoria (treino e teste misturados)",
    "Por esqueleto (nucleos separados)"))
for col, conjunto in [(1, conjunto_ale), (2, conjunto_esq)]:
    for nome, cor in [("treino", "#3266ad"), ("teste", "#c0392b")]:
        m = conjunto == nome
        figura_espaco.add_trace(go.Scattergl(
            x=coords[m, 0], y=coords[m, 1], mode="markers", name=nome,
            marker=dict(size=5, color=cor, opacity=0.5),
            showlegend=(col == 1)), row=1, col=col)
figura_espaco.update_layout(height=430,
    title="Espaco quimico dos esqueletos (PCA) — a separacao que o olho vê")
figura_espaco.show()
""")

# ------------------------------- Os modelos --------------------------------
md(r"""
## 6. Treinando e comparando modelos

Com a partição **por esqueleto** (a honesta), treinamos três modelos de famílias
diferentes, todos com a mesma interface do scikit-learn:

- **Regressão logística** — linear, rápida, ótima linha de base;
- **Floresta aleatória** — muitas árvores em conjunto, captura interações;
- **Rede neural (MLP)** — pequena, para contrastar.

Avaliamos pelo **MCC** (coeficiente de correlação de Matthews, robusto a classes
desbalanceadas: 0 = chute, 1 = perfeito) e pela **AUC** (área sob a curva ROC).
""")

code(r"""
# Célula 09 — Treina os tres modelos (particao por esqueleto)
X_treino, y_treino = X[idx_treino_esq], y[idx_treino_esq]
X_teste, y_teste = X[idx_teste_esq], y[idx_teste_esq]

modelos = {}
modelos["Regressao logistica"] = Pipeline([
    ("escala", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=SEMENTE))])
modelos["Floresta aleatoria"] = RandomForestClassifier(
    n_estimators=300, n_jobs=-1, random_state=SEMENTE)
modelos["Rede neural (MLP)"] = Pipeline([
    ("escala", StandardScaler()),
    ("clf", MLPClassifier(hidden_layer_sizes=(64,), early_stopping=True,
                          max_iter=300, random_state=SEMENTE))])

for nome, modelo in modelos.items():
    modelo.fit(X_treino, y_treino)
    mcc = matthews_corrcoef(y_teste, modelo.predict(X_teste))
    print(f"{nome:22s} treinado | MCC no teste = {mcc:.3f}")
""")

code(r"""
# Célula 10 — Comparacao: MCC e AUC de cada modelo
resultados = {}
for nome, modelo in modelos.items():
    proba = modelo.predict_proba(X_teste)[:, 1]
    resultados[nome] = {
        "MCC": matthews_corrcoef(y_teste, modelo.predict(X_teste)),
        "AUC": roc_auc_score(y_teste, proba),
        "proba": proba}

tabela = pd.DataFrame({n: {"MCC": r["MCC"], "AUC": r["AUC"]}
                       for n, r in resultados.items()}).T.round(3)
print(tabela.to_string())

figura_comp = go.Figure()
for metrica, cor in [("MCC", "#3266ad"), ("AUC", "#7e9603")]:
    figura_comp.add_trace(go.Bar(x=tabela.index, y=tabela[metrica],
                                 name=metrica, marker_color=cor))
figura_comp.add_hline(y=0.5, line_dash="dash", line_color="#c0392b",
                      annotation_text="AUC base = 0,5")
figura_comp.update_layout(barmode="group", height=360,
                          title="Comparacao de modelos (MCC e AUC)")
figura_comp.show()
""")

code(r"""
# Célula 11 — Curvas ROC sobrepostas
figura_roc = go.Figure()
figura_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                line=dict(dash="dash", color="#999"),
                                name="acaso (AUC 0,5)"))
for nome, cor in zip(resultados, ["#3266ad", "#7e9603", "#c0392b"]):
    fpr, tpr, _ = roc_curve(y_teste, resultados[nome]["proba"])
    figura_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
        name=f"{nome} (AUC {resultados[nome]['AUC']:.3f})",
        line=dict(color=cor, width=2)))
figura_roc.update_layout(height=430, title="Curvas ROC",
    xaxis_title="falsos positivos (1 - especificidade)",
    yaxis_title="verdadeiros positivos (sensibilidade)")
figura_roc.show()
""")

# --------------------------- Interpretabilidade ----------------------------
md(r"""
## 7. O que o modelo está usando?

Um modelo bom que decide pelo motivo errado é uma armadilha. Medimos a
**importância por permutação** sobre a floresta: embaralhamos cada coluna e vemos
**quanto o MCC cai** — se cair muito, aquela informação importava. Comparamos os
9 descritores com o **fingerprint inteiro** (tratado como um bloco), para ver se
a decisão vem da estrutura fina ou de propriedades globais como o tamanho.
""")

code(r"""
# Célula 12 — Importancia por permutacao: descritores x fingerprint (bloco)
floresta = modelos["Floresta aleatoria"]
gerador = np.random.RandomState(SEMENTE)
n_amostra = min(400, len(X_teste))
sel = gerador.choice(len(X_teste), n_amostra, replace=False)
Xp, yp = X_teste[sel].copy(), y_teste[sel]
mcc_ref = matthews_corrcoef(yp, floresta.predict(Xp))

importancias, nomes = [], []
for j in range(len(nomes_descritores)):          # cada descritor
    quedas = []
    for _ in range(3):
        Xe = Xp.copy()
        gerador.shuffle(Xe[:, j])
        quedas.append(mcc_ref - matthews_corrcoef(yp, floresta.predict(Xe)))
    importancias.append(float(np.mean(quedas)))
    nomes.append(nomes_descritores[j])

colunas_fp = list(range(len(nomes_descritores), Xp.shape[1]))   # o fingerprint inteiro
quedas_fp = []
for _ in range(3):
    Xe = Xp.copy()
    Xe[:, colunas_fp] = Xp[np.ix_(gerador.permutation(len(Xp)), colunas_fp)]
    quedas_fp.append(mcc_ref - matthews_corrcoef(yp, floresta.predict(Xe)))
importancias.append(float(np.mean(quedas_fp)))
nomes.append("fingerprint (2048 bits)")

ordem = np.argsort(importancias)
cores = ["#7e9603"] * len(nomes_descritores) + ["#3266ad"]
figura_imp = go.Figure(go.Bar(
    x=[importancias[i] for i in ordem], y=[nomes[i] for i in ordem],
    orientation="h", marker_color=[cores[i] for i in ordem]))
figura_imp.update_layout(height=420, title="Importancia por permutacao",
                         xaxis_title="queda média de MCC ao embaralhar")
figura_imp.show()
""")

# --------------------------- O classificador em uso ------------------------
md(r"""
## 8. O classificador em uso — com abstenção

Por fim, colocamos a floresta para trabalhar em moléculas **novas**. Um modelo
responsável sabe dizer **"não sei"**: se a molécula é muito diferente de tudo que
ele viu (baixa similaridade de **Tanimoto** ao treino — o *domínio de
aplicabilidade*) ou se a probabilidade fica em cima do muro, ele se **abstém**
(INDEFINIDA) em vez de chutar.

Testamos com inibidores conhecidos da AChE (**donepezila, tacrina,
galantamina**) e com moléculas que **não** deveriam dar alerta (aspirina,
cafeína, etanol).
""")

code(r"""
# Célula 13 — Classifica moleculas novas (dominio + abstencao), sem funcoes
LIMIAR_DOMINIO, MARGEM = 0.30, 0.15   # Tanimoto minimo ao treino; margem de abstencao

# fingerprints do treino, para medir a similaridade de novas moleculas
fps_treino = []
for i in idx_treino_esq:
    fps_treino.append(AllChem.GetMorganFingerprintAsBitVect(
        agregados.loc[i, "molecula"], RAIO_MORGAN, nBits=N_BITS))

galeria = [
    ("donepezila",  "O=C1CC2(CCN(Cc3ccccc3)CC2)Cc2cc(OC)c(OC)cc21"),
    ("tacrina",     "Nc1c2c(nc3ccccc13)CCCC2"),
    ("galantamina", "CN1CCC23C=CC(O)CC2Oc2c(OC)ccc(c23)C1"),
    ("aspirina",    "CC(=O)Oc1ccccc1C(=O)O"),
    ("cafeina",     "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
    ("etanol",      "CCO"),
]
mols_galeria, legendas = [], []
for nome, smiles in galeria:
    molecula = Chem.MolFromSmiles(smiles)
    fp = AllChem.GetMorganFingerprintAsBitVect(molecula, RAIO_MORGAN, nBits=N_BITS)
    similaridade = max(DataStructs.BulkTanimotoSimilarity(fp, fps_treino))
    entrada = [
        Descriptors.MolWt(molecula), Descriptors.MolLogP(molecula),
        Descriptors.TPSA(molecula), Descriptors.NumHDonors(molecula),
        Descriptors.NumHAcceptors(molecula), Descriptors.NumRotatableBonds(molecula),
        Descriptors.NumAromaticRings(molecula), Descriptors.FractionCSP3(molecula),
        molecula.GetNumHeavyAtoms(),
    ]
    vetor_fp = np.zeros((N_BITS,), dtype=float)
    DataStructs.ConvertToNumpyArray(fp, vetor_fp)
    prob = float(floresta.predict_proba(
        np.hstack([entrada, vetor_fp]).reshape(1, -1))[0, 1])

    if similaridade < LIMIAR_DOMINIO:
        classe = "INDEFINIDA (fora do dominio)"
    elif abs(prob - 0.5) < MARGEM:
        classe = "INDEFINIDA (ambiguo)"
    else:
        classe = "FORTE" if prob >= 0.5 else "FRACO"
    print(f"{nome:14s} -> {classe:28s} (p_forte={prob:.2f}, Tanimoto={similaridade:.2f})")
    mols_galeria.append(molecula)
    legendas.append(f"{nome}: {classe.split(' ')[0]}")

Draw.MolsToGridImage(mols_galeria, legends=legendas, molsPerRow=3, subImgSize=(240, 190))
""")

# --------------------------------- Fecho -----------------------------------
md(r"""
## 9. Fecho

Em ~1h30 você percorreu o esqueleto de um projeto real de ML aplicado à química:
**curar** os dados, **representar** a molécula em números, **particionar com
honestidade** (por esqueleto, não ao acaso), **comparar** modelos com uma métrica
adequada, **interpretar** o que pesou na decisão e **usar** o classificador
sabendo quando ele deve **se abster**.

**Para aprofundar**, abra a versão completa (`aula.ipynb`): proveniência dos
dados, a mesma rede aberta em **PyTorch** com **TensorBoard**, explicações
**SHAP** por molécula, o **domínio de aplicabilidade** calibrado e uma **triagem
virtual** sobre fármacos aprovados.

**Exercício rápido.** Reexecute a Célula 09 trocando a partição por esqueleto
(`idx_treino_esq`/`idx_teste_esq`) pela aleatória (`idx_treino_ale`/`idx_teste_ale`).
O MCC sobe. Por que esse número maior é **enganoso**?
""")

# ---------------------------------------------------------------------------
notebook = {
    "cells": CELULAS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

destino = os.path.join(os.path.dirname(__file__), "..", "08_aula_bioinfo",
                       "aula_resumida.ipynb")
destino = os.path.abspath(destino)
with open(destino, "w", encoding="utf-8") as arquivo:
    json.dump(notebook, arquivo, ensure_ascii=False, indent=1)

n_code = sum(1 for c in CELULAS if c["cell_type"] == "code")
n_md = sum(1 for c in CELULAS if c["cell_type"] == "markdown")
print("gerado:", destino)
print(f"celulas: {len(CELULAS)} (codigo {n_code} + markdown {n_md})")
