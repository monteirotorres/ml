# -*- coding: utf-8 -*-
"""Gera os notebooks da aula prática de ML em bioinformática.

Fonte única para as duas versões:
  - aula_gabarito.ipynb  — completo (perguntas com resposta; células de modelagem preenchidas)
  - aula_aluno.ipynb     — seções 5-7 de modelagem esvaziadas; respostas removidas

Modelo de célula (lista CELULAS):
  ("md",   texto)                     célula de texto, igual nas duas versões
  ("mdq",  pergunta, resposta)        pergunta dirigida; no aluno some a resposta
  ("code", src)                       código igual nas duas versões
  ("codex", src, instrucao)           código de modelagem; no aluno vira um stub com a instrução

Uso:
    python tools/gerar_aula_notebooks.py
"""

from pathlib import Path
import nbformat as nbf

BASE = Path(__file__).parent.parent
PASTA = BASE / "08_aula_bioinfo"

CELULAS = []
def md(t):            CELULAS.append(("md", t))
def mdq(p, r):        CELULAS.append(("mdq", p, r))
def code(t):          CELULAS.append(("code", t))
def codex(t, instr):  CELULAS.append(("codex", t, instr))


# ══════════════════════════════════════════════════════════════════════════
# TÍTULO
# ══════════════════════════════════════════════════════════════════════════
md("""# Aprendizado de máquina aplicado à bioinformática
## Classificar inibidores de um alvo proteico a partir do ChEMBL

Aula prática. A partir de dados reais de inibidores da **acetilcolinesterase**
(AChE) extraídos do ChEMBL, vamos treinar modelos que recebem qualquer molécula
(pelo seu SMILES) e a classificam como **FORTE** ou **FRACO** para esse alvo — ou
se **abstêm** (classe **INDEFINIDA**) quando não têm base para decidir.

A acetilcolinesterase degrada o neurotransmissor acetilcolina. Seus inibidores
incluem fármacos contra o Alzheimer (donepezila, rivastigmina), inseticidas
organofosforados e agentes neurotóxicos — um alvo de grande relevância
biomédica e toxicológica, com milhares de moléculas medidas no ChEMBL.

**Como usar este notebook.** Cada célula de código vem depois de uma célula de
texto que explica *o que* ela faz, *por que* é necessária e *o que observar* no
resultado. Rode as células em ordem, de cima para baixo. O notebook inteiro roda
em CPU, sem GPU, em poucos minutos.

**A decisão de projeto central: treino binário, resposta com abstenção.** O modelo
é treinado só para separar FORTE de FRACO (o rótulo, por um limiar de pIC50). Mas,
na hora de responder, ele pode **se abster** e devolver a classe **INDEFINIDA** por
duas razões distintas: (a) **abstenção por ambiguidade** — a probabilidade fica
perto do meio, sem evidência clara; (b) **fora do domínio de aplicabilidade** — a
molécula não se parece com nada que o modelo viu, então ele não deveria opinar.
Saber quando não decidir é parte do trabalho.""")

# ══════════════════════════════════════════════════════════════════════════
# SEÇÃO 0 — AMBIENTE
# ══════════════════════════════════════════════════════════════════════════
md("""## Seção 0 — Ambiente

Antes de qualquer ciência, deixamos o ambiente reprodutível: instalamos o que
falta, importamos as bibliotecas agrupadas por finalidade, imprimimos as versões
e fixamos **uma única semente** propagada a tudo que é aleatório. Reprodutível
quer dizer que rodar de novo dá o mesmo resultado — condição para poder confiar
em qualquer número que aparecer depois.""")

md("""### 0.1 — Instalação

`scikit-learn`, `pandas`, `numpy`, `matplotlib` e `torch` já vêm no Colab. Faltam
o `rdkit` (química), o `chembl_webresource_client` (acesso ao ChEMBL) e o
`plotly` (gráficos interativos). Instalamos apenas o ausente, testando o
`import` antes — assim a célula é rápida quando já está tudo instalado.

Este bloco usa `subprocess` em vez da mágica `!pip` para funcionar igual no Colab
e em qualquer outro kernel. Pode levar cerca de um minuto na primeira execução.""")
code(r'''import importlib.util
import subprocess
import sys

# pares (nome para importar, nome para instalar via pip)
pacotes = [
    ("rdkit", "rdkit"),
    ("plotly", "plotly"),
    ("chembl_webresource_client", "chembl_webresource_client"),
    ("shap", "shap"),
    ("umap", "umap-learn"),
    ("tqdm", "tqdm"),
]

# instala cada um so se ainda nao puder ser importado
for nome_import, nome_pip in pacotes:
    if importlib.util.find_spec(nome_import) is not None:
        print("ja instalado:", nome_import)
    else:
        print("instalando:", nome_pip)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", nome_pip], check=True)
print("instalacao concluida")''')

md("""### 0.2 — Importações, agrupadas por finalidade

Importar tudo em um lugar só, comentado, evita surpresas mais adiante e mostra ao
leitor o mapa de ferramentas da aula.""")
code(r'''# --- básicas: dados e números ---
import numpy as np
import pandas as pd

# --- química: ler moléculas, calcular descritores e fingerprints, desenhar ---
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")   # silencia avisos verbosos do RDKit

# --- modelos e avaliação (tudo scikit-learn: uma única interface) ---
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
from sklearn.metrics import (matthews_corrcoef, roc_auc_score, roc_curve,
                             precision_recall_curve, confusion_matrix,
                             classification_report)

# --- rede neural "aberta" em PyTorch (só na subseção 5b) ---
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# --- gráficos interativos ---
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- desenho estático e utilidades ---
import matplotlib.pyplot as plt
import time
import joblib
from tqdm.auto import tqdm

print("importacoes ok")''')

md("""### 0.3 — Versões e semente

Imprimimos as versões (inclusive a do PyTorch, verificada em tempo de execução, e
não suposta) e confirmamos que **não há GPU**: a aula roda em CPU por escolha,
para ser idêntica em qualquer máquina. A semente é uma constante única no topo,
propagada a `random`, `numpy` e `torch`, e passada como `random_state` a todo
estimador daqui para frente.""")
code(r'''import random
import sklearn

SEMENTE = 42
random.seed(SEMENTE)
np.random.seed(SEMENTE)
torch.manual_seed(SEMENTE)

print("numpy       ", np.__version__)
print("pandas      ", pd.__version__)
print("scikit-learn", sklearn.__version__)
print("torch       ", torch.__version__)
print("cuda disponivel:", torch.cuda.is_available(), "(esperado: False, rodamos em CPU)")
DISPOSITIVO = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("dispositivo :", DISPOSITIVO)''')


# ══════════════════════════════════════════════════════════════════════════
# SEÇÃO 1 — A PERGUNTA E OS DADOS
# ══════════════════════════════════════════════════════════════════════════
md("""## Seção 1 — A pergunta e os dados

Antes de qualquer código: **o que estamos medindo?**

Um inibidor é uma molécula que reduz a atividade da enzima. Quanto menos inibidor
é preciso para reduzir a atividade pela metade, mais potente ele é. Essa
quantidade é o **IC50** — a concentração que inibe 50% da atividade. IC50 baixo =
inibidor potente.

Como o IC50 varia por ordens de grandeza (de nanomolar a milimolar), trabalhamos
com o **pIC50**:

$$\\text{pIC50} = -\\log_{10}(\\text{IC50 em mol/L})$$

A escala logarítmica transforma "10 vezes mais potente" em "+1 unidade",
tornando a variável comparável e bem-comportada para modelos. pIC50 **alto** =
inibidor **potente**.

**Uma ressalva honesta:** IC50 não é energia de ligação. Ele depende da
concentração de substrato, do $K_m$ da enzima e do formato do ensaio (relação de
Cheng-Prusoff). Medidas de laboratórios e ensaios diferentes têm ruído — motivo
pelo qual a curadoria da Seção 2 existe.""")

md("""### 1.1 — Carregar os dados

Nós já extraímos os dados do ChEMBL e deixamos o CSV num **link público** no
repositório. Assim o notebook lê direto da internet — **sem precisar subir
arquivo nenhum** no Colab. É o caminho padrão, e é rápido.""")
code(r'''URL_DADOS = ("https://raw.githubusercontent.com/monteirotorres/ml/"
             "main/data/dados_alvo_bruto.csv")

# le o CSV direto da URL; se a internet falhar, tenta um arquivo local de mesmo nome
try:
    dados_brutos = pd.read_csv(URL_DADOS)
    print("lidos do repositorio (URL):", len(dados_brutos), "registros")
except Exception as erro:
    print("URL indisponivel (", type(erro).__name__, "); tentando arquivo local")
    dados_brutos = pd.read_csv("dados_alvo_bruto.csv")
    print("lidos do arquivo local:", len(dados_brutos), "registros")

dados_brutos.head()''')

md("""**Opcional — como o CSV foi produzido.** A célula abaixo re-extrai os dados
direto da API do ChEMBL, para mostrar de onde eles vêm. É **lenta** (alguns
minutos, baixa ~10 mil medidas uma a uma), então vem **desligada** — deixe
`EXECUTAR_EXTRACAO_API = False` em sala e rode em casa se tiver curiosidade. O
alvo é resolvido pelo **nome do gene**, nunca fixado no código: trocar de alvo é
mudar uma linha.""")
code(r'''ALVO_GENE = "ACHE"                    # gene da acetilcolinesterase humana
EXECUTAR_EXTRACAO_API = False         # mude para True para baixar da API (lento)

if EXECUTAR_EXTRACAO_API:
    from chembl_webresource_client.new_client import new_client

    # 1. resolve o ChEMBL ID do alvo a partir do simbolo do gene
    alvo_id = None
    for alvo in new_client.target.filter(target_synonym__icontains=ALVO_GENE):
        if alvo.get("organism") == "Homo sapiens":
            alvo_id = alvo["target_chembl_id"]
            break
    print("alvo resolvido pela API:", alvo_id)

    # 2. baixa as atividades IC50 desse alvo, uma a uma, para uma tabela
    colunas = ["molecule_chembl_id", "canonical_smiles", "standard_type",
               "standard_relation", "standard_value", "standard_units",
               "pchembl_value", "assay_type", "assay_chembl_id",
               "data_validity_comment", "document_chembl_id", "target_chembl_id"]
    linhas = []
    for atividade in new_client.activity.filter(target_chembl_id=alvo_id,
                                                standard_type="IC50"):
        linha = {}
        for coluna in colunas:
            linha[coluna] = atividade.get(coluna)
        linhas.append(linha)
    dados_brutos = pd.DataFrame(linhas)
    print("re-extraidos da API:", len(dados_brutos), "registros")
else:
    print("extracao da API desligada; usando o CSV lido acima")''')

md("""### 1.2 — Primeiro olhar

Antes de mexer em qualquer coisa, olhamos o formato bruto: quantas linhas,
quantas moléculas únicas, e a distribuição das relações de medida (`=`, `>`,
`<`). As relações vão importar muito na curadoria — uma medida "> 10000 nM"
significa "pelo menos tão fraco quanto isso", e essa informação não pode ser
jogada fora.""")
code(r'''print("linhas totais      :", len(dados_brutos))
print("moleculas unicas   :", dados_brutos["molecule_chembl_id"].nunique())
print("com SMILES         :", dados_brutos["canonical_smiles"].notna().sum())
print()
print("relacoes de medida (standard_relation):")
print(dados_brutos["standard_relation"].value_counts(dropna=False))''')


# ══════════════════════════════════════════════════════════════════════════
# SEÇÃO 2 — CURADORIA
# ══════════════════════════════════════════════════════════════════════════
md("""## Seção 2 — Curadoria, com tabela de proveniência

Dados de repositório público chegam sujos. Cada filtro abaixo corrige um defeito
específico, e **cada etapa imprime quantas linhas entraram e quantas saíram** —
nenhuma transformação silenciosa. Ao final, montamos uma tabela de proveniência
(etapa, n restante, n removido, motivo) e um gráfico de funil.

Começamos criando um registro da proveniência e uma cópia de trabalho.""")
code(r'''proveniencia = []   # lista de (etapa, n_restante, n_removido, motivo)

def registrar_etapa(nome, antes, depois, motivo):
    """Imprime o efeito de uma etapa e o guarda para a tabela de proveniencia."""
    removido = antes - depois
    proveniencia.append((nome, depois, removido, motivo))
    print(f"{nome:32s} | antes {antes:5d} -> depois {depois:5d} | removidos {removido:5d}")

curados = dados_brutos.copy()
registrar_etapa("bruto", len(curados), len(curados), "dados como vieram do ChEMBL")''')

md("""### 2.1 — SMILES ausentes ou inválidos

Um SMILES é a representação textual da molécula. Se estiver faltando ou não puder
ser lido pelo RDKit (parênteses errados, valência impossível), a linha é inútil —
não há molécula para calcular nada. Testamos cada SMILES com um laço explícito.""")
code(r'''antes = len(curados)
curados = curados[curados["canonical_smiles"].notna()].copy()

smiles_valido = []
for smiles in curados["canonical_smiles"]:
    molecula = Chem.MolFromSmiles(smiles)
    smiles_valido.append(molecula is not None)
curados = curados[pd.Series(smiles_valido, index=curados.index)].copy()

registrar_etapa("SMILES valido", antes, len(curados), "SMILES ausente ou ilegivel pelo RDKit")''')

md("""### 2.2 — Comentários de validade e tipo de ensaio

O ChEMBL marca medidas suspeitas em `data_validity_comment` ("Outside typical
range", "Potential transcription error"). Removemos as marcadas. Também mantemos
apenas ensaios de ligação/funcionais do tipo `B` (binding), que medem inibição
direta, descartando os demais formatos que não são comparáveis.""")
code(r'''antes = len(curados)
curados = curados[curados["data_validity_comment"].isna()].copy()
registrar_etapa("validade ok", antes, len(curados), "data_validity_comment sinalizado")

antes = len(curados)
curados = curados[curados["assay_type"] == "B"].copy()
registrar_etapa("ensaio tipo B", antes, len(curados), "ensaio nao-binding (nao comparavel)")''')

md("""### 2.3 — Unidades e valores numéricos

Queremos tudo em **nanomolar (nM)** para que o pIC50 faça sentido. Aqui vale um
detalhe do ChEMBL: usamos o campo `standard_units`, que é a unidade **já
padronizada** pela curadoria do ChEMBL. Ou seja, os µM e pM que apareciam nos
artigos originais **já foram convertidos para nM** rio acima — por isso a coluna é
quase toda "nM". Vamos primeiro **olhar** a distribuição de unidades antes de
filtrar (nunca uma etapa silenciosa).""")
code(r'''print("distribuicao de unidades (standard_units):")
print(curados["standard_units"].value_counts(dropna=False))''')

md("""Mesmo assim, para o código ficar **correto em qualquer alvo** (nem todo
conjunto vem tão limpo), convertemos explicitamente as unidades molares
recuperáveis para nM — µM, mM, M, pM — em vez de descartá-las. Imprimimos quantas
linhas foram convertidas. O que sobra e não é molar (por exemplo `ug.mL-1`,
concentração em massa, que exigiria a massa molecular) ou está malformado será
descartado no filtro seguinte.""")
code(r'''# fatores para converter cada unidade molar para nM
fator_para_nM = {
    "nM": 1.0, "uM": 1000.0, "um": 1000.0, "µM": 1000.0,
    "mM": 1e6, "M": 1e9, "pM": 0.001, "fM": 1e-6,
}
antes = len(curados)
convertidas = 0
valores_nM = []
unidades_nM = []
for valor, unidade in zip(curados["standard_value"], curados["standard_units"]):
    if unidade in fator_para_nM:
        try:
            valores_nM.append(float(valor) * fator_para_nM[unidade])
            unidades_nM.append("nM")
            if unidade != "nM":
                convertidas = convertidas + 1
        except (TypeError, ValueError):
            valores_nM.append(np.nan); unidades_nM.append(unidade)
    else:
        valores_nM.append(np.nan); unidades_nM.append(unidade)   # nao-molar: sera descartado
curados["standard_value"] = valores_nM
curados["standard_units"] = unidades_nM
print("linhas convertidas de outra unidade molar para nM:", convertidas)

# agora filtra o que sobrou: manter nM com valor numerico positivo
curados = curados[curados["standard_units"] == "nM"].copy()
curados = curados[curados["standard_value"].notna()].copy()
curados = curados[curados["standard_value"] > 0].copy()
registrar_etapa("nM, valor > 0", antes, len(curados),
                "unidade nao-molar (ex. ug/mL), malformada, ou valor ausente/<=0")''')

md("""### 2.4 — Medidas de limite (`>`): por que manter os "maior que"

Uma medida com relação `>` ("IC50 > 30000 nM") não é um valor pontual: ela dá só
um **limite** — "não conseguimos inibir nem nessa concentração alta" —, ou seja,
um inibidor **fraco de verdade**. Em um problema de **regressão** essas medidas
seriam descartadas por não terem valor exato. Aqui, num problema de
**classificação**, elas são exatamente a classe FRACO — a classe que o viés de
publicação torna escassa, porque poucos artigos relatam moléculas que não
funcionam.

Regra: **manter as relações `>`** e rotulá-las FRACO; **descartar as `<`** (uma
molécula "melhor que" um limite baixo é ambígua para a classe). As relações `=`
seguem normalmente.""")
code(r'''antes = len(curados)
curados = curados[curados["standard_relation"] != "<"].copy()
registrar_etapa("descarta relacao <", antes, len(curados), "relacao '<' (limite inferior, ambiguo)")

# marca quais linhas dao apenas um limite '>' (serao forcadas a FRACO adiante)
eh_apenas_limite = []
for relacao in curados["standard_relation"]:
    eh_apenas_limite.append(relacao in (">", ">>"))
curados["apenas_limite"] = eh_apenas_limite
print("medidas '>' mantidas (viram FRACO nas moleculas que so tem elas):",
      int(curados["apenas_limite"].sum()))''')

md("""### 2.5 — pIC50 e agregação de duplicatas

O ChEMBL **já fornece o pIC50 pronto**, na coluna `pchembl_value` — é
$-\\log_{10}$ da potência em mol/L, curado por eles. Usamos esse valor direto (para
medidas exatas o `pchembl_value` e o cálculo à mão $9 - \\log_{10}(\\text{nM})$
coincidem, o que serve de verificação).

**E as medidas de limite `>`?** Elas **não recebem pIC50**. Uma medida
"IC50 > 30000 nM" não é um valor pontual — é só um **limite** ("no mínimo tão
fraca quanto isto"). Fabricar um número a partir dela seria inventar precisão que
não existe, e esse número nem é usado: uma molécula que só tem medidas `>` é
rotulada **FRACO por regra** (é um inativo conhecido). O pIC50 só faz falta para
molécula que tem medida **exata** — e essas têm.

Consequência importante: o rótulo de uma molécula vem das suas medidas **exatas**,
quando existem. Uma molécula com um IC50 exato potente **não** vira FRACO só
porque um outro ensaio reportou `>` — as medidas exatas decidem; as de limite só
confirmam a fraqueza de quem não tem nenhuma medida exata.

Agregamos por molécula pela **mediana** das medidas exatas (robusta a outliers) e
**descartamos moléculas cuja dispersão passa de uma unidade log** (exatas que
discordam por mais de 10x não são confiáveis).""")
code(r'''# pIC50 apenas das medidas EXATAS: usa o pchembl_value curado; onde faltar
# (raro, medidas exatas sem pchembl) calcula 9 - log10(nM).
pchembl = pd.to_numeric(curados["pchembl_value"], errors="coerce")
pic50_calculado = 9.0 - np.log10(curados["standard_value"])   # = -log10(mol/L)
curados["pic50"] = pchembl.where(pchembl.notna(), pic50_calculado)
# medidas '>' dao apenas um limite: nao entram no pIC50
curados.loc[curados["apenas_limite"], "pic50"] = np.nan

n_exatas = int(curados["pic50"].notna().sum())
print("medidas com pIC50 (exatas):", n_exatas,
      "| de limite (so '>', sem pIC50):", len(curados) - n_exatas)
mascara_ambos = pchembl.notna() & curados["pic50"].notna()
print("diferenca media |pchembl - calculado| (medidas exatas):",
      round((pchembl[mascara_ambos] - pic50_calculado[mascara_ambos]).abs().mean(), 3),
      "(esperado ~0)")

# agrega por molecula usando SO as medidas exatas
grupos = curados.groupby("molecule_chembl_id")
linhas_agregadas = []
for id_molecula, bloco in grupos:
    pic50_exatas = bloco["pic50"].dropna()          # descarta os NaN das medidas de limite
    sem_medida_exata = len(pic50_exatas) == 0       # molecula so tem medidas '>'
    if sem_medida_exata:
        pic50_mediana = np.nan
        dispersao = 0.0                             # sem exatas para discordar
    else:
        pic50_mediana = pic50_exatas.median()
        dispersao = pic50_exatas.max() - pic50_exatas.min()
    smiles = bloco["canonical_smiles"].iloc[0]
    linhas_agregadas.append({
        "molecule_chembl_id": id_molecula,
        "canonical_smiles": smiles,
        "pic50": pic50_mediana,
        "dispersao_log": dispersao,
        "sem_medida_exata": sem_medida_exata,
        "n_medidas": len(bloco),
    })
agregados = pd.DataFrame(linhas_agregadas)
print("moleculas unicas apos agregacao:", len(agregados))
print("  destas, so com medida '>' (serao FRACO por regra):",
      int(agregados["sem_medida_exata"].sum()))

antes = len(agregados)
agregados = agregados[agregados["dispersao_log"] <= 1.0].copy()
registrar_etapa("dispersao <= 1 log", antes, len(agregados),
                "medidas discordam mais de 10x (pIC50 nao confiavel)")''')

md("""### 2.6 — Tabela de proveniência e funil

Agora a prestação de contas: a tabela cumulativa de tudo que fizemos, e o mesmo
em um gráfico de funil interativo. Nada saiu dos dados sem motivo registrado.""")
code(r'''tabela_proveniencia = pd.DataFrame(
    proveniencia, columns=["etapa", "n_restante", "n_removido", "motivo"])
print(tabela_proveniencia.to_string(index=False))

figura_funil = go.Figure(go.Funnel(
    y=tabela_proveniencia["etapa"],
    x=tabela_proveniencia["n_restante"],
    textinfo="value+percent initial"))
figura_funil.update_layout(title="Funil de curadoria dos dados", height=420)
figura_funil.show()''')

mdq("""**Pergunta.** Por que foi importante manter as medidas com relação `>` em
vez de descartá-las, como se faria numa regressão?""",
"""**Resposta.** Porque elas são inativos verdadeiros — a classe FRACO. O viés de
publicação faz com que moléculas que não funcionam raramente sejam publicadas com
valor exato; descartá-las esvaziaria justamente a classe minoritária e o modelo
aprenderia um mundo onde quase tudo é potente, degradando a detecção de FRACO.""")


# ══════════════════════════════════════════════════════════════════════════
# SEÇÃO 3 — DESCRITORES
# ══════════════════════════════════════════════════════════════════════════
md("""## Seção 3 — Os descritores, explicados um a um

Um modelo não "vê" moléculas: vê números. Precisamos transformar cada molécula em
um vetor de números — os **descritores**. Usaremos dois tipos: descritores
físico-químicos interpretáveis e o fingerprint de Morgan (estrutural).

### Descritores físico-químicos

| Descritor | O que mede | Por que importa | Faixa típica |
| --- | --- | --- | --- |
| MW | massa molecular | tamanho; ligado à permeabilidade e à potência bruta | 150–500 |
| LogP | lipofilicidade | equilíbrio água/gordura; permeação de membrana | -1 a 5 |
| TPSA | área polar topológica | ligada à absorção e à passagem pela barreira hematoencefálica | 20–140 |
| HBD | doadores de ligação de H | interações com o alvo; permeabilidade | 0–5 |
| HBA | aceptores de ligação de H | interações com o alvo; solubilidade | 0–10 |
| Ligações rotáveis | flexibilidade | rigidez favorece afinidade e biodisponibilidade | 0–10 |
| Anéis aromáticos | número de anéis aromáticos | empilhamento com resíduos aromáticos do sítio | 0–4 |
| Fração sp3 | carbonos sp3 / carbonos | "tridimensionalidade"; ligada a sucesso clínico | 0–1 |
| Átomos pesados | átomos não-hidrogênio | tamanho bruto; volta a importar na Seção 6 | 10–40 |

A regra dos cinco de Lipinski (MW≤500, LogP≤5, HBD≤5, HBA≤10) resume boa parte
disso para fármacos orais.""")
code(r'''# converte cada SMILES em objeto molecula uma unica vez (reaproveitado adiante)
moleculas = []
for smiles in agregados["canonical_smiles"]:
    moleculas.append(Chem.MolFromSmiles(smiles))
agregados["molecula"] = moleculas

# calcula os nove descritores de cada molecula, com um laco explicito
lista_descritores = []
for molecula in agregados["molecula"]:
    descritores = {}
    descritores["MW"] = Descriptors.MolWt(molecula)
    descritores["LogP"] = Descriptors.MolLogP(molecula)
    descritores["TPSA"] = Descriptors.TPSA(molecula)
    descritores["HBD"] = Descriptors.NumHDonors(molecula)
    descritores["HBA"] = Descriptors.NumHAcceptors(molecula)
    descritores["ligacoes_rotaveis"] = Descriptors.NumRotatableBonds(molecula)
    descritores["aneis_aromaticos"] = Descriptors.NumAromaticRings(molecula)
    descritores["fracao_sp3"] = Descriptors.FractionCSP3(molecula)
    descritores["atomos_pesados"] = molecula.GetNumHeavyAtoms()
    lista_descritores.append(descritores)
tabela_descritores = pd.DataFrame(lista_descritores, index=agregados.index)
print("dimensao da tabela de descritores:", tabela_descritores.shape)
tabela_descritores.describe().round(2)''')

md("""### 3.1 — O fingerprint de Morgan, visualmente

Descritores físico-químicos resumem a molécula em poucos números, mas não dizem
*quais subestruturas* ela tem. O **fingerprint de Morgan** faz isso: percorre
cada átomo, olha a vizinhança em círculos de raio crescente, e liga um bit para
cada subestrutura encontrada. O resultado é um vetor binário longo (aqui 2048
bits): 1 se a subestrutura está presente, 0 se não.

Cada bit é, portanto, "esta molécula contém esta pequena subestrutura?". Vamos
ver isso com os olhos: pegamos uma molécula e desenhamos duas subestruturas
(bits) que ela ativa.""")
code(r'''RAIO_MORGAN = 2
N_BITS = 2048

# escolhe uma molecula relativamente potente para ilustrar
indice_exemplo = agregados["pic50"].idxmax()
molecula_exemplo = agregados.loc[indice_exemplo, "molecula"]

# calcula o fingerprint guardando quais atomos ativaram cada bit (bitInfo)
info_bits = {}
fp_exemplo = AllChem.GetMorganFingerprintAsBitVect(
    molecula_exemplo, RAIO_MORGAN, nBits=N_BITS, bitInfo=info_bits)
bits_ativos = list(info_bits.keys())
print("molecula de exemplo:", agregados.loc[indice_exemplo, "molecule_chembl_id"])
print("bits ativados nesta molecula:", len(bits_ativos), "de", N_BITS)

# desenha a molecula
Draw.MolToImage(molecula_exemplo, size=(360, 260))''')

md("""Agora dois ou três bits específicos: cada painel mostra o átomo central
(destacado) e a vizinhança que define aquela subestrutura. É isso que um "1" no
fingerprint significa.""")
code(r'''bits_para_mostrar = bits_ativos[:3]
tuplas_bits = []
for bit in bits_para_mostrar:
    tuplas_bits.append((molecula_exemplo, bit, info_bits))

imagem_bits = Draw.DrawMorganBits(
    tuplas_bits, molsPerRow=3,
    legends=["bit " + str(bit) for bit in bits_para_mostrar])
imagem_bits''')

md("""**Raio, número de bits e colisão.** O raio (2) controla o tamanho das
vizinhanças consideradas; mais bits (2048) reduzem, mas não eliminam, a
**colisão** — duas subestruturas diferentes podem cair no mesmo bit, porque o
número de subestruturas possíveis é maior que o número de bits. É uma perda de
informação que aceitamos em troca de um vetor de tamanho fixo.""")

md("""### 3.2 — A tabela onde o treino realmente acontece

Aqui está o ponto que desmistifica "treinar um modelo": juntamos os descritores,
os primeiros bits do fingerprint e o rótulo em **uma tabela numérica**. Treinar é
ajustar uma função que, dessa tabela de entrada, prevê a coluna de rótulo. Não há
mágica — há uma tabela de números.

O rótulo binário: uma molécula que só tem medida `>` é **FRACO** por regra (é um
inativo conhecido, sem pIC50). As demais são **FORTE** se o pIC50 ≥
`LIMIAR_POTENCIA`, senão **FRACO** — decididas pela sua medida exata, como na
Seção 2.5.""")
code(r'''LIMIAR_POTENCIA = 6.0   # pIC50 >= 6 equivale a IC50 <= 1 uM (1000 nM)

# calcula o fingerprint de Morgan de cada molecula e o converte num vetor 0/1
lista_fingerprints = []
for molecula in agregados["molecula"]:
    fp = AllChem.GetMorganFingerprintAsBitVect(molecula, RAIO_MORGAN, nBits=N_BITS)
    vetor = np.zeros((N_BITS,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, vetor)
    lista_fingerprints.append(vetor)
matriz_fingerprint = np.array(lista_fingerprints)

# rotulo: 1 = FORTE, 0 = FRACO
rotulo = []
for indice in agregados.index:
    if agregados.loc[indice, "sem_medida_exata"]:
        rotulo.append(0)                       # so tem medida '>': FRACO por regra
    elif agregados.loc[indice, "pic50"] >= LIMIAR_POTENCIA:
        rotulo.append(1)                       # FORTE (decidido pela medida exata)
    else:
        rotulo.append(0)                       # FRACO
agregados["rotulo"] = rotulo

# tabela de treino = descritores + fingerprint + rotulo
tabela_treino = tabela_descritores.copy()
for indice_bit in range(8):
    tabela_treino["fp_" + str(indice_bit)] = matriz_fingerprint[:, indice_bit]
tabela_treino["rotulo"] = agregados["rotulo"].values
print("dimensao da tabela de treino (com fingerprint truncado):", tabela_treino.shape)
tabela_treino.head()''')

md("""### 3.3 — Dimensões e balanço de classes

Imprimimos o tamanho da matriz completa, a densidade do fingerprint (fração de
bits ligados — fingerprints são esparsos) e quantas moléculas há em cada classe.
O balanço de classes vai definir a linha de base contra a qual julgamos todo
modelo.""")
code(r'''# matriz de entrada completa (X) = descritores + fingerprint inteiro
matriz_descritores = tabela_descritores.values.astype(float)
X = np.hstack([matriz_descritores, matriz_fingerprint.astype(float)])
y = agregados["rotulo"].values

nomes_features = list(tabela_descritores.columns)
for indice_bit in range(N_BITS):
    nomes_features.append("fp_" + str(indice_bit))

print("X (moleculas x features):", X.shape)
print("densidade do fingerprint :", round(matriz_fingerprint.mean(), 4),
      "(fracao de bits ligados)")
n_forte = int((y == 1).sum())
n_fraco = int((y == 0).sum())
print("FORTE:", n_forte, "| FRACO:", n_fraco)
maior_classe = max(n_forte, n_fraco)
print("linha de base (chutar a classe maioritaria):",
      round(maior_classe / len(y), 3), "de acuracia")''')

md("""O mesmo balanço, em um gráfico — para **ver** o tamanho de cada classe. A
barra maior é a linha de base: um modelo que chutasse sempre a classe maior já
acertaria essa fração. Só faz sentido celebrar um modelo que **supere** essa
barra.""")
code(r'''# tamanho de cada classe, com o numero e a fracao anotados na barra
contagem_classes = [n_fraco, n_forte]
nomes_classes = ["FRACO (0)", "FORTE (1)"]
cores_classes = ["#c0392b", "#3266ad"]
total_moleculas = len(y)

figura_classes, eixo_classes = plt.subplots(figsize=(5, 3.2))
barras = eixo_classes.bar(nomes_classes, contagem_classes, color=cores_classes)
for barra, contagem in zip(barras, contagem_classes):
    fracao = contagem / total_moleculas
    altura = barra.get_height()
    rotulo_barra = str(contagem) + "\n(" + str(round(100 * fracao, 1)) + "%)"
    eixo_classes.text(barra.get_x() + barra.get_width() / 2, altura,
                      rotulo_barra, ha="center", va="bottom", fontsize=9)
eixo_classes.set_ylabel("numero de moleculas")
eixo_classes.set_title("Tamanho de cada classe (n = " + str(total_moleculas) + ")")
eixo_classes.set_ylim(0, max(contagem_classes) * 1.18)
plt.tight_layout()
plt.show()''')

# ══════════════════════════════════════════════════════════════════════════
# SEÇÃO 4 — PARTIÇÃO
# ══════════════════════════════════════════════════════════════════════════
md("""## Seção 4 — Partição, com representação gráfica

Para estimar honestamente o desempenho, separamos os dados em três papéis:
**treino** (ajusta o modelo), **calibração** e **teste** (tocado só no fim).

Um esclarecimento que confunde muita gente: **não há aqui um conjunto de
"validação" no sentido usual.** Um conjunto de validação separado serve para
**escolher hiperparâmetros** ou comparar modelos durante o desenvolvimento —
e nesta aula não fazemos busca de hiperparâmetros. Quando precisamos comparar de
forma robusta, usamos **validação cruzada** (o segundo diagrama abaixo), que
valida *sem* reservar um bloco fixo. Duas ressalvas: a rede neural cria
internamente uma fração de validação **a partir do próprio treino** para decidir
quando parar (early stopping) — não é um quarto bloco; e o terceiro bloco aqui, a
**calibração**, tem um papel específico e diferente — a predição conformal da
Seção 8 precisa de dados não usados nem no treino nem no teste. Se você pular a
Seção 8, pode devolver a calibração ao treino.

*Como* separamos importa mais do que o tamanho de cada parte.

A regra, em uma frase: **moléculas de mesmo esqueleto devem ficar do mesmo lado**
(todas no treino ou todas no teste), **nunca separadas** entre treino e teste.

- **Divisão aleatória**: sorteia moléculas uma a uma. O problema é que ela
  **separa** os análogos — séries químicas têm muitos quase-idênticos, e sortear
  faz um cair no treino e o quase-gêmeo no teste. O modelo então "acerta" no teste
  porque já viu um primo no treino, sem ter aprendido nada generalizável — e o
  desempenho **infla**. Essa separação é, na prática, um vazamento.
- **Divisão por esqueleto (Bemis-Murcko)**: agrupa as moléculas pelo esqueleto
  (o núcleo, sem as cadeias laterais) e mantém **cada esqueleto inteiro em um único
  conjunto** — os análogos nunca se separam. Assim o teste contém núcleos que o
  treino **nunca viu**, que é a situação real de uso. É mais dura e mais honesta.""")
code(r'''# extrai o esqueleto de Bemis-Murcko (o nucleo, sem cadeias laterais) de cada molecula
esqueletos = []
for molecula in agregados["molecula"]:
    esqueleto = MurckoScaffold.GetScaffoldForMol(molecula)
    esqueletos.append(Chem.MolToSmiles(esqueleto))
agregados["esqueleto"] = esqueletos
print("moleculas:", len(agregados), "| esqueletos distintos:", agregados["esqueleto"].nunique())''')

md("""### 4.1 — As duas divisões

Implementamos as duas com laços explícitos. Na divisão por esqueleto, ordenamos os
grupos do maior para o menor e vamos preenchendo treino, calibração e teste até
atingir as proporções — nenhum esqueleto se divide entre conjuntos.""")
code(r'''PROP_TREINO, PROP_CALIB = 0.70, 0.15   # o resto (0.15) vai para teste
n_total = len(agregados)

# --- divisao POR ESQUELETO: cada esqueleto inteiro fica em um unico conjunto ---
# 1. agrupa os indices das moleculas por esqueleto
grupos_por_esqueleto = {}
for indice in agregados.index:
    chave = agregados.loc[indice, "esqueleto"]
    if chave not in grupos_por_esqueleto:
        grupos_por_esqueleto[chave] = []
    grupos_por_esqueleto[chave].append(indice)

# 2. do maior grupo para o menor, enche treino, depois calibracao, depois teste
grupos_ordenados = sorted(grupos_por_esqueleto.values(),
                          key=lambda grupo: (-len(grupo), agregados.loc[grupo[0], "esqueleto"]))
idx_treino_esq, idx_calib_esq, idx_teste_esq = [], [], []
for grupo in grupos_ordenados:
    if len(idx_treino_esq) < PROP_TREINO * n_total:
        idx_treino_esq.extend(grupo)
    elif len(idx_calib_esq) < PROP_CALIB * n_total:
        idx_calib_esq.extend(grupo)
    else:
        idx_teste_esq.extend(grupo)

# --- divisao ALEATORIA: sorteia moleculas, ignorando o esqueleto ---
indices_embaralhados = list(agregados.index)
np.random.RandomState(SEMENTE).shuffle(indices_embaralhados)
corte_treino = int(PROP_TREINO * n_total)
corte_calib = int((PROP_TREINO + PROP_CALIB) * n_total)
idx_treino_ale = indices_embaralhados[:corte_treino]
idx_calib_ale = indices_embaralhados[corte_treino:corte_calib]
idx_teste_ale = indices_embaralhados[corte_calib:]

print("divisao por esqueleto -> treino", len(idx_treino_esq),
      "calib", len(idx_calib_esq), "teste", len(idx_teste_esq))
print("divisao aleatoria     -> treino", len(idx_treino_ale),
      "calib", len(idx_calib_ale), "teste", len(idx_teste_ale))''')

md("""### 4.2 — Diagramas do particionamento e da validação cruzada

Dois esquemas desenhados no próprio notebook (Plotly, sem imagem externa). O
primeiro mostra os três blocos em proporção — treino, **calibração** (o bloco do
meio; não é um conjunto de validação, e sim o reservado para a Seção 8) e teste.
O segundo mostra a validação cruzada em k dobras: repare que ela **não tem um
bloco fixo de validação** — o papel de teste **roda** por todas as dobras, e é
exatamente assim que ela valida sem reservar um conjunto à parte. Por isso o
"conjunto de validação" não aparece como um bloco no segundo diagrama: ele foi
substituído pelo rodízio.""")
code(r'''# diagrama 1: blocos treino/calibracao/teste
tamanhos = [len(idx_treino_esq), len(idx_calib_esq), len(idx_teste_esq)]
nomes = ["treino", "calibracao", "teste"]
cores = ["#3266ad", "#7e9603", "#c0392b"]
figura_blocos = go.Figure()
inicio = 0
for nome, tamanho, cor in zip(nomes, tamanhos, cores):
    figura_blocos.add_trace(go.Bar(
        y=["particao"], x=[tamanho], name=nome, orientation="h", marker_color=cor,
        text=nome + "<br>" + str(tamanho), textposition="inside"))
    inicio = inicio + tamanho
figura_blocos.update_layout(barmode="stack", height=180,
                            title="Particionamento treino / calibracao / teste",
                            showlegend=False)
figura_blocos.show()

# diagrama 2: validacao cruzada em k dobras
K = 5
figura_cv = go.Figure()
for rodada in range(K):
    for bloco in range(K):
        eh_teste = (bloco == rodada)
        figura_cv.add_trace(go.Bar(
            y=["rodada " + str(rodada + 1)], x=[1], orientation="h",
            marker_color=("#c0392b" if eh_teste else "#3266ad"),
            marker_line_color="white", marker_line_width=2,
            showlegend=False,
            hovertext=("teste" if eh_teste else "treino")))
figura_cv.update_layout(barmode="stack", height=280,
                        title="Validacao cruzada em " + str(K) + " dobras (vermelho = teste)",
                        xaxis_showticklabels=False)
figura_cv.show()''')

md("""### 4.3 — O espaço químico das duas divisões, lado a lado

Vamos **ver** as duas divisões. Aqui há uma escolha importante de projeção:
**projetamos o esqueleto (Bemis-Murcko) de cada molécula, não a molécula inteira.**
O motivo é que a divisão é feita *por esqueleto* — então essa é a lente natural
para enxergá-la. Como cada molécula passa a ocupar a posição do seu **núcleo**,
todas as moléculas de um mesmo esqueleto **caem exatamente no mesmo ponto**. É esse
colapso que torna o mecanismo visível.

Cada ponto é pintado pelo **conjunto** a que a molécula pertence (treino,
calibração ou teste), uma vez para a divisão aleatória, outra para a por esqueleto.
O que procurar:

- na **por esqueleto**, cada aglomerado de pontos coincidentes é de **uma cor só**
  — o esqueleto inteiro foi para um único conjunto; treino e teste nunca partilham
  o mesmo núcleo;
- na **aleatória**, o mesmo aglomerado aparece com **cores misturadas** — os
  análogos de um núcleo foram rasgados entre treino e teste. **Essa mistura é o
  vazamento**, agora visível a olho nu.

**Uma ressalva honesta:** estes dois eixos de PCA ainda resumem só parte da
variação (o valor é impresso abaixo), então não leia distâncias absolutas ao pé da
letra. O que o gráfico mostra bem é o **mecanismo** — a coincidência de pontos por
esqueleto e como cada divisão a colore. A magnitude da separação a gente confirma
com um número na célula 4.3b.

*(E se coloríssemos por classe FORTE/FRACO em vez de por conjunto? As duas classes
apareceriam bem misturadas — dois eixos de PCA não separam as classes, e não
deveriam: se um gráfico 2D já separasse, não precisaríamos de modelo nenhum. A
separação que importa para uma avaliação honesta é entre treino e teste.)*""")
code(r'''from sklearn.decomposition import PCA

# fingerprint de Morgan do ESQUELETO (nao da molecula inteira) de cada linha.
# moleculas de mesmo esqueleto ficam no mesmo ponto; molecula sem anel tem
# esqueleto vazio -> vetor de zeros (cai na origem).
lista_fp_esqueleto = []
for molecula in agregados["molecula"]:
    mol_esqueleto = MurckoScaffold.GetScaffoldForMol(molecula)
    vetor = np.zeros((N_BITS,), dtype=np.int8)
    if mol_esqueleto is not None and mol_esqueleto.GetNumAtoms() > 0:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol_esqueleto, RAIO_MORGAN, nBits=N_BITS)
        DataStructs.ConvertToNumpyArray(fp, vetor)
    lista_fp_esqueleto.append(vetor)
matriz_fp_esqueleto = np.array(lista_fp_esqueleto)

# projecao 2D dos ESQUELETOS (uma PCA so, compartilhada pelas duas visualizacoes)
pca = PCA(n_components=2, random_state=SEMENTE)
coords = pca.fit_transform(matriz_fp_esqueleto.astype(float))
agregados["pca_x"] = coords[:, 0]
agregados["pca_y"] = coords[:, 1]

# quanto da variacao dos esqueletos estes 2 eixos realmente capturam
var_pc1 = pca.explained_variance_ratio_[0]
var_pc2 = pca.explained_variance_ratio_[1]
var_total = var_pc1 + var_pc2
print("variancia explicada (esqueletos) -> PC1:", round(100 * var_pc1, 1), "%",
      "| PC2:", round(100 * var_pc2, 1), "%",
      "| juntos:", round(100 * var_total, 1), "%")
print("(o resto da estrutura vive nas dezenas de eixos que nao vemos aqui)")

# marca, para cada molecula, a qual conjunto ela pertence em cada divisao
# (uma coluna por divisao, so para colorir a projecao)
for nome_coluna, idx_tr, idx_ca, idx_te in [
    ("conjunto_aleatorio", idx_treino_ale, idx_calib_ale, idx_teste_ale),
    ("conjunto_esqueleto", idx_treino_esq, idx_calib_esq, idx_teste_esq),
]:
    pertence = {}
    for indice in idx_tr: pertence[indice] = "treino"
    for indice in idx_ca: pertence[indice] = "calibracao"
    for indice in idx_te: pertence[indice] = "teste"
    coluna = []
    for indice in agregados.index:
        coluna.append(pertence.get(indice, "?"))
    agregados[nome_coluna] = coluna

titulo_esq = "Divisao por esqueleto (cada nucleo de uma cor so)"
figura_espaco = make_subplots(rows=1, cols=2,
    subplot_titles=("Divisao aleatoria (nucleos com cores misturadas)", titulo_esq))
mapa_cores = {"treino": "#3266ad", "calibracao": "#7e9603", "teste": "#c0392b"}
for coluna_conjunto, col in [("conjunto_aleatorio", 1), ("conjunto_esqueleto", 2)]:
    for nome_conjunto in ["treino", "calibracao", "teste"]:
        sub = agregados[agregados[coluna_conjunto] == nome_conjunto]
        figura_espaco.add_trace(go.Scattergl(
            x=sub["pca_x"], y=sub["pca_y"], mode="markers", name=nome_conjunto,
            marker=dict(size=4, color=mapa_cores[nome_conjunto], opacity=0.5),
            showlegend=(col == 1),
            text=sub["esqueleto"], customdata=sub["pic50"],
            hovertemplate="pIC50=%{customdata:.2f}<br>esqueleto: %{text}<extra></extra>"),
            row=1, col=col)
rotulo_var = "(PC1+PC2 = " + str(round(100 * var_total, 1)) + "% da variancia)"
figura_espaco.update_layout(height=430,
    title="Espaco quimico dos ESQUELETOS (PCA) " + rotulo_var)
figura_espaco.show()''')

md("""### 4.3b — A separação que o olho não vê, medida com um número

Em vez de confiar no gráfico, medimos diretamente **quão longe do treino está o
teste** em cada divisão. Para cada molécula de teste, calculamos a **maior
similaridade de Tanimoto** contra qualquer molécula de treino (1 = idêntica, 0 =
nada em comum). A média dessas similaridades resume a divisão:

- na divisão **aleatória**, cada molécula de teste costuma ter um primo quase
  idêntico no treino — a média fica **alta**;
- na divisão **por esqueleto**, os núcleos do teste foram mantidos fora do treino —
  a média cai. **É esse número menor que a projeção 2D não conseguia mostrar.**""")
code(r'''def similaridade_media_ao_treino(idx_treino, idx_teste):
    # fingerprints do RDKit (para o Tanimoto) de cada conjunto
    fps_treino = []
    for indice in idx_treino:
        molecula = agregados.loc[indice, "molecula"]
        fps_treino.append(AllChem.GetMorganFingerprintAsBitVect(molecula, RAIO_MORGAN, nBits=N_BITS))
    maiores_similaridades = []
    for indice in idx_teste:
        molecula = agregados.loc[indice, "molecula"]
        fp = AllChem.GetMorganFingerprintAsBitVect(molecula, RAIO_MORGAN, nBits=N_BITS)
        similaridades = DataStructs.BulkTanimotoSimilarity(fp, fps_treino)
        maiores_similaridades.append(max(similaridades))
    return float(np.mean(maiores_similaridades))

sim_aleatoria = similaridade_media_ao_treino(idx_treino_ale, idx_teste_ale)
sim_esqueleto = similaridade_media_ao_treino(idx_treino_esq, idx_teste_esq)
print("Tanimoto media do teste ao vizinho mais proximo no treino:")
print("  divisao aleatoria  :", round(sim_aleatoria, 3), "(teste tem primos proximos no treino)")
print("  divisao esqueleto  :", round(sim_esqueleto, 3), "(teste mais distante do treino)")
print("  queda              :", round(sim_aleatoria - sim_esqueleto, 3),
      "-> por isso a divisao por esqueleto e o teste mais honesto")''')

mdq("""**Pergunta.** Se a divisão aleatória dá acurácia mais alta que a por
esqueleto, qual das duas estima melhor o desempenho em moléculas realmente novas?""",
"""**Resposta.** A por esqueleto. A acurácia mais alta da aleatória é ilusória:
ela vem de o modelo reencontrar no teste primos próximos de moléculas do treino
(a similaridade média alta que medimos em 4.3b). Em uma triagem real, as moléculas
candidatas têm núcleos novos — a situação que a divisão por esqueleto reproduz, com
sua similaridade média mais baixa. Preferimos a estimativa mais baixa e mais
honesta.""")


# ══════════════════════════════════════════════════════════════════════════
# SEÇÃO 5 — MODELOS
# ══════════════════════════════════════════════════════════════════════════
md("""## Seção 5 — Os modelos

Todos os modelos que precisam conversar com o resto do notebook vêm do
**scikit-learn** e compartilham a mesma interface: `fit`, `predict`,
`predict_proba`. Isso não é detalhe: é o que permite avaliar, interpretar e
calibrar todos eles com o **mesmo** código, sem "adaptadores" que o aluno teria
de decifrar.

A partir daqui usamos a **divisão por esqueleto** como padrão (a honesta).""")

md("""**A célula abaixo prepara o material dos modelos.** Até aqui a divisão por
esqueleto (Seção 4.1) nos deu três listas de **rótulos de linha** —
`idx_treino_esq`, `idx_calib_esq`, `idx_teste_esq` —, que são os índices do
DataFrame `agregados`. Mas `X` e `y` são **arrays NumPy**, indexados por
**posição** (0, 1, 2, …), não por esses rótulos. Então o primeiro passo é
**traduzir cada rótulo na sua posição** com `agregados.index.get_loc(...)`, e só
então fatiar `X` e `y` nos três conjuntos. É pura logística de indexação — nenhum
modelo ainda.

Aproveitamos a mesma célula para duas coisas que todo o resto da seção vai usar:
a **linha de base** do teste (a acurácia de quem chuta sempre a classe
maioritária — o piso que qualquer modelo precisa superar) e dois **dicionários
vazios**, `modelos_treinados` e `tempos_treino`, onde cada modelo vai se guardar
para a comparação única da Seção 5.5.""")
code(r'''# recorta X e y para treino/calibracao/teste (divisao por esqueleto).
# X e y estao na ordem de agregados.index; convertemos cada indice na sua posicao.
pos_treino = [agregados.index.get_loc(indice) for indice in idx_treino_esq]
pos_calib  = [agregados.index.get_loc(indice) for indice in idx_calib_esq]
pos_teste  = [agregados.index.get_loc(indice) for indice in idx_teste_esq]
X_treino, y_treino = X[pos_treino], y[pos_treino]
X_calib,  y_calib  = X[pos_calib],  y[pos_calib]
X_teste,  y_teste  = X[pos_teste],  y[pos_teste]
print("treino", X_treino.shape, "| calib", X_calib.shape, "| teste", X_teste.shape)

# linha de base do teste: chutar sempre a classe maioritaria do treino
classe_maioritaria = 1 if (y_treino == 1).mean() >= 0.5 else 0
acuracia_base = (y_teste == classe_maioritaria).mean()
print("linha de base (classe maioritaria) no teste:", round(acuracia_base, 3))

# guardaremos aqui cada modelo treinado e seu tempo, para comparar todos juntos em 5.5
modelos_treinados = {}
tempos_treino = {}''')

md("""### 5.0 — Como vamos avaliar

Treinamos cada modelo em sua própria célula (guardando-o num dicionário
`modelos_treinados`) e, ao final, comparamos **todos de uma vez** com um laço
explícito na Seção 5.5. Assim o código de avaliação aparece **uma vez só**, e você
vê o mesmo cálculo aplicado igualmente a cada modelo. A métrica principal é o
**MCC** (coeficiente de correlação de Matthews): diferente da acurácia, ele não se
deixa enganar por classes desbalanceadas (base = 0; perfeito = 1).""")

md("""### 5.1 — Regressão logística

O modelo mais simples que ainda é um modelo — e a referência contra a qual os
outros são julgados. Ele faz duas coisas: (1) calcula uma **soma ponderada** das
features, $z = w_1 x_1 + w_2 x_2 + \\dots + b$ (cada peso $w$ diz o quanto aquela
feature conta, e o sinal, para que lado); (2) passa esse $z$ por uma **sigmoide**,
que espreme qualquer número real para o intervalo $[0, 1]$ — virando a
**probabilidade** de a molécula ser FORTE. Se a probabilidade passa de 0,5, a
predição é FORTE; senão, FRACO. Os pesos são **interpretáveis**: o sinal diz se a
feature empurra para FORTE ou para FRACO, algo que retomaremos na Seção 6.

O gráfico abaixo mostra exatamente essa conversão — a soma ponderada no eixo x, a
probabilidade no eixo y, e o limiar de 0,5 que separa as duas decisões.""")
code(r'''# figura didatica: como a regressao logistica transforma a soma ponderada
# (z) numa probabilidade, via sigmoide. NAO usa os dados; ilustra o mecanismo.
z = np.linspace(-6, 6, 300)
probabilidade = 1.0 / (1.0 + np.exp(-z))          # a funcao sigmoide

figura_logistica, eixo_log = plt.subplots(figsize=(6.4, 4.0))
eixo_log.plot(z, probabilidade, color="#3266ad", linewidth=2.5)
eixo_log.axhline(0.5, color="gray", linestyle="--", linewidth=1)
eixo_log.axvline(0.0, color="gray", linestyle=":", linewidth=1)
eixo_log.fill_between(z, 0.5, probabilidade, where=(probabilidade >= 0.5),
                      color="#3266ad", alpha=0.12)
eixo_log.fill_between(z, probabilidade, 0.5, where=(probabilidade < 0.5),
                      color="#c0392b", alpha=0.12)
eixo_log.text(3.1, 0.14, "preve FORTE\n(prob > 0.5)", color="#3266ad",
              ha="center", fontsize=9)
eixo_log.text(-3.1, 0.86, "preve FRACO\n(prob < 0.5)", color="#c0392b",
              ha="center", fontsize=9)
eixo_log.set_xlabel("z = soma ponderada das features  (w . x + b)")
eixo_log.set_ylabel("probabilidade de FORTE = sigmoide(z)")
eixo_log.set_title("Regressao logistica: da soma ponderada a probabilidade")
eixo_log.set_ylim(-0.02, 1.02)
plt.tight_layout()
plt.show()''')

md("""**Como funciona o `Pipeline` do scikit-learn.** A célula de treino não chama
a regressão logística sozinha — ela a embrulha num `Pipeline`, que **encadeia
etapas e as trata como um único modelo**. Aqui são duas etapas nomeadas:
`("escala", StandardScaler())`, que padroniza cada feature (subtrai a média,
divide pelo desvio, deixando tudo na mesma escala — a regressão logística é
sensível a isso), e `("clf", LogisticRegression(...))`, o classificador em si. O
`Pipeline` expõe a mesma interface de qualquer modelo (`fit`, `predict`,
`predict_proba`) e roteia os dados pelos passos na ordem:

- no **`fit`**: o scaler **aprende** a média e o desvio *do treino* e transforma o
  treino; a regressão logística treina sobre esse treino já padronizado;
- no **`predict`**: o scaler **reaplica ao teste os mesmos números que aprendeu no
  treino** (não recalcula) e a regressão logística prediz.

**Por que embrulhar, em vez de padronizar antes?** Para não vazar. Se ajustássemos
o `StandardScaler` no conjunto **todo** antes de dividir, a média e o desvio
carregariam informação do teste para dentro do treino, e a estimativa de
desempenho sairia otimista. Dentro do `Pipeline`, como o `fit` só enxerga
`X_treino`, o scaler nunca "vê" o teste. É o mesmo vazamento da partição (Seção
4.3), agora no nível da transformação das features — e o testaremos de frente na
Seção 6.5. Os modelos sensíveis à escala (5.1 e a SVM da 5.2) usam esse padrão; a
floresta (5.3) dispensa o scaler.""")
codex(r'''inicio = time.time()
modelo_logistico = Pipeline([
    ("escala", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=SEMENTE)),
])
modelo_logistico.fit(X_treino, y_treino)
tempo = time.time() - inicio

modelos_treinados["Regressao logistica"] = modelo_logistico
tempos_treino["Regressao logistica"] = tempo
mcc = matthews_corrcoef(y_teste, modelo_logistico.predict(X_teste))
print(f"Regressao logistica treinada em {tempo:.1f}s | MCC no teste = {mcc:.3f}")''',
"""Treine uma regressão logística dentro de um Pipeline com StandardScaler
(max_iter=1000, random_state=SEMENTE), meça o tempo com time.time(), guarde o
modelo em modelos_treinados["Regressao logistica"] e o tempo em tempos_treino, e
imprima o MCC no teste.""")

md("""### 5.2 — Máquina de vetores de suporte (SVM)

A SVM procura a fronteira que separa as classes com a **maior margem**. Com o
**truque do kernel**, ela traça fronteiras curvas sem calcular explicitamente as
coordenadas em alta dimensão. Vejamos a ideia em 2D, com dados sintéticos não
separáveis por uma reta e a fronteira do kernel RBF traçada sobre eles.""")
code(r'''from sklearn.datasets import make_circles

X_circulos, y_circulos = make_circles(n_samples=300, factor=0.4, noise=0.12,
                                      random_state=SEMENTE)
svm_demo = SVC(kernel="rbf", gamma=2.0, C=1.0, random_state=SEMENTE)
svm_demo.fit(X_circulos, y_circulos)

# malha para desenhar a fronteira de decisao
passo = 0.02
x_min, x_max = X_circulos[:, 0].min() - 0.3, X_circulos[:, 0].max() + 0.3
y_min, y_max = X_circulos[:, 1].min() - 0.3, X_circulos[:, 1].max() + 0.3
grade_x, grade_y = np.meshgrid(np.arange(x_min, x_max, passo),
                               np.arange(y_min, y_max, passo))
pontos_grade = np.c_[grade_x.ravel(), grade_y.ravel()]
decisao = svm_demo.decision_function(pontos_grade).reshape(grade_x.shape)

plt.figure(figsize=(5, 4.5))
plt.contourf(grade_x, grade_y, decisao, levels=20, cmap="RdBu", alpha=0.6)
plt.contour(grade_x, grade_y, decisao, levels=[0], colors="k", linewidths=1.5)
plt.scatter(X_circulos[:, 0], X_circulos[:, 1], c=y_circulos, cmap="RdBu",
            edgecolors="k", s=25)
plt.title("SVM com kernel RBF: fronteira nao linear")
plt.show()''')

md("""Agora a SVM sobre os dados reais. Atenção ao custo: kernel RBF sobre
milhares de amostras e 2048 bits é **lento** — é o gargalo mais provável do
tempo de execução. Por isso **subamostramos** o treino e avisamos. Usamos
`probability=True` porque precisamos das probabilidades para a curva ROC; isso
acrescenta uma calibração interna e multiplica o tempo — mais um motivo para
subamostrar. Esta célula pode levar cerca de 1 minuto.""")
codex(r'''MAX_SVM = 1500   # subamostra do treino para caber no orcamento de tempo
gerador_svm = np.random.RandomState(SEMENTE)
if len(X_treino) > MAX_SVM:
    escolhidos = gerador_svm.choice(len(X_treino), MAX_SVM, replace=False)
    X_treino_svm, y_treino_svm = X_treino[escolhidos], y_treino[escolhidos]
    print("SVM subamostrada para", MAX_SVM, "moleculas (de", len(X_treino), ") por custo de tempo")
else:
    X_treino_svm, y_treino_svm = X_treino, y_treino

inicio = time.time()
modelo_svm = Pipeline([
    ("escala", StandardScaler()),
    ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=SEMENTE)),
])
modelo_svm.fit(X_treino_svm, y_treino_svm)
tempo = time.time() - inicio

modelos_treinados["SVM (RBF)"] = modelo_svm
tempos_treino["SVM (RBF)"] = tempo
mcc = matthews_corrcoef(y_teste, modelo_svm.predict(X_teste))
print(f"SVM (RBF) treinada em {tempo:.1f}s | MCC no teste = {mcc:.3f}")''',
"""Treine uma SVM RBF dentro de um Pipeline com StandardScaler (probability=True,
random_state=SEMENTE). Subamostre o treino para no máximo 1500 moléculas e avise o
aluno. Meça o tempo, guarde em modelos_treinados["SVM (RBF)"] e tempos_treino, e
imprima o MCC no teste.""")

md("""### 5.3 — Floresta aleatória

Antes da floresta, a **árvore**. Uma árvore de decisão classifica por uma sequência
de perguntas de sim/não sobre uma feature de cada vez ("TPSA > 60?", "tem o bit
512?"). Como ela **escolhe** cada pergunta? Testa todas as features e todos os
cortes possíveis e fica com o que **melhor separa as classes** — o que deixa os
dois lados mais "puros" (mais perto de conter uma classe só, medido pela impureza
de Gini). Feito o corte, repete o processo em cada lado, recursivamente, até os
grupos ficarem puros ou até bater um limite de profundidade. Cada caixa no fim do
caminho é uma **folha**, rotulada pela classe majoritária das moléculas de treino
que ali caíram. Para **prever**, uma molécula nova desce a árvore respondendo às
perguntas e recebe o rótulo da folha onde parar.

Uma árvore fundo demais decora o treino (sobreajuste). Desenhamos uma árvore rasa
(profundidade 3), legível, só para ver o mecanismo. Depois, a **floresta**: muitas
árvores treinadas em reamostras diferentes dos dados e das features, cujo **voto
médio** reduz a variância de uma árvore isolada — o todo generaliza melhor que
qualquer árvore sozinha.""")
code(r'''# arvore rasa, so para visualizar o mecanismo (nao entra na comparacao)
arvore_rasa = DecisionTreeClassifier(max_depth=3, random_state=SEMENTE)
arvore_rasa.fit(X_treino, y_treino)
plt.figure(figsize=(13, 6))
plot_tree(arvore_rasa, max_depth=3, feature_names=nomes_features,
          class_names=["FRACO", "FORTE"], filled=True, fontsize=7, impurity=False)
plt.title("Uma arvore de decisao rasa (profundidade 3)")
plt.show()''')
codex(r'''inicio = time.time()
modelo_floresta = RandomForestClassifier(
    n_estimators=300, max_depth=None, n_jobs=-1, random_state=SEMENTE)
modelo_floresta.fit(X_treino, y_treino)
tempo = time.time() - inicio

modelos_treinados["Floresta aleatoria"] = modelo_floresta
tempos_treino["Floresta aleatoria"] = tempo
mcc = matthews_corrcoef(y_teste, modelo_floresta.predict(X_teste))
print(f"Floresta aleatoria treinada em {tempo:.1f}s | MCC no teste = {mcc:.3f}")''',
"""Treine uma RandomForestClassifier (n_estimators=300, n_jobs=-1,
random_state=SEMENTE), meça o tempo, guarde em modelos_treinados["Floresta
aleatoria"] e tempos_treino, e imprima o MCC no teste.""")

md("""### 5.4a — Rede neural (scikit-learn)

Uma rede neural encadeia **camadas**: cada camada combina as entradas com pesos e
aplica uma **ativação** não linear. Termos, na primeira vez que aparecem:
**época** é uma passada por todos os dados; **minilote** é um punhado de exemplos
processado por vez; **taxa de aprendizado** é o tamanho do passo com que os pesos
são ajustados. O `MLPClassifier` faz tudo isso por dentro.

Usamos `early_stopping=True`: o treino para quando o desempenho numa fração de
validação deixa de melhorar. Plotamos as duas curvas por época — perda de treino
(`loss_curve_`) e desempenho na validação (`validation_scores_`). O descolamento
entre elas é o **sobreajuste** ficando visível.""")
codex(r'''inicio = time.time()
modelo_rede = MLPClassifier(
    hidden_layer_sizes=(128, 64), activation="relu",
    early_stopping=True, n_iter_no_change=10, max_iter=200,
    random_state=SEMENTE)
modelo_rede.fit(X_treino, y_treino)
tempo = time.time() - inicio

modelos_treinados["Rede neural (MLP)"] = modelo_rede
tempos_treino["Rede neural (MLP)"] = tempo
mcc = matthews_corrcoef(y_teste, modelo_rede.predict(X_teste))
print(f"Rede neural (MLP) treinada em {tempo:.1f}s | MCC no teste = {mcc:.3f}")

figura_rede = make_subplots(specs=[[{"secondary_y": True}]])
epocas = list(range(1, len(modelo_rede.loss_curve_) + 1))
figura_rede.add_trace(go.Scatter(x=epocas, y=modelo_rede.loss_curve_,
                                 name="perda de treino", line=dict(color="#3266ad")),
                      secondary_y=False)
figura_rede.add_trace(go.Scatter(x=list(range(1, len(modelo_rede.validation_scores_) + 1)),
                                 y=modelo_rede.validation_scores_,
                                 name="acuracia na validacao", line=dict(color="#7e9603")),
                      secondary_y=True)
figura_rede.update_layout(title="MLP: perda de treino x desempenho na validacao", height=380)
figura_rede.show()''',
"""Treine um MLPClassifier hidden_layer_sizes=(128, 64), activation='relu',
early_stopping=True, max_iter=200, random_state=SEMENTE. Guarde-o em
modelos_treinados["Rede neural (MLP)"] e o tempo em tempos_treino, e imprima o MCC.
Depois plote em Plotly a loss_curve_ (perda de treino) e validation_scores_
(validação) por época, e explique o sobreajuste onde as curvas se descolam.""")

md("""### 5.4b — A mesma rede, aberta, em PyTorch (não opcional)

A parte (a) chama `fit` e o treino acontece numa caixa preta. Aqui abrimos a
caixa: reimplementamos **a mesma** rede em PyTorch e escrevemos o laço de treino
à mão. O objetivo não é um modelo melhor — é **ver o que o `fit` faz por dentro**.

Antes do código, os conceitos. Um **gradiente** é a direção em que a perda mais
cresce; andamos no sentido oposto para diminuí-la. A cada minilote, quatro passos:
zerar os gradientes acumulados, calcular a saída (passagem direta), medir a perda,
e então `loss.backward()` — que calcula os gradientes por retropropagação —
seguido de `optimizer.step()`, que dá o passo. `backward()` vem **depois** da
perda (precisa dela para saber o erro) e **antes** do passo (que usa os gradientes
que ela produziu).

Esta rede não conversa com o resto do notebook, então não precisa de invólucro:
fatiamos os tensores em minilotes com um laço sobre índices embaralhados, sem
`DataLoader`.""")
code(r'''# preparo: padroniza (mesma ideia dos pipelines) e converte para tensores
from sklearn.preprocessing import StandardScaler as _Escala
escala_torch = _Escala().fit(X_treino)
Xt_treino = torch.tensor(escala_torch.transform(X_treino), dtype=torch.float32)
yt_treino = torch.tensor(y_treino, dtype=torch.long)
Xt_teste = torch.tensor(escala_torch.transform(X_teste), dtype=torch.float32)

class RedeMLP(nn.Module):
    """Mesma arquitetura da parte (a): 2 camadas ocultas ReLU, saida com 2 classes."""
    def __init__(self, n_entradas):
        super().__init__()
        self.camada1 = nn.Linear(n_entradas, 128)
        self.camada2 = nn.Linear(128, 64)
        self.saida = nn.Linear(64, 2)
        self.ativacao = nn.ReLU()

    def forward(self, entrada):
        oculta1 = self.ativacao(self.camada1(entrada))
        oculta2 = self.ativacao(self.camada2(oculta1))
        return self.saida(oculta2)

torch.manual_seed(SEMENTE)
rede_torch = RedeMLP(X_treino.shape[1]).to(DISPOSITIVO)
funcao_perda = nn.CrossEntropyLoss()
otimizador = torch.optim.Adam(rede_torch.parameters(), lr=0.001)
print("rede criada em", DISPOSITIVO, "| parametros:",
      sum(p.numel() for p in rede_torch.parameters()))''')

md("""**Vamos desenhar a arquitetura que acabamos de montar** — o diagrama real,
gerado a partir do próprio modelo, não um desenho decorativo. Para modelos PyTorch,
o pacote mais prático hoje é o **`torchview`** (`draw_graph`): ele passa um tensor
de exemplo pela rede, rastreia as operações e desenha as camadas **com as formas
dos tensores** em cada ponto — do vetor de entrada até os 2 logits de saída.
(Alternativas, se quiser explorar: o **`torchinfo`** dá uma tabela textual no
estilo do `model.summary()` do Keras; o **`torchviz`** desenha o grafo de
*autograd*, isto é, as operações da retropropagação, útil para depurar em vez da
arquitetura.)

O `torchview` não vem no Colab, então a célula abaixo o **instala** (rápido) e
gera o diagrama. Antes dele, imprimimos um resumo camada a camada com a contagem
de parâmetros.""")
code(r'''# instala o torchview se faltar (mesmo padrao robusto da Secao 0)
import importlib.util
import subprocess
import sys
if importlib.util.find_spec("torchview") is None:
    print("instalando torchview...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torchview"], check=True)

# resumo textual camada a camada, direto do modelo
print("camada a camada:")
for nome_camada, modulo in rede_torch.named_children():
    n_parametros = sum(p.numel() for p in modulo.parameters())
    print("  ", nome_camada, "->", modulo, "| parametros:", n_parametros)
print("total de parametros:", sum(p.numel() for p in rede_torch.parameters()))

# diagrama real da arquitetura, com as formas dos tensores (renderiza no Colab)
from torchview import draw_graph
grafo_rede = draw_graph(rede_torch, input_size=(1, X_treino.shape[1]),
                        device=str(DISPOSITIVO), graph_name="RedeMLP")
grafo_rede.visual_graph   # ultima expressao: o Colab desenha o diagrama aqui''')

md("""O laço de treino, comentado passo a passo, instrumentado com o
`SummaryWriter` do TensorBoard (registra a perda por época) e com uma barra
`tqdm`. Registramos também, ao lado, a `loss_curve_` do `MLPClassifier` da parte
(a), para que as duas apareçam no mesmo painel. Guardamos a curva do PyTorch em
uma lista para replicá-la em Plotly logo depois — o painel do TensorBoard não
fica salvo no arquivo.""")
code(r'''import os
import shutil

PASTA_LOGS = "logs_tb"
if os.path.exists(PASTA_LOGS):
    shutil.rmtree(PASTA_LOGS)   # limpa logs antigos para nao acumular curvas
os.makedirs(PASTA_LOGS, exist_ok=True)

N_EPOCAS = 40
TAM_MINILOTE = 64
escritor = SummaryWriter(os.path.join(PASTA_LOGS, "pytorch"))

perda_por_epoca_torch = []
n_treino = Xt_treino.shape[0]
for epoca in tqdm(range(N_EPOCAS), desc="treino PyTorch"):
    rede_torch.train()
    # embaralha os indices a cada epoca e fatia em minilotes
    ordem = torch.randperm(n_treino)
    perda_acumulada = 0.0
    n_lotes = 0
    for inicio_lote in range(0, n_treino, TAM_MINILOTE):
        indices_lote = ordem[inicio_lote:inicio_lote + TAM_MINILOTE]
        entradas = Xt_treino[indices_lote].to(DISPOSITIVO)
        alvos = yt_treino[indices_lote].to(DISPOSITIVO)

        otimizador.zero_grad()                 # 1. zera gradientes acumulados
        saidas = rede_torch(entradas)          # 2. passagem direta
        perda = funcao_perda(saidas, alvos)    # 3. mede a perda
        perda.backward()                       # 4. retropropaga (calcula gradientes)
        otimizador.step()                      # 5. da o passo (ajusta os pesos)

        perda_acumulada = perda_acumulada + perda.item()
        n_lotes = n_lotes + 1
    perda_media = perda_acumulada / n_lotes
    perda_por_epoca_torch.append(perda_media)
    escritor.add_scalar("perda/treino", perda_media, epoca)

# registra tambem a curva do MLPClassifier (parte a) em subpasta propria
escritor_mlp = SummaryWriter(os.path.join(PASTA_LOGS, "sklearn_mlp"))
for indice_epoca in range(len(modelo_rede.loss_curve_)):
    escritor_mlp.add_scalar("perda/treino", modelo_rede.loss_curve_[indice_epoca], indice_epoca)
escritor.close()
escritor_mlp.close()
print("treino PyTorch concluido; perda final:", round(perda_por_epoca_torch[-1], 4))''')

md("""**TensorBoard (célula de Colab).** No Colab, abra o painel **antes** de
treinar para vê-lo atualizar ao vivo. Como o painel não fica salvo no arquivo,
replicamos toda curva em Plotly na célula seguinte — o TensorBoard é a experiência
ao vivo; o Plotly é o material de estudo. Fora do Colab, esta célula apenas avisa.""")
code(r'''# Em Colab, as duas linhas idiomaticas sao:
#   %load_ext tensorboard
#   %tensorboard --logdir logs_tb
# Usamos a forma robusta abaixo para nao quebrar fora do Colab.
try:
    get_ipython().run_line_magic("load_ext", "tensorboard")
    get_ipython().run_line_magic("tensorboard", "--logdir " + PASTA_LOGS)
except Exception as erro:
    print("TensorBoard so abre no Colab/Jupyter com a extensao:", type(erro).__name__)
    print("As curvas estao replicadas em Plotly na proxima celula.")''')

md("""Registro permanente em Plotly: as duas curvas de perda sobrepostas.
Elas **não coincidem**, embora sejam a mesma arquitetura — inicialização de pesos
diferente, otimizadores diferentes, ordem dos minilotes diferente. Que dois
códigos corretos para o mesmo modelo deem curvas diferentes é, em si, uma lição
sobre estocasticidade em aprendizado de máquina.""")
code(r'''figura_perdas = go.Figure()
figura_perdas.add_trace(go.Scatter(
    x=list(range(1, len(perda_por_epoca_torch) + 1)), y=perda_por_epoca_torch,
    name="PyTorch (laco a mao)", line=dict(color="#c0392b")))
figura_perdas.add_trace(go.Scatter(
    x=list(range(1, len(modelo_rede.loss_curve_) + 1)), y=modelo_rede.loss_curve_,
    name="MLPClassifier (sklearn)", line=dict(color="#3266ad")))
figura_perdas.update_layout(title="Mesma rede, duas implementacoes: perda por epoca",
                            xaxis_title="epoca", yaxis_title="perda", height=380)
figura_perdas.show()''')

md("""**Comparação de execuções no TensorBoard.** Onde o TensorBoard ganha da
curva estática é comparar execuções. Treinamos a mesma rede três vezes com taxas
de aprendizado diferentes, cada uma gravada em sua **subpasta** de `logs_tb`. A
célula abaixo treina e grava as três; a seguinte **abre o painel do TensorBoard**,
onde as três curvas aparecem sobrepostas; e o Plotly na sequência é o registro
permanente.""")
code(r'''taxas = [0.0001, 0.001, 0.05]   # baixa, boa, alta demais
curvas_por_taxa = {}

# treina a MESMA rede uma vez para cada taxa de aprendizado (mesmo laco da 5.4b)
for taxa in taxas:
    torch.manual_seed(SEMENTE)
    rede = RedeMLP(X_treino.shape[1]).to(DISPOSITIVO)
    otim = torch.optim.Adam(rede.parameters(), lr=taxa)
    escritor_taxa = SummaryWriter(os.path.join(PASTA_LOGS, "lr_" + str(taxa)))
    curva = []
    for epoca in range(N_EPOCAS):
        ordem = torch.randperm(n_treino)
        soma, n_lotes = 0.0, 0
        for inicio_lote in range(0, n_treino, TAM_MINILOTE):
            idx = ordem[inicio_lote:inicio_lote + TAM_MINILOTE]
            otim.zero_grad()
            perda = funcao_perda(rede(Xt_treino[idx].to(DISPOSITIVO)),
                                 yt_treino[idx].to(DISPOSITIVO))
            perda.backward()
            otim.step()
            soma += perda.item()
            n_lotes += 1
        media = soma / n_lotes
        curva.append(media)
        escritor_taxa.add_scalar("perda/treino", media, epoca)
    escritor_taxa.close()
    curvas_por_taxa[taxa] = curva
print("tres execucoes gravadas em", PASTA_LOGS, "(subpastas lr_*)")''')

md("""O painel do TensorBoard com as três execuções. No Colab, as curvas das três
taxas aparecem sobrepostas no mesmo gráfico (uma cor por subpasta `lr_*`) e você
pode ligar/desligar cada uma. É aqui que ele supera a figura estática: comparar
execuções lado a lado sem redesenhar nada.""")
code(r'''# reabre o painel do TensorBoard, agora com as tres execucoes de taxa
try:
    get_ipython().run_line_magic("tensorboard", "--logdir " + PASTA_LOGS)
except Exception as erro:
    print("TensorBoard so aparece no Colab/Jupyter:", type(erro).__name__)
    print("As tres curvas estao replicadas no grafico Plotly abaixo.")''')

md("""Registro permanente em Plotly das mesmas três curvas.""")
code(r'''figura_taxas = go.Figure()
for taxa in taxas:
    figura_taxas.add_trace(go.Scatter(
        x=list(range(1, N_EPOCAS + 1)), y=curvas_por_taxa[taxa],
        name="lr = " + str(taxa)))
figura_taxas.update_layout(title="Tres taxas de aprendizado (mesma rede)",
                           xaxis_title="epoca", yaxis_title="perda", height=380)
figura_taxas.show()''')

mdq("""**Pergunta.** Entre as três taxas de aprendizado, qual é "alta demais" e
qual é a assinatura visual disso na curva de perda?""",
"""**Resposta.** A maior (0.05). A assinatura é uma curva que **não desce de forma
estável**: oscila, sobe e desce, ou estaciona alta — o passo é grande demais e o
otimizador "salta por cima" do mínimo. A taxa muito baixa (0.0001) desce, mas
devagar demais; a intermediária desce de forma suave e consistente.""")

md("""### 5.4c — TensorBoard não é só para redes neurais

O TensorBoard aceita qualquer sequência de números via `add_scalar` — não precisa
ser a perda de uma rede. A rigor, os modelos do scikit-learn que não treinam por
época (regressão logística, SVM, floresta) não têm uma "curva de treino" a
registrar; mas dá para registrar **qualquer progressão** que faça sentido. Para a
floresta, a progressão natural é o desempenho **conforme se adicionam árvores**.
Usamos `warm_start=True` para crescer a floresta em etapas e registramos o MCC no
teste a cada etapa — primeiro gravamos no TensorBoard, depois abrimos o painel e,
como sempre, replicamos em Plotly.""")
code(r'''escritor_floresta = SummaryWriter(os.path.join(PASTA_LOGS, "floresta_arvores"))
floresta_incremental = RandomForestClassifier(
    n_estimators=10, warm_start=True, n_jobs=-1, random_state=SEMENTE)

n_arvores_etapas = list(range(20, 301, 20))
curva_floresta = []
for n_arvores in n_arvores_etapas:
    floresta_incremental.set_params(n_estimators=n_arvores)
    floresta_incremental.fit(X_treino, y_treino)     # warm_start: so cresce as novas arvores
    mcc = matthews_corrcoef(y_teste, floresta_incremental.predict(X_teste))
    curva_floresta.append(mcc)
    escritor_floresta.add_scalar("mcc/teste", mcc, n_arvores)
escritor_floresta.close()
print("curva MCC x n_arvores gravada em", PASTA_LOGS, "(subpasta floresta_arvores)")''')

md("""O painel do TensorBoard com a curva da floresta — o mesmo `%tensorboard`,
agora mostrando um escalar (`mcc/teste`) que **não** é a perda de uma rede.""")
code(r'''# reabre o painel; a subpasta floresta_arvores aparece junto das curvas de rede
try:
    get_ipython().run_line_magic("tensorboard", "--logdir " + PASTA_LOGS)
except Exception as erro:
    print("TensorBoard so aparece no Colab/Jupyter:", type(erro).__name__)
    print("A curva esta replicada no grafico Plotly abaixo.")''')

md("""Registro permanente em Plotly da mesma curva.""")
code(r'''figura_floresta = go.Figure(go.Scatter(
    x=n_arvores_etapas, y=curva_floresta, mode="lines+markers", line=dict(color="#1a7a4a")))
figura_floresta.update_layout(
    title="Floresta aleatoria: MCC no teste conforme se adicionam arvores",
    xaxis_title="numero de arvores", yaxis_title="MCC", height=360)
figura_floresta.show()
print("MCC estabiliza em torno de", round(curva_floresta[-1], 3),
      "- mais arvores alem disso quase nao mudam nada")''')

mdq("""**Pergunta.** A curva de MCC da floresta em função do número de árvores tem
o mesmo significado que a curva de perda por época da rede? O que cada uma diz?""",
"""**Resposta.** Não são a mesma coisa. A curva de **perda por época** da rede
mostra o modelo **aprendendo** (ajustando pesos) e pode revelar sobreajuste se a
validação descolar. A curva de **MCC × número de árvores** da floresta mostra
outra coisa: como o desempenho **converge** conforme agregamos mais estimadores —
ela sobe e **estabiliza** num platô (mais árvores não pioram, só param de ajudar).
São progressões diferentes; o TensorBoard registra ambas porque é só um plotador
de escalares ao longo de um eixo — o eixo é que muda de sentido (épocas vs.
árvores).""")

md("""### 5.5 — Comparação de todos os modelos

Agora a **avaliação em um laço só**: percorremos os modelos guardados e aplicamos
a cada um exatamente o mesmo cálculo de métricas. Como todos compartilham a
interface do scikit-learn (`predict`, `predict_proba`), o mesmo código serve para
todos — sem ramificações. Guardamos tudo em `resultados` para os gráficos.""")
code(r'''# avalia todos os modelos com o MESMO codigo, num laco explicito
resultados = {}
for nome, modelo in modelos_treinados.items():
    predito = modelo.predict(X_teste)
    if hasattr(modelo, "predict_proba"):
        proba = modelo.predict_proba(X_teste)[:, 1]
    else:
        proba = modelo.decision_function(X_teste)
    relatorio = classification_report(y_teste, predito, output_dict=True,
                                      target_names=["FRACO", "FORTE"], zero_division=0)
    resultados[nome] = {
        "MCC": matthews_corrcoef(y_teste, predito),
        "AUC": roc_auc_score(y_teste, proba),
        "prec_FORTE": relatorio["FORTE"]["precision"],
        "rec_FORTE": relatorio["FORTE"]["recall"],
        "prec_FRACO": relatorio["FRACO"]["precision"],
        "rec_FRACO": relatorio["FRACO"]["recall"],
        "tempo_s": tempos_treino[nome],
        "proba": proba, "predito": predito,
    }
    print(f"{nome}: MCC={resultados[nome]['MCC']:.3f}  AUC={resultados[nome]['AUC']:.3f}"
          f"  (base MCC=0.000, AUC=0.500)")''')

md("""Uma tabela e um gráfico com MCC, AUC, precisão e revocação por classe e
tempo de treino; depois as curvas ROC e de precisão-revocação sobrepostas e as
matrizes de confusão lado a lado. Nenhum número aparece sozinho: a linha de base
(MCC 0, AUC 0,5) está sempre à vista.""")
code(r'''tabela_comparacao = pd.DataFrame(resultados).T[
    ["MCC", "AUC", "prec_FORTE", "rec_FORTE", "prec_FRACO", "rec_FRACO", "tempo_s"]]
tabela_comparacao = tabela_comparacao.astype(float).round(3)
print(tabela_comparacao.to_string())

figura_barras = go.Figure()
for metrica, cor in [("MCC", "#3266ad"), ("AUC", "#7e9603")]:
    figura_barras.add_trace(go.Bar(x=tabela_comparacao.index,
                                   y=tabela_comparacao[metrica], name=metrica,
                                   marker_color=cor))
figura_barras.add_hline(y=0.5, line_dash="dash", line_color="#c0392b",
                        annotation_text="AUC base = 0,5")
figura_barras.update_layout(barmode="group", title="Comparacao de modelos (MCC e AUC)",
                            height=380)
figura_barras.show()''')
code(r'''# curvas ROC e precisao-revocacao sobrepostas
figura_curvas = make_subplots(rows=1, cols=2, subplot_titles=("ROC", "Precisao-Revocacao"))
for nome in resultados:
    proba = resultados[nome]["proba"]
    fpr, tpr, _ = roc_curve(y_teste, proba)
    figura_curvas.add_trace(go.Scatter(x=fpr, y=tpr, name=nome, mode="lines"), row=1, col=1)
    precisao, revocacao, _ = precision_recall_curve(y_teste, proba)
    figura_curvas.add_trace(go.Scatter(x=revocacao, y=precisao, name=nome,
                                       mode="lines", showlegend=False), row=1, col=2)
figura_curvas.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                   line=dict(dash="dash", color="gray"),
                                   name="acaso"), row=1, col=1)
figura_curvas.update_layout(height=400, title="ROC e Precisao-Revocacao (teste)")
figura_curvas.show()''')
code(r'''# matrizes de confusao lado a lado
nomes_modelos = list(resultados.keys())
figura_conf, eixos = plt.subplots(1, len(nomes_modelos),
                                  figsize=(3.2 * len(nomes_modelos), 3))
for eixo, nome in zip(eixos, nomes_modelos):
    matriz = confusion_matrix(y_teste, resultados[nome]["predito"])
    eixo.imshow(matriz, cmap="Greens")
    eixo.set_title(nome, fontsize=9)
    eixo.set_xticks([0, 1]); eixo.set_xticklabels(["FRACO", "FORTE"], fontsize=8)
    eixo.set_yticks([0, 1]); eixo.set_yticklabels(["FRACO", "FORTE"], fontsize=8)
    for i in range(2):
        for j in range(2):
            eixo.text(j, i, matriz[i, j], ha="center", va="center", fontsize=11)
plt.tight_layout(); plt.show()''')

mdq("""**Pergunta.** A diferença de desempenho entre os quatro modelos é grande ou
pequena? Como ela se compara à diferença que a escolha da partição (aleatória x
esqueleto) costuma causar?""",
"""**Resposta.** Depende do que os dados mostrarem nesta execução — olhe a coluna
MCC da tabela. O padrão típico, que você pode verificar refazendo a Seção 4 com a
divisão aleatória, é que a diferença **entre modelos** é menor do que a diferença
**entre esquemas de partição**. Trocar de algoritmo rende pouco; escolher a
avaliação honesta muda o número que você reporta.""")


md("""### 5.6 — E se o alvo fosse contínuo? Regressão do pIC50

Ao binarizar o pIC50 em FORTE/FRACO, jogamos fora informação: uma molécula com
pIC50 8,9 e outra com 6,1 viram a mesma classe "FORTE". A alternativa é **prever o
pIC50 diretamente** — uma tarefa de **regressão** (valor contínuo), não de
classificação.

Uma diferença importante liga isto à Seção 2.4: as moléculas que só têm medida
de limite `>` **não entram** na regressão. Elas não têm um valor pontual (só um
limite), e um regressor precisa de um número exato como alvo. Na classificação
elas eram FRACO por regra; na regressão, sem pIC50, saem. É o outro lado da mesma
decisão.

Como sempre, todo número vem **ao lado de referências**. Comparamos a floresta com
duas linhas de base: uma **linear** — o `Ridge`, regressão linear com um freio L2
nos pesos (a análoga da regressão logística, agora prevendo um número) — e uma
**trivial**, que chuta sempre a média do treino. As métricas: **RMSE** e **MAE**
(erro médio, em unidades de pIC50 — menor é melhor) e **R²** (fração da variância
explicada — maior é melhor; 0 = tão bom quanto chutar a média).

Um detalhe que **importa** no Ridge: a força do freio (`alpha`) não é chutada — é
escolhida por validação cruzada (`RidgeCV`). Aqui isso não é luxo: com ~2000 bits
de fingerprint esparsos, um freio fraco deixa o modelo linear **estourar** (pior
que a média); o freio certo é o que o torna uma referência honesta. É a mesma
lição da rede (5.4): num modelo linear, regularizar bem vale mais que o algoritmo.""")
codex(r'''from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# dados de regressao: so moleculas com pIC50 exato (as 'sem_medida_exata' saem).
# monta as posicoes de treino e de teste com dois lacos explicitos.
pos_reg_treino = []
for indice in idx_treino_esq:
    if not agregados.loc[indice, "sem_medida_exata"]:
        pos_reg_treino.append(agregados.index.get_loc(indice))
pos_reg_teste = []
for indice in idx_teste_esq:
    if not agregados.loc[indice, "sem_medida_exata"]:
        pos_reg_teste.append(agregados.index.get_loc(indice))

Xr_treino, yr_treino = X[pos_reg_treino], agregados["pic50"].values[pos_reg_treino]
Xr_teste, yr_teste = X[pos_reg_teste], agregados["pic50"].values[pos_reg_teste]
print("regressao -> treino:", Xr_treino.shape[0], "| teste:", Xr_teste.shape[0],
      "(medidas de limite removidas)")

# modelo principal: floresta de regressao (a familia que venceu a classificacao)
regressor = RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=SEMENTE)
regressor.fit(Xr_treino, yr_treino)
predito_reg = regressor.predict(Xr_teste)

# linha de base LINEAR: Ridge = regressao linear com freio L2, padronizada num Pipeline.
# a forca do freio (alpha) e ESCOLHIDA por validacao cruzada (RidgeCV) - crucial aqui:
# com freio fraco, o modelo linear estoura nos ~2000 bits esparsos e fica pior que a media.
alphas_testados = np.logspace(-1, 4, 20)
ridge = Pipeline([("escala", StandardScaler()), ("reg", RidgeCV(alphas=alphas_testados))])
ridge.fit(Xr_treino, yr_treino)
predito_ridge = ridge.predict(Xr_teste)
print("alpha escolhido pelo RidgeCV:", round(ridge.named_steps["reg"].alpha_, 2))

# linha de base TRIVIAL: prever sempre a media do treino
predito_media = np.full_like(yr_teste, yr_treino.mean())

# compara os tres com um laco (RMSE/MAE menores sao melhores; R2 maior e melhor)
print(f"{'modelo':20s} {'RMSE':>7s} {'MAE':>7s} {'R2':>7s}")
for nome_modelo, predito in [("Floresta (RF)", predito_reg),
                             ("Ridge (linear)", predito_ridge),
                             ("base (media)", predito_media)]:
    rmse = np.sqrt(mean_squared_error(yr_teste, predito))
    mae = mean_absolute_error(yr_teste, predito)
    r2 = r2_score(yr_teste, predito)
    print(f"{nome_modelo:20s} {rmse:7.3f} {mae:7.3f} {r2:7.3f}")

# R2 da floresta (o modelo desenhado no grafico predito x observado abaixo)
r2 = r2_score(yr_teste, predito_reg)''',
"""Monte os dados de regressão excluindo as moléculas que só têm medida '>'
(sem_medida_exata — elas não têm pIC50 pontual). Treine um RandomForestRegressor e,
como linha de base linear, um RidgeCV (dentro de um Pipeline com StandardScaler, que
escolhe o alpha por validação cruzada). Compare os dois com a base trivial (prever a
média) imprimindo RMSE, MAE e R2 lado a lado.""")

md("""O gráfico de predito × observado mostra o ajuste. As linhas tracejadas
marcam o limiar de potência: os pontos se dividem em quatro quadrantes que são,
na prática, a matriz de confusão da classificação obtida ao **aplicar o limiar às
predições da regressão** — as duas tarefas são duas vistas do mesmo sinal.""")
code(r'''figura_reg = px.scatter(
    x=yr_teste, y=predito_reg,
    labels={"x": "pIC50 observado", "y": "pIC50 previsto"},
    title="Regressao do pIC50: previsto x observado (R2 = " + str(round(r2, 3)) + ")",
    opacity=0.4)
figura_reg.update_traces(marker=dict(size=5, color="#3266ad"))
minimo = float(min(yr_teste.min(), predito_reg.min()))
maximo = float(max(yr_teste.max(), predito_reg.max()))
figura_reg.add_trace(go.Scatter(x=[minimo, maximo], y=[minimo, maximo],
                                mode="lines", line=dict(dash="dash", color="gray"),
                                name="ideal"))
figura_reg.add_hline(y=LIMIAR_POTENCIA, line_dash="dot", line_color="#c0392b")
figura_reg.add_vline(x=LIMIAR_POTENCIA, line_dash="dot", line_color="#c0392b")
figura_reg.show()

# a classificacao "derivada" da regressao, so para mostrar a ligacao
classe_derivada = (predito_reg >= LIMIAR_POTENCIA).astype(int)
classe_real = (yr_teste >= LIMIAR_POTENCIA).astype(int)
print("MCC da classificacao derivada da regressao:",
      round(matthews_corrcoef(classe_real, classe_derivada), 3))''')

mdq("""**Pergunta.** Por que as moléculas de limite (`>`), que na classificação
eram úteis como FRACO, tiveram de ser removidas da regressão?""",
"""**Resposta.** Porque a regressão prevê um **número exato** e precisa de um alvo
numérico exato para treinar. Uma medida "IC50 > 30000 nM" só diz "pelo menos tão
fraco quanto isto" — é um limite, não um ponto. Na classificação isso bastava
(qualquer valor acima do limite é FRACO), mas na regressão não há valor pontual
para ajustar. Mantê-las forçaria um número inventado e enviesaria o modelo. É por
isso que a mesma decisão de curadoria (Seção 2.4) leva a caminhos opostos nas duas
tarefas.""")


# ══════════════════════════════════════════════════════════════════════════
# SEÇÃO 6 — INTERPRETABILIDADE
# ══════════════════════════════════════════════════════════════════════════
md("""## Seção 6 — Interpretabilidade e o confundimento por tamanho

Um modelo pode acertar pelo motivo errado. Aqui investigamos uma hipótese
concreta: e se os modelos estiverem aprendendo majoritariamente **tamanho
molecular** em vez de química específica de reconhecimento? Potência bruta
correlaciona-se com número de átomos pesados — moléculas maiores fazem mais
contatos — e é por isso que a área usa eficiência de ligante em vez de potência
bruta. **Não afirmamos o resultado de antemão**: testamos.""")
code(r'''from scipy.stats import spearmanr

atomos_pesados = tabela_descritores["atomos_pesados"].values
correlacao, _ = spearmanr(atomos_pesados, agregados["pic50"].values)
figura_tamanho = px.scatter(
    x=atomos_pesados, y=agregados["pic50"].values,
    labels={"x": "atomos pesados", "y": "pIC50"},
    title="pIC50 x tamanho molecular (Spearman = " + str(round(correlacao, 3)) + ")",
    opacity=0.4)
figura_tamanho.update_traces(marker=dict(size=4, color="#3266ad"))
figura_tamanho.show()
print("correlacao de Spearman (tamanho x potencia):", round(correlacao, 3))''')

md("""### 6.1 — Modelo-controle de uma única feature

O teste decisivo: um classificador que usa **apenas** o número de átomos pesados.
Se o MCC dele for próximo ao dos modelos completos, então o fingerprint —
toda a química fina — está contribuindo pouco, e a conclusão da aula muda.""")
codex(r'''X_treino_tamanho = tabela_descritores.loc[idx_treino_esq, ["atomos_pesados"]].values
X_teste_tamanho = tabela_descritores.loc[idx_teste_esq, ["atomos_pesados"]].values

modelo_tamanho = LogisticRegression(max_iter=1000, random_state=SEMENTE)
modelo_tamanho.fit(X_treino_tamanho, y_treino)
predito_tamanho = modelo_tamanho.predict(X_teste_tamanho)
mcc_tamanho = matthews_corrcoef(y_teste, predito_tamanho)
mcc_completo = resultados["Floresta aleatoria"]["MCC"]
print("MCC so com tamanho     :", round(mcc_tamanho, 3))
print("MCC floresta completa  :", round(mcc_completo, 3))
print("diferenca              :", round(mcc_completo - mcc_tamanho, 3))''',
"""Treine uma LogisticRegression usando SÓ a coluna 'atomos_pesados' como feature.
Calcule o MCC no teste e compare com o MCC da floresta completa. Interprete: se
forem próximos, o fingerprint contribui pouco.""")

md("""### 6.2 — Importância por permutação

Qual feature o modelo realmente usa? A **importância por permutação** embaralha
uma feature de cada vez e mede quanto o desempenho cai: se cai muito, a feature
importa. Ela é preferível à importância por impureza da floresta, que é
**enviesada** a favor de variáveis com muitos valores distintos (como os
descritores contínuos) contra as binárias (os bits).

Permutar os 2048 bits **um a um** é caro e enganoso: cada bit isolado quase não
muda o resultado, então todos apareceriam como "sem importância" — e daria a
falsa impressão de que o fingerprint não conta. A saída correta é permutar o
fingerprint **como um bloco**: embaralhamos as linhas de todas as 2048 colunas de
uma vez (mantendo cada fingerprint interno intacto, mas quebrando sua associação
com o rótulo) e medimos a queda. Assim comparamos, na mesma escala, a importância
de **cada descritor** e a do **fingerprint inteiro**. Fazemos tudo com laços
explícitos.""")
code(r'''gerador_perm = np.random.RandomState(SEMENTE)
n_amostra_perm = min(400, len(X_teste))
amostra_perm = gerador_perm.choice(len(X_teste), n_amostra_perm, replace=False)
X_perm = X_teste[amostra_perm].copy()
y_perm = y_teste[amostra_perm]

# MCC de referencia (sem permutar nada)
mcc_referencia = matthews_corrcoef(y_perm, modelo_floresta.predict(X_perm))

# permuta cada coluna de descritor 3 vezes e mede a queda media de MCC
n_descritores = len(tabela_descritores.columns)
importancia_descritores = []
for indice_coluna in range(n_descritores):
    quedas = []
    for repeticao in range(3):
        X_embaralhado = X_perm.copy()
        coluna = X_embaralhado[:, indice_coluna].copy()
        gerador_perm.shuffle(coluna)
        X_embaralhado[:, indice_coluna] = coluna
        mcc_permutado = matthews_corrcoef(y_perm, modelo_floresta.predict(X_embaralhado))
        quedas.append(mcc_referencia - mcc_permutado)
    importancia_descritores.append(float(np.mean(quedas)))

# permuta o FINGERPRINT como um bloco: embaralha as linhas das colunas de bits
# juntas (cada fingerprint fica intacto, mas some a associacao com o rotulo)
colunas_fp = list(range(n_descritores, X_perm.shape[1]))
quedas_fp = []
for repeticao in range(3):
    X_embaralhado = X_perm.copy()
    ordem_linhas = gerador_perm.permutation(len(X_perm))
    X_embaralhado[:, colunas_fp] = X_perm[np.ix_(ordem_linhas, colunas_fp)]
    mcc_permutado = matthews_corrcoef(y_perm, modelo_floresta.predict(X_embaralhado))
    quedas_fp.append(mcc_referencia - mcc_permutado)
importancia_fp = float(np.mean(quedas_fp))

# junta os 9 descritores + o bloco do fingerprint na mesma escala
nomes = list(tabela_descritores.columns) + ["fingerprint (2048 bits)"]
valores = importancia_descritores + [importancia_fp]
cores = ["#7e9603"] * n_descritores + ["#3266ad"]     # fingerprint em azul
ordem_importancia = np.argsort(valores)
figura_importancia = go.Figure(go.Bar(
    x=[valores[i] for i in ordem_importancia],
    y=[nomes[i] for i in ordem_importancia],
    orientation="h", marker_color=[cores[i] for i in ordem_importancia]))
figura_importancia.update_layout(
    title="Importancia por permutacao: descritores x fingerprint (bloco)",
    height=400, xaxis_title="queda media de MCC ao embaralhar")
figura_importancia.show()
print("queda de MCC ao embaralhar o fingerprint inteiro:", round(importancia_fp, 3))
print("maior queda entre os descritores:", round(max(importancia_descritores), 3))''')

md("""### 6.3 — SHAP sobre a floresta

O SHAP atribui a cada feature, para cada molécula, o quanto ela empurrou a
predição para FORTE ou FRACO. O gráfico de resumo mostra o padrão geral; depois
explicamos duas moléculas individualmente, com o desenho ao lado. Restringimos a
uma amostra do teste porque o SHAP é custoso.""")
code(r'''import shap

n_amostra_shap = min(120, len(X_teste))
amostra_shap = gerador_perm.choice(len(X_teste), n_amostra_shap, replace=False)
X_shap = X_teste[amostra_shap]

# feature_perturbation e check_additivity=False evitam um erro de aditividade
# comum em florestas grandes com muitas features binarias
explicador = shap.TreeExplainer(modelo_floresta, feature_perturbation="tree_path_dependent")
valores_shap = explicador.shap_values(X_shap, check_additivity=False)
# em classificacao binaria, pegamos as contribuicoes para a classe FORTE (indice 1)
if isinstance(valores_shap, list):
    valores_shap_forte = valores_shap[1]
else:
    valores_shap_forte = valores_shap[:, :, 1]
shap.summary_plot(valores_shap_forte, X_shap, feature_names=nomes_features,
                  max_display=12, show=True)''')

md("""Duas moléculas explicadas individualmente: para cada uma, o desenho da
molécula ao lado das features que mais empurraram a predição para FORTE (verde) ou
para FRACO (vermelho). É a mesma informação do resumo, agora molécula a molécula.""")
code(r'''# explica duas moleculas (as posicoes 0 e 1 da amostra do SHAP), uma por vez
for posicao_no_shap in [0, 1]:
    indice_molecula = idx_teste_esq[amostra_shap[posicao_no_shap]]
    molecula = agregados.loc[indice_molecula, "molecula"]
    contribuicoes = valores_shap_forte[posicao_no_shap]

    # pega as 8 features de maior contribuicao (em modulo)
    ordem = np.argsort(np.abs(contribuicoes))[::-1][:8]
    nomes = []
    valores = []
    cores = []
    for i in ordem:
        nomes.append(nomes_features[i])
        valores.append(contribuicoes[i])
        cores.append("#1a7a4a" if contribuicoes[i] > 0 else "#c0392b")

    figura, (eixo_mol, eixo_shap) = plt.subplots(1, 2, figsize=(11, 3.6))
    eixo_mol.imshow(Draw.MolToImage(molecula, size=(320, 300)))
    eixo_mol.axis("off")
    eixo_mol.set_title(agregados.loc[indice_molecula, "molecule_chembl_id"], fontsize=10)
    eixo_shap.barh(range(len(valores)), valores[::-1], color=cores[::-1])
    eixo_shap.set_yticks(range(len(valores)))
    eixo_shap.set_yticklabels(nomes[::-1], fontsize=9)
    eixo_shap.axvline(0, color="black", linewidth=0.8)
    eixo_shap.set_title("contribuicao SHAP (verde=FORTE, vermelho=FRACO)", fontsize=9)
    plt.tight_layout()
    plt.show()''')

md("""### 6.4 — Dependência parcial

Como a predição varia quando movemos um descritor, mantendo os demais? As curvas
de dependência parcial de MW, LogP e TPSA mostram a direção do efeito aprendido —
por exemplo, se ganho de massa empurra para FORTE, é mais evidência de
confundimento por tamanho.""")
code(r'''indices_pdp = [nomes_features.index(nome) for nome in ["MW", "LogP", "TPSA"]]
figura_pdp, eixo_pdp = plt.subplots(1, 3, figsize=(12, 3.5))
PartialDependenceDisplay.from_estimator(
    modelo_floresta, X_treino, indices_pdp, feature_names=nomes_features, ax=eixo_pdp)
plt.tight_layout(); plt.show()''')

md("""### 6.5 — Controle negativo: rótulos embaralhados

O teste de sanidade final. Embaralhamos os rótulos do treino — destruindo
qualquer relação entre molécula e classe — e treinamos de novo. O desempenho
**deve** cair ao nível do acaso (MCC ≈ 0). Se **não** cair, há vazamento na
montagem dos dados, e o notebook precisa dizer isso claramente.""")
code(r'''gerador_embaralho = np.random.RandomState(SEMENTE)
y_treino_embaralhado = y_treino.copy()
gerador_embaralho.shuffle(y_treino_embaralhado)

floresta_controle = RandomForestClassifier(
    n_estimators=300, n_jobs=-1, random_state=SEMENTE)
floresta_controle.fit(X_treino, y_treino_embaralhado)
mcc_controle = matthews_corrcoef(y_teste, floresta_controle.predict(X_teste))
print("MCC com rotulos embaralhados:", round(mcc_controle, 3), "(esperado: proximo de 0)")
if abs(mcc_controle) > 0.15:
    print("ATENCAO: MCC longe de zero sugere vazamento de dados. Investigar.")
else:
    print("OK: desempenho caiu ao acaso, como deveria. Sem sinal de vazamento.")''')


# ══════════════════════════════════════════════════════════════════════════
# SEÇÃO 7 — DOMÍNIO DE APLICABILIDADE
# ══════════════════════════════════════════════════════════════════════════
md("""## Seção 7 — Domínio de aplicabilidade, revisitado

Um modelo só deveria opinar sobre moléculas parecidas com as que viu. Medimos,
para cada molécula de teste, a **similaridade de Tanimoto máxima** contra o
treino.

Antes, um esclarecimento que costuma confundir: **Morgan e Tanimoto não são duas
representações concorrentes.** O fingerprint de Morgan (Seção 3) é a
**representação** — o vetor de bits que descreve a molécula, e que serve de
*feature* para os modelos. A similaridade de Tanimoto é uma **medida** de quão
parecidos são **dois desses fingerprints** (interseção sobre união dos bits
ligados). Ou seja, calculamos Tanimoto **sobre** os mesmos fingerprints de
Morgan; um é a representação, o outro é a régua que compara duas representações.
Aqui a régua serve para o domínio de aplicabilidade, não para treinar.

O limiar que separa "dentro" de "fora" do domínio é derivado dos próprios dados —
o **percentil 5** das similaridades internas do treino — e não arbitrado. Uma
molécula menos conectada ao espaço químico do que 95% do treino é considerada
fora do domínio.""")
code(r'''# fingerprints de Morgan (objetos do RDKit) do treino e do teste, para o Tanimoto
fp_treino_rdkit = []
for indice in idx_treino_esq:
    molecula = agregados.loc[indice, "molecula"]
    fp_treino_rdkit.append(AllChem.GetMorganFingerprintAsBitVect(molecula, RAIO_MORGAN, nBits=N_BITS))
fp_teste_rdkit = []
for indice in idx_teste_esq:
    molecula = agregados.loc[indice, "molecula"]
    fp_teste_rdkit.append(AllChem.GetMorganFingerprintAsBitVect(molecula, RAIO_MORGAN, nBits=N_BITS))

# para cada molecula de teste, a MAIOR similaridade de Tanimoto contra o treino
similaridade_teste = []
for fp in fp_teste_rdkit:
    similaridades = DataStructs.BulkTanimotoSimilarity(fp, fp_treino_rdkit)
    similaridade_teste.append(max(similaridades))
similaridade_teste = np.array(similaridade_teste)

# limiar = percentil 5 das similaridades treino-treino (auto-similaridade excluida)
similaridade_treino_interna = []
for i in range(len(fp_treino_rdkit)):
    sims = DataStructs.BulkTanimotoSimilarity(fp_treino_rdkit[i], fp_treino_rdkit)
    sims[i] = -1.0                 # ignora a similaridade da molecula consigo mesma
    similaridade_treino_interna.append(max(sims))
LIMIAR_DOMINIO = float(np.percentile(similaridade_treino_interna, 5))
print("limiar de dominio (percentil 5 do treino):", round(LIMIAR_DOMINIO, 3))''')
code(r'''figura_dominio = px.histogram(
    x=similaridade_teste, nbins=40,
    labels={"x": "Tanimoto maxima ao treino"},
    title="Similaridade das moleculas de teste ao treino")
figura_dominio.add_vline(x=LIMIAR_DOMINIO, line_dash="dash", line_color="#c0392b",
                         annotation_text="limiar de dominio")
figura_dominio.update_traces(marker_color="#3266ad")
figura_dominio.show()''')

md("""### 7.1 — Desempenho dentro e fora do domínio, com cuidado

Comparamos o desempenho nas moléculas **dentro** e **fora** do domínio — mas com
uma advertência: comparar acurácias entre grupos de tamanhos e composições de
classe diferentes **engana**. Por isso imprimimos o número de amostras e a
proporção de FORTE em cada grupo, e olhamos o MCC, não a acurácia crua. Foi
exatamente ignorar isso que, num teste preliminar com dados sintéticos, produziu
a observação contraintuitiva de "melhor fora do que dentro".""")
code(r'''dentro = similaridade_teste >= LIMIAR_DOMINIO
fora = ~dentro
predito_floresta = resultados["Floresta aleatoria"]["predito"]

# para cada grupo (dentro/fora), imprime n, composicao de classe e MCC
print("Comparacao estratificada (floresta aleatoria):")
for nome_grupo, mascara in [("dentro dom.", dentro), ("fora dom.", fora)]:
    n = int(mascara.sum())
    if n == 0:
        print(nome_grupo, "-> grupo vazio")
        continue
    frac_forte = float(y_teste[mascara].mean())
    if len(np.unique(y_teste[mascara])) < 2:
        mcc = float("nan")            # so uma classe no grupo: MCC indefinido
    else:
        mcc = matthews_corrcoef(y_teste[mascara], predito_floresta[mascara])
    print(f"{nome_grupo:14s} n={n:4d}  fracao FORTE={frac_forte:.2f}  MCC={mcc:.3f}")
print("\\nObservacao: se os grupos tem composicao de classe muito diferente,")
print("a comparacao direta de desempenho e enganosa. Olhe n e fracao FORTE antes de concluir.")''')


# ══════════════════════════════════════════════════════════════════════════
# SEÇÃO 8 — CONFORMAL (OPCIONAL)
# ══════════════════════════════════════════════════════════════════════════
md("""## Seção 8 — Predição conformal (opcional)

*Esta seção é opcional: pode ser pulada sem quebrar o resto do notebook.*

A predição conformal transforma probabilidades em **conjuntos de predição** com
uma garantia de cobertura. Usamos a variante **indutiva de Mondrian** (uma
calibração por classe) sobre o conjunto de calibração separado na Seção 4. Para
cada molécula de teste, o conjunto de predição pode ser: `{FORTE}`, `{FRACO}`,
`{ambos}` (ambíguo) ou `{}` (vazio, atípico).""")
code(r'''NIVEL_CONFIANCA = 0.80   # cobertura pretendida: 80%
alfa = 1.0 - NIVEL_CONFIANCA

# escores de nao-conformidade = 1 - probabilidade da classe verdadeira, na calibracao
proba_calib = modelo_floresta.predict_proba(X_calib)
escores_por_classe = {0: [], 1: []}
for posicao in range(len(y_calib)):
    classe_real = y_calib[posicao]
    escore = 1.0 - proba_calib[posicao, classe_real]
    escores_por_classe[classe_real].append(escore)

# limiar de Mondrian: quantil (1-alfa) dos escores de cada classe separadamente
limiar_conformal = {}
for classe in (0, 1):
    limiar_conformal[classe] = float(np.quantile(escores_por_classe[classe], 1.0 - alfa))
print("limiares conformais por classe:", {k: round(v, 3) for k, v in limiar_conformal.items()})

# aplica ao teste: para cada molecula, monta o conjunto de predicao e checa cobertura
proba_teste_rf = modelo_floresta.predict_proba(X_teste)
conjuntos = []
cobertos = 0
for posicao in range(len(y_teste)):
    # o conjunto tem cada classe cujo escore de nao-conformidade cabe no limiar dela
    conjunto = []
    for classe in (0, 1):
        escore = 1.0 - proba_teste_rf[posicao, classe]
        if escore <= limiar_conformal[classe]:
            conjunto.append(classe)
    conjuntos.append(conjunto)
    if y_teste[posicao] in conjunto:
        cobertos = cobertos + 1
cobertura_empirica = cobertos / len(y_teste)
print("cobertura pretendida:", NIVEL_CONFIANCA, "| cobertura empirica:", round(cobertura_empirica, 3))

# distribuicao dos quatro tipos de conjunto
from collections import Counter
tipos = Counter()
for conjunto in conjuntos:
    if len(conjunto) == 2:   tipos["ambos (ambiguo)"] += 1
    elif len(conjunto) == 0: tipos["vazio (atipico)"] += 1
    elif conjunto[0] == 1:   tipos["so FORTE"] += 1
    else:                    tipos["so FRACO"] += 1
print("tipos de conjunto:", dict(tipos))''')

mdq("""**Pergunta.** O que a garantia conformal de 80% significa — e o que ela
**não** significa?""",
"""**Resposta.** Significa que, em média, o conjunto de predição contém a classe
verdadeira em cerca de 80% das moléculas (cobertura marginal). **Não** significa
que cada predição individual tem 80% de chance de estar certa, nem que uma
molécula com conjunto `{FORTE}` é forte com 80% de probabilidade. É uma garantia
sobre a taxa de acerto do procedimento no agregado, não sobre uma molécula
específica.""")


# ══════════════════════════════════════════════════════════════════════════
# SEÇÃO 9 — O CLASSIFICADOR EM USO
# ══════════════════════════════════════════════════════════════════════════
md("""## Seção 9 — O classificador em uso

Até aqui o código foi **procedural** — cada célula mostra os passos, sem funções
escondendo a lógica. Agora, porém, uma função **se justifica**: `classificar` é a
**ferramenta** que queremos reutilizar em qualquer molécula nova, e reutilização é
exatamente para o que servem as funções. É o único caso deste tipo na aula (fora a
classe da rede em PyTorch e o widget, que os frameworks exigem).

A função `classificar(smiles)` devolve **FORTE**, **FRACO** ou **INDEFINIDA**,
sempre com o motivo. A ordem importa: primeiro verificamos o **domínio de
aplicabilidade**; só então interpretamos a probabilidade. Uma molécula fora do
domínio é INDEFINIDA por atipicidade, independentemente do que o modelo "acharia".
A função funciona mesmo se a Seção 8 for pulada.""")
code(r'''MARGEM_ABSTENCAO = 0.15   # se |proba - 0.5| < margem, o modelo se abstem (INDEFINIDA)

def classificar(smiles):
    """Classifica uma molecula (SMILES) como FORTE, FRACO ou INDEFINIDA, com motivo."""
    molecula = Chem.MolFromSmiles(smiles)
    if molecula is None:
        return {"classe": "INDEFINIDA", "motivo": "SMILES invalido",
                "prob_forte": None, "similaridade": None}

    # 1. dominio de aplicabilidade primeiro
    fp = AllChem.GetMorganFingerprintAsBitVect(molecula, RAIO_MORGAN, nBits=N_BITS)
    similaridade = max(DataStructs.BulkTanimotoSimilarity(fp, fp_treino_rdkit))
    if similaridade < LIMIAR_DOMINIO:
        return {"classe": "INDEFINIDA",
                "motivo": "fora do dominio (Tanimoto max %.2f < %.2f)" % (similaridade, LIMIAR_DOMINIO),
                "prob_forte": None, "similaridade": similaridade}

    # 2. so agora a probabilidade do modelo. Monta o mesmo vetor de features do treino:
    #    os 9 descritores (na ordem das colunas) seguidos dos 2048 bits do fingerprint.
    vetor_desc = np.array([
        Descriptors.MolWt(molecula), Descriptors.MolLogP(molecula),
        Descriptors.TPSA(molecula), Descriptors.NumHDonors(molecula),
        Descriptors.NumHAcceptors(molecula), Descriptors.NumRotatableBonds(molecula),
        Descriptors.NumAromaticRings(molecula), Descriptors.FractionCSP3(molecula),
        molecula.GetNumHeavyAtoms(),
    ], dtype=float)
    vetor_fp = np.zeros((N_BITS,), dtype=float)
    DataStructs.ConvertToNumpyArray(fp, vetor_fp)
    entrada = np.hstack([vetor_desc, vetor_fp]).reshape(1, -1)
    probabilidade_forte = float(modelo_floresta.predict_proba(entrada)[0, 1])

    # 3. abstencao por ambiguidade estatistica
    if abs(probabilidade_forte - 0.5) < MARGEM_ABSTENCAO:
        return {"classe": "INDEFINIDA",
                "motivo": "ambiguo (probabilidade FORTE = %.2f)" % probabilidade_forte,
                "prob_forte": probabilidade_forte, "similaridade": similaridade}
    if probabilidade_forte >= 0.5:
        return {"classe": "FORTE", "motivo": "probabilidade FORTE = %.2f" % probabilidade_forte,
                "prob_forte": probabilidade_forte, "similaridade": similaridade}
    return {"classe": "FRACO", "motivo": "probabilidade FORTE = %.2f" % probabilidade_forte,
            "prob_forte": probabilidade_forte, "similaridade": similaridade}

# teste rapido com uma molecula do proprio conjunto
exemplo_smiles = agregados.loc[idx_teste_esq[0], "canonical_smiles"]
print(exemplo_smiles, "->", classificar(exemplo_smiles))''')

md("""### 9.1 — Galeria de teste

Testamos com uma grade de moléculas desenhadas pelo RDKit, com a resposta do
modelo como legenda: inibidores conhecidos da acetilcolinesterase (donepezila,
tacrina, galantamina), fármacos sem relação com o alvo (aspirina, paracetamol) e
moléculas cotidianas (cafeína, etanol).

Espere respostas de três naturezas, e **leia os motivos impressos**, não só as
classes: moléculas muito diferentes de tudo que o modelo viu (como o etanol) caem
em INDEFINIDA por atipicidade; um inibidor conhecido de esqueleto familiar tende a
FORTE; e há um caso instrutivo — uma molécula "cotidiana" pode, ainda assim,
**estar dentro do domínio** se sua estrutura se parece com algo do treino, e então
receber uma resposta confiante. Não presuma o resultado: observe o que o modelo de
fato responde e por quê.""")
code(r'''galeria = [
    ("donepezila",  "O=C1CC2(CCN(Cc3ccccc3)CC2)Cc2cc(OC)c(OC)cc21"),
    ("tacrina",     "Nc1c2c(nc3ccccc13)CCCC2"),
    ("galantamina", "CN1CCC23C=CC(O)CC2Oc2c(OC)ccc(c23)C1"),
    ("aspirina",    "CC(=O)Oc1ccccc1C(=O)O"),
    ("paracetamol", "CC(=O)Nc1ccc(O)cc1"),
    ("cafeina",     "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
    ("etanol",      "CCO"),
]
moleculas_galeria = []
legendas_galeria = []
for nome, smiles in galeria:
    resposta = classificar(smiles)
    moleculas_galeria.append(Chem.MolFromSmiles(smiles))
    legendas_galeria.append(nome + ": " + resposta["classe"])
    print(f"{nome:14s} -> {resposta['classe']:8s} ({resposta['motivo']})")

Draw.MolsToGridImage(moleculas_galeria, legends=legendas_galeria,
                     molsPerRow=4, subImgSize=(230, 180))''')

mdq("""**Pergunta.** Olhe os motivos impressos para a cafeína e o etanol. Um deles
caiu em INDEFINIDA por estar fora do domínio; o outro recebeu uma classe com
probabilidade confiante. O que isso revela sobre a diferença entre "irrelevante
para um biólogo" e "fora do domínio do modelo"?""",
"""**Resposta.** O **etanol** cai em INDEFINIDA por atipicidade — é tão pequeno e
diferente do treino que o modelo, corretamente, se abstém. A **cafeína**, porém,
fica **dentro do domínio**: seu anel purínico/xantínico se parece com esqueletos
presentes no conjunto (xantinas já foram estudadas contra a acetilcolinesterase),
então o modelo tem base para opinar — e responde FRACO com alta confiança (~0,06
de probabilidade de FORTE), o que é **quimicamente correto**: cafeína não é um
inibidor potente. A lição: "irrelevante" é uma intuição biológica; "fora do
domínio" é uma medida estrutural. Elas nem sempre coincidem — e um FRACO confiante
e correto é uma resposta tão honesta quanto um INDEFINIDA. O modelo só deve abster-se
quando de fato não tem base, não sempre que a molécula parece incomum aos nossos
olhos.""")

md("""### 9.2 — Triagem virtual: procurar inibidores entre fármacos aprovados

Agora o pagamento de verdade da aula. Pegamos uma **biblioteca real** de fármacos
aprovados no mundo (o conjunto `world` do ZINC15, 512 moléculas) e passamos **cada
uma** pelo `classificar`. A pergunta prática: **existe, entre remédios que já são
usados para outras coisas, algum candidato a inibidor da acetilcolinesterase?** É
exatamente o raciocínio do **reposicionamento de fármacos** — e é para isso que a
`classificar` foi feita.

Carregamos o arquivo direto do repositório (`data/world.smi`), no formato
`SMILES  ID_ZINC` por linha.""")
code(r'''URL_WORLD = ("https://raw.githubusercontent.com/monteirotorres/ml/"
             "main/data/world.smi")
try:
    tabela_world = pd.read_csv(URL_WORLD, sep=r"\s+", header=None, names=["smiles", "zinc_id"])
    print("biblioteca lida da URL:", len(tabela_world), "farmacos aprovados")
except Exception as erro:
    print("URL indisponivel (", type(erro).__name__, "); tentando arquivo local")
    tabela_world = pd.read_csv("world.smi", sep=r"\s+", header=None, names=["smiles", "zinc_id"])
    print("biblioteca lida do arquivo local:", len(tabela_world))
tabela_world.head()''')

md("""Rodamos o classificador em toda a biblioteca. Como a imensa maioria dos
fármacos aprovados **não** tem nada a ver com a acetilcolinesterase, o esperado é
um mar de FRACO e INDEFINIDA e **poucos** FORTE — e são justamente esses poucos que
interessam. Guardamos a probabilidade de FORTE para ordenar os candidatos depois.""")
code(r'''# passa cada farmaco pela ferramenta da Secao 9 (reuso, nao reimplementacao)
linhas_triagem = []
for posicao in range(len(tabela_world)):
    smiles = tabela_world.iloc[posicao]["smiles"]
    zinc_id = tabela_world.iloc[posicao]["zinc_id"]
    resposta = classificar(smiles)
    linhas_triagem.append({
        "zinc_id": zinc_id, "smiles": smiles,
        "classe": resposta["classe"], "prob_forte": resposta["prob_forte"],
    })
triagem = pd.DataFrame(linhas_triagem)

print("distribuicao das", len(triagem), "moleculas aprovadas:")
contagem_triagem = triagem["classe"].value_counts()
for nome_classe in ["FORTE", "FRACO", "INDEFINIDA"]:
    n = int(contagem_triagem.get(nome_classe, 0))
    print(f"  {nome_classe:11s}: {n:4d}  ({round(100 * n / len(triagem), 1)}%)")''')

md("""### 9.2a — Sanidade: recuperamos os inibidores já conhecidos?

Antes de acreditar num candidato novo, uma pergunta de **controle**: dos fármacos
desta biblioteca que **já têm medida de IC50 no ChEMBL** contra a acetilcolinesterase
(ou seja, cuja potência real conhecemos), o modelo acerta? Achamos essa interseção
casando o **esqueleto InChIKey** (robusto a sal/estereo/protonação) e comparamos a
classe do modelo com o rótulo verdadeiro.

**Ressalva honesta e importante:** alguns desses conhecidos estavam no **treino** do
modelo — acertá-los ali é em parte memorização, não generalização. Por isso separamos
os que estavam no treino dos que **não** estavam (o teste honesto), e olhamos os dois
grupos.""")
code(r'''# mapa: esqueleto-InChIKey -> (rotulo verdadeiro, estava no treino do modelo?)
def esqueleto_inchikey(molecula):
    return Chem.MolToInchiKey(molecula).split("-")[0]

conjunto_treino = set(idx_treino_esq)
verdade_por_esqueleto = {}
for indice in agregados.index:
    chave = esqueleto_inchikey(agregados.loc[indice, "molecula"])
    verdade_por_esqueleto[chave] = {
        "rotulo": int(agregados.loc[indice, "rotulo"]),
        "no_treino": indice in conjunto_treino,
    }

# quais farmacos da biblioteca tambem estao no ChEMBL AChE (esqueleto conhecido)
linhas_conhecidas = []
for posicao in range(len(triagem)):
    smiles = triagem.iloc[posicao]["smiles"]
    molecula = Chem.MolFromSmiles(smiles)
    if molecula is None:
        continue
    chave = esqueleto_inchikey(molecula)
    if chave in verdade_por_esqueleto:
        verdade = verdade_por_esqueleto[chave]
        linhas_conhecidas.append({
            "zinc_id": triagem.iloc[posicao]["zinc_id"],
            "classe_modelo": triagem.iloc[posicao]["classe"],
            "verdade": "FORTE" if verdade["rotulo"] == 1 else "FRACO",
            "no_treino": verdade["no_treino"],
        })
conhecidas = pd.DataFrame(linhas_conhecidas)
print("farmacos aprovados que TAMBEM tem medida no ChEMBL AChE:", len(conhecidas))

# entre os que a VERDADE diz FORTE: quantos o modelo recupera como FORTE?
verdadeiros_fortes = conhecidas[conhecidas["verdade"] == "FORTE"]
print("\ndestes, com potencia real FORTE:", len(verdadeiros_fortes))
for rotulo_grupo, grupo in [("no treino (memoria)", verdadeiros_fortes[verdadeiros_fortes["no_treino"]]),
                            ("fora do treino (teste honesto)", verdadeiros_fortes[~verdadeiros_fortes["no_treino"]])]:
    if len(grupo) == 0:
        print(f"  {rotulo_grupo}: nenhum")
        continue
    n_forte = int((grupo["classe_modelo"] == "FORTE").sum())
    n_fraco = int((grupo["classe_modelo"] == "FRACO").sum())
    n_indef = int((grupo["classe_modelo"] == "INDEFINIDA").sum())
    print(f"  {rotulo_grupo}: {len(grupo)} conhecidos -> "
          f"FORTE {n_forte} | FRACO {n_fraco} (falsos negativos) | INDEFINIDA {n_indef}")''')

md("""### 9.2b — Os candidatos novos

Agora o que a triagem tem de mais interessante: fármacos classificados **FORTE**
cujo esqueleto **não** aparece no ChEMBL AChE — ou seja, moléculas para as quais o
modelo aposta em atividade **sem** ter visto nada igual medido contra o alvo. Cada
uma é uma **hipótese de reposicionamento** a levar para a bancada, não uma verdade:
o modelo dá a pista, o experimento decide. Ordenamos pela probabilidade de FORTE e
desenhamos os primeiros.""")
code(r'''esqueletos_conhecidos = set(verdade_por_esqueleto.keys())

candidatas_novas = []
for posicao in range(len(triagem)):
    if triagem.iloc[posicao]["classe"] != "FORTE":
        continue
    smiles = triagem.iloc[posicao]["smiles"]
    molecula = Chem.MolFromSmiles(smiles)
    if molecula is None:
        continue
    if esqueleto_inchikey(molecula) in esqueletos_conhecidos:
        continue                                   # ja conhecida, nao e novidade
    candidatas_novas.append((triagem.iloc[posicao]["zinc_id"], smiles,
                             float(triagem.iloc[posicao]["prob_forte"])))

candidatas_novas.sort(key=lambda t: -t[2])         # maior probabilidade primeiro
print("candidatas NOVAS classificadas FORTE (esqueleto inedito):", len(candidatas_novas))

n_mostrar = min(8, len(candidatas_novas))
moleculas_cand = []
legendas_cand = []
for zinc_id, smiles, prob in candidatas_novas[:n_mostrar]:
    moleculas_cand.append(Chem.MolFromSmiles(smiles))
    legendas_cand.append(zinc_id + "  p=" + str(round(prob, 2)))
    print(f"  {zinc_id}  prob_forte={round(prob, 2)}  {smiles}")

if n_mostrar > 0:
    imagem_candidatas = Draw.MolsToGridImage(moleculas_cand, legends=legendas_cand,
                                             molsPerRow=4, subImgSize=(230, 180))
else:
    imagem_candidatas = None
imagem_candidatas''')

mdq("""**Pergunta.** Um fármaco aprovado classificado FORTE aqui é um inibidor da
acetilcolinesterase? E se um inibidor conhecido tivesse saído FRACO, o que isso
significaria?""",
"""**Resposta.** Não — um FORTE aqui é uma **hipótese**, não um veredito. O modelo
diz "esta molécula se parece, na representação que ele usa, com inibidores potentes
que ele viu"; confirmar exige o ensaio de IC50 na bancada. Muitos FORTE serão
falsos positivos, e tudo bem: a triagem virtual serve para **priorizar** o que
testar, não para substituir o teste. Já um inibidor conhecido classificado FRACO
seria um **falso negativo** — e o mais custoso dos erros numa triagem, porque
descartaria em silêncio um composto que funciona. É por isso que a Seção 9.2a
mede exatamente isso (recuperação dos conhecidos e falsos negativos): a taxa de
falsos negativos, sobretudo **fora do treino**, é o que diz se podemos confiar na
triagem. E note o papel do **domínio**: candidatos bons demais para serem verdade
muitas vezes caem em INDEFINIDA por atipicidade — a abstenção protegendo você de
apostar onde o modelo não tem base.""")


# ══════════════════════════════════════════════════════════════════════════
# SEÇÃO 10 — PERSISTÊNCIA E EXERCÍCIOS
# ══════════════════════════════════════════════════════════════════════════
md("""## Seção 10 — Persistência e exercícios

Salvamos tudo que a função `classificar` precisa para rodar amanhã sem refazer o
treino: o modelo, os fingerprints do treino (para o domínio), os limiares e um
**dicionário de versões**. As versões importam porque um modelo salvo com uma
versão de scikit-learn ou RDKit pode não recarregar corretamente em outra — sem
registrar isso, um modelo "que funcionava" vira irreproduzível.""")
code(r'''pacote = {
    "modelo_floresta": modelo_floresta,
    "fp_treino_rdkit": fp_treino_rdkit,
    "limiar_dominio": LIMIAR_DOMINIO,
    "limiar_potencia": LIMIAR_POTENCIA,
    "margem_abstencao": MARGEM_ABSTENCAO,
    "raio_morgan": RAIO_MORGAN,
    "n_bits": N_BITS,
    "nomes_descritores": list(tabela_descritores.columns),
    "versoes": {
        "numpy": np.__version__, "pandas": pd.__version__,
        "sklearn": sklearn.__version__, "torch": torch.__version__,
    },
}
joblib.dump(pacote, "classificador_ache.joblib")
print("salvo: classificador_ache.joblib")
print("versoes registradas:", pacote["versoes"])''')

md("""### Exercícios

1. **Limiar de potência.** Reexecute a partir da Seção 3 com `LIMIAR_POTENCIA =
   7.0` (mais exigente). Como mudam o balanço de classes e o MCC dos modelos?
2. **Remover as medidas de limite.** Refaça a curadoria descartando as medidas
   `>`. O que acontece com a classe FRACO e com a revocação dela? Por quê?
3. **Comparar representações.** Troque o fingerprint de Morgan por apenas os nove
   descritores físico-químicos. Quanto se perde? E só com o fingerprint, sem os
   descritores?
4. **Taxa de aprendizado.** Na rede em PyTorch, encontre uma taxa alta demais e
   descreva a assinatura visual na curva de perda.
5. **Domínio de aplicabilidade.** Varie o percentil do limiar de domínio (de 1 a
   20). Como muda a fração de moléculas classificadas como INDEFINIDA?
6. **Escrita.** Um usuário recebe FORTE do seu modelo para uma molécula nova. O
   que ele **pode** e o que ele **não pode** concluir a partir dessa resposta?
   (Pense em domínio de aplicabilidade, calibração e na diferença entre potência
   prevista e atividade confirmada em bancada.)
7. **Confundimento.** Com base na Seção 6, o modelo aprende química específica ou
   majoritariamente tamanho? Que evidência do notebook sustenta sua resposta?""")

md("""### Controle interativo (célula de Colab)

Um controle deslizante para o limiar de abstenção que atualiza ao vivo a fração de
moléculas classificadas como INDEFINIDA no teste. A interatividade muda o que se
**entende**, não só o que se vê: dá para sentir o compromisso entre abster-se mais
(mais INDEFINIDA, menos erros declarados) e decidir mais. O estado dos controles
**não** é salvo no arquivo — é preciso reexecutar a célula.""")
code(r'''try:
    from ipywidgets import interact, FloatSlider

    proba_teste_forte = modelo_floresta.predict_proba(X_teste)[:, 1]

    def mostrar_incerto(margem):
        """Recalcula e imprime a fracao de INDEFINIDA para uma margem de abstencao."""
        ambiguo = np.abs(proba_teste_forte - 0.5) < margem
        fora_dom = similaridade_teste < LIMIAR_DOMINIO
        incerto = ambiguo | fora_dom
        print("margem =", round(margem, 2),
              "| fracao INDEFINIDA =", round(incerto.mean(), 3),
              "| por ambiguidade =", round(ambiguo.mean(), 3),
              "| fora do dominio =", round(fora_dom.mean(), 3))

    interact(mostrar_incerto,
             margem=FloatSlider(min=0.0, max=0.45, step=0.05, value=0.15))
except Exception as erro:
    print("ipywidgets so e interativo no Colab/Jupyter:", type(erro).__name__)''')

md("""---

Fim da aula. Você extraiu dados reais, curou-os com prestação de contas,
comparou partições honestas e desonestas, treinou e comparou quatro modelos sob
a mesma interface (e ainda previu o pIC50 contínuo por regressão), abriu a caixa
preta de uma rede em PyTorch, investigou se o modelo aprende química ou tamanho,
controlou o vazamento de dados de frente, delimitou onde o modelo pode opinar e o
empacotou em uma função honesta que sabe **se abster**. O mais importante que
fica: um modelo bom não é o que sempre responde, é o que sabe quando não deveria
responder.""")


# ══════════════════════════════════════════════════════════════════════════
# CONSTRUÇÃO DOS DOIS NOTEBOOKS
# ══════════════════════════════════════════════════════════════════════════
def construir(versao):
    """versao='gabarito' (completo) ou 'aluno' (modelagem 5-7 e respostas removidas)."""
    nb = nbf.v4.new_notebook()
    celulas = []
    for item in CELULAS:
        tipo = item[0]
        if tipo == "md":
            celulas.append(nbf.v4.new_markdown_cell(item[1]))
        elif tipo == "mdq":
            pergunta, resposta = item[1], item[2]
            if versao == "gabarito":
                celulas.append(nbf.v4.new_markdown_cell(pergunta + "\n\n" + resposta))
            else:
                celulas.append(nbf.v4.new_markdown_cell(
                    pergunta + "\n\n> _Escreva sua interpretacao aqui antes de conferir o gabarito._"))
        elif tipo == "code":
            celulas.append(nbf.v4.new_code_cell(item[1]))
        elif tipo == "codex":
            src, instrucao = item[1], item[2]
            if versao == "gabarito":
                celulas.append(nbf.v4.new_code_cell(src))
            else:
                stub = ("# COMPLETE ESTA CELULA\n# " +
                        instrucao.strip().replace("\n", "\n# ") +
                        "\n\n# seu codigo aqui\n")
                celulas.append(nbf.v4.new_code_cell(stub))
    nb.cells = celulas
    nb.metadata["language_info"] = {"name": "python"}
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nome = "aula_gabarito.ipynb" if versao == "gabarito" else "aula_aluno.ipynb"
    (PASTA / nome).write_text(nbf.writes(nb), encoding="utf-8")
    print("gerado:", nome, "|", len(celulas), "celulas")


if __name__ == "__main__":
    construir("gabarito")
    construir("aluno")
    print("total de itens de celula na fonte:", len(CELULAS))

