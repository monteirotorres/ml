# ML aplicado a inibidores (ChEMBL)

Uma aula prática de aprendizagem de máquina aplicada à **bioinformática e à
descoberta de fármacos**. A partir de dados reais de inibidores de um alvo
proteico extraídos do [ChEMBL](https://www.ebi.ac.uk/chembl/), treinamos modelos
que recebem qualquer molécula (via SMILES) e a classificam como **FORTE**,
**FRACO** ou **INCERTO** para aquele alvo.

O material foi escrito para alunos de graduação em ciências biológicas e
biomédicas que já conhecem Python básico, `numpy`, `pandas` e `matplotlib`, mas
**nunca treinaram um modelo**. Roda inteiro no Google Colab, em CPU.

**Abra no Colab:**
[![Gabarito](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/monteirotorres/ml/blob/main/08_aula_bioinfo/aula_gabarito.ipynb)
&nbsp; (gabarito completo) &nbsp;·&nbsp;
[![Aluno](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/monteirotorres/ml/blob/main/08_aula_bioinfo/aula_aluno.ipynb)
&nbsp; (versão do aluno)

## O alvo

O notebook usa a **acetilcolinesterase** (AChE, `CHEMBL220`) — enzima que degrada
o neurotransmissor acetilcolina. É um alvo de enorme relevância: seus inibidores
incluem fármacos contra o Alzheimer (donepezila, rivastigmina) e também
inseticidas organofosforados e agentes neurotóxicos. O ChEMBL reúne milhares de
moléculas com potência (IC50) medida contra ela.

O identificador do alvo é resolvido pela API do ChEMBL dentro do notebook, então
trocar de alvo é uma mudança de uma linha.

## Por que "INCERTO" não é uma classe dos dados

Esta é a decisão de projeto central da aula. O ChEMBL registra **potência
medida**, não incerteza — não existe uma coluna "incerto". Por isso:

- o rótulo de treino é **binário** (FORTE/FRACO, por um limiar de pIC50);
- a resposta **INCERTO** é produzida pelo próprio modelo, por duas vias
  independentes: (a) **ambiguidade estatística** (o modelo não tem confiança
  suficiente) e (b) molécula **fora do domínio de aplicabilidade** químico (o
  modelo nunca viu nada parecido e não deveria opinar).

Reconhecer o que o modelo **não sabe** é tão importante quanto a predição em si —
uma lição que vale muito além da quimioinformática.

## O que a aula cobre

| Seção | Conteúdo |
| --- | --- |
| 0–1 | Ambiente reprodutível; o que é IC50, pIC50 e por que a escala é logarítmica |
| 2 | Curadoria dos dados com tabela de proveniência e gráfico de funil |
| 3 | Descritores moleculares e fingerprint de Morgan, explicados um a um |
| 4 | Partição aleatória × por esqueleto (Bemis-Murcko), com projeção 2D do espaço químico |
| 5 | Quatro modelos (regressão logística, SVM, floresta aleatória, rede neural) + a mesma rede aberta em PyTorch |
| 6 | Interpretabilidade (importância por permutação, SHAP) e o confundimento por tamanho molecular |
| 7 | Domínio de aplicabilidade por similaridade de Tanimoto |
| 8 | Predição conformal (opcional) |
| 9 | A função `classificar(smiles)` em uso, com galeria de moléculas |
| 10 | Persistência do modelo e exercícios |

## Como usar

O notebook acompanha o material em duas versões: um **gabarito** completo e
executado, e uma versão **do aluno** com as células de modelagem esvaziadas para
preencher em sala. Consulte o `README.md` da pasta para o roteiro do docente
(o que rodar antes da aula, o que pode ser cortado se o tempo apertar, e o tempo
estimado por seção).

> **Nota.** Este é um material extenso, voltado ao Colab. Partes específicas do
> Colab — o painel do TensorBoard ao vivo, as mágicas `%tensorboard`/`%pip` e os
> controles interativos `ipywidgets` — só se comportam plenamente no Colab; toda
> curva mostrada no TensorBoard é também replicada em Plotly como registro
> permanente no arquivo.
