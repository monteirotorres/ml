# ML aplicado a inibidores (ChEMBL)

Uma aula prática de aprendizagem de máquina aplicada à **bioinformática e à
descoberta de fármacos**. A partir de dados reais de inibidores de um alvo
proteico extraídos do [ChEMBL](https://www.ebi.ac.uk/chembl/), treinamos modelos
que recebem qualquer molécula (via SMILES) e a classificam como **FORTE** ou
**FRACO** para aquele alvo — ou se **abstêm** (classe **INDEFINIDA**) quando não
têm base para decidir.

O material foi escrito para alunos de graduação em ciências biológicas e
biomédicas que já conhecem Python básico, `numpy`, `pandas` e `matplotlib`, mas
**nunca treinaram um modelo**. Roda inteiro no Google Colab, em CPU.

**Abra no Colab:**
[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/monteirotorres/ml/blob/main/08_aula_bioinfo/aula.ipynb)

## O alvo

O notebook usa a **acetilcolinesterase** (AChE, `CHEMBL220`) — enzima que degrada
o neurotransmissor acetilcolina. É um alvo de enorme relevância: seus inibidores
incluem fármacos contra o Alzheimer (donepezila, rivastigmina) e também
inseticidas organofosforados e agentes neurotóxicos. O ChEMBL reúne milhares de
moléculas com potência (IC50) medida contra ela.

O identificador do alvo é resolvido pela API do ChEMBL dentro do notebook, então
trocar de alvo é uma mudança de uma linha.

## Treino binário, resposta com abstenção

Esta é a decisão de projeto central da aula. O modelo é treinado só para separar
**FORTE** de **FRACO** (o rótulo, por um limiar de pIC50). Mas, na hora de
responder, ele pode **se abster** — devolver a classe **INDEFINIDA** — por duas
razões distintas:

- **abstenção por ambiguidade**: a probabilidade fica perto do meio, sem
  evidência clara para nenhum dos lados;
- **fora do domínio de aplicabilidade**: a molécula não se parece com nada que o
  modelo viu, então ele não tem base para opinar.

Saber **quando não decidir** é tão importante quanto a predição em si — uma lição
que vale muito além da quimioinformática.

## O que a aula cobre

| Seção | Conteúdo |
| --- | --- |
| 0–1 | Ambiente reprodutível; o que é IC50, pIC50 e por que a escala é logarítmica |
| 2 | Curadoria dos dados com tabela de proveniência e gráfico de funil |
| 3 | Descritores moleculares e fingerprint de Morgan, explicados um a um |
| 4 | Partição (aleatória × esqueleto de Bemis-Murcko) e projeção 2D do espaço químico dos esqueletos |
| 5 | Modelos de classificação (logística, SVM, floresta e MLP) + a mesma rede aberta em PyTorch, com comparação |
| 6 | Interpretabilidade: confundimento por tamanho, importância por permutação e SHAP |
| 7 | Domínio de aplicabilidade por similaridade de Tanimoto |
| 9 | A função `classificar(smiles)` em uso, com galeria de moléculas e triagem virtual |
| 10 | Persistência do modelo |

## Como usar

O material é um único notebook completo, com as células de código prontas (cada
uma marcada com `# Célula NN` para referência em sala). Consulte o `README.md` da
pasta para o roteiro do docente (o que rodar antes da aula, o que pode ser cortado
se o tempo apertar, e o tempo estimado por seção).

> **Nota.** Este é um material extenso, voltado ao Colab. Alguns recursos — as
> mágicas `%pip` e a renderização inline do Plotly — se comportam plenamente no
> Colab; ao abrir o notebook fora dele, reexecute as células para ver os gráficos.
