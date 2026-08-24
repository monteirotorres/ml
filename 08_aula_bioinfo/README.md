# Aula prática — ML aplicado a inibidores (ChEMBL) · guia do docente

Material de uma aula prática de aprendizado de máquina aplicado à bioinformática.
A partir de dados reais de inibidores da **acetilcolinesterase** (AChE, `CHEMBL220`),
os alunos treinam modelos que classificam qualquer molécula (via SMILES) como
**FORTE** ou **FRACO** — ou que se **abstêm** (classe **INDEFINIDA**) quando não
têm base para decidir.

Público: graduação em ciências biológicas/biomédicas com Python básico, `numpy`,
`pandas` e `matplotlib`, mas **sem** experiência em treinar modelos. Roda no Google
Colab, em CPU.

## Arquivos

| Arquivo | O que é |
| --- | --- |
| `aula_gabarito.ipynb` | Versão completa: perguntas com resposta e todas as células de modelagem preenchidas. |
| `aula_aluno.ipynb` | Mesma estrutura, com as células de modelagem das Seções 5–7 esvaziadas (instruções em comentário) e as respostas das perguntas removidas. |
| `../dados_alvo_bruto.csv` | Extração pré-feita do ChEMBL (10.079 medidas de IC50), na **raiz do repositório**. O notebook lê direto do link público; este é o caminho alternativo quando a API do EBI não está acessível em sala. |
| `01_ml_bioinformatica.md` | Página de apresentação da aula no site do curso. |

## Antes da aula

1. Abra o `aula_aluno.ipynb` no Colab (`File → Upload notebook` ou via GitHub).
2. Nada a subir: a célula 1.1 lê o `dados_alvo_bruto.csv` direto do link público
   do repositório. (Se preferir trabalhar offline, baixe o CSV da raiz do
   repositório e suba-o para a sessão do Colab — a célula cai no arquivo local se
   a URL falhar.)
3. Rode a Seção 0 uma vez para instalar `rdkit`, `plotly` e demais pacotes
   (cerca de 1 minuto). O restante já vem no Colab.
4. Se for usar o TensorBoard ao vivo (Seção 5b), **abra o painel antes de
   treinar**, para vê-lo atualizar durante a aula.

## Tempo estimado por seção

| Seção | Assunto | Tempo | Pode cortar? |
| --- | --- | --- | --- |
| 0 | Ambiente, semente, versões | 5 min | não |
| 1 | IC50/pIC50; carregar dados | 10 min | não |
| 2 | Curadoria + funil de proveniência | 15 min | não |
| 3 | Descritores e fingerprint de Morgan | 20 min | não |
| 4 | Partição aleatória × esqueleto; projeção do espaço químico; prova quantitativa da separação (Tanimoto) | 20 min | 4.3b (Tanimoto) para 2ª aula |
| 5 | Quatro modelos + rede em PyTorch + TensorBoard + regressão do pIC50 (5.6) | 40 min | 5b (PyTorch/TensorBoard) e 5.6 (regressão) para 2ª aula |
| 6 | Interpretabilidade, confundimento por tamanho e vazamento | 20 min | SHAP (6.3) é o mais lento — pode cortar |
| 7 | Domínio de aplicabilidade | 10 min | não (é o que sustenta a abstenção) |
| 8 | Predição conformal | 10 min | **opcional** — marcada como tal |
| 9 | `classificar()` em uso + galeria | 10 min | não (é o pagamento da aula) |
| 10 | Persistência e exercícios | 5 min | exercícios ficam de casa |

**Total cheio:** ~2 h 40 min. Para caber em **2 h**, corte a Seção 8 (conformal)
e adie a subseção 5b (PyTorch/TensorBoard) e o SHAP (6.3) para uma segunda aula.
O núcleo mínimo — Seções 0 a 4, os modelos sklearn da Seção 5, a Seção 7 e a 9 —
cabe em ~90 min.

## O que não pode ser cortado (a espinha pedagógica)

- **Seção 2** (curadoria com proveniência) — sem ela os modelos aprendem lixo.
- **Treino binário com abstenção** — o rótulo é FORTE/FRACO; a classe INDEFINIDA
  é uma **abstenção** do modelo (por ambiguidade ou por estar fora do domínio de
  aplicabilidade), não uma terceira faixa de potência. Não vire três classes.
- **Seção 4** (divisão por esqueleto) — a lição central: a partição importa mais
  que o algoritmo.
- **Seção 7** (domínio) e **Seção 9** (`classificar`) — é onde a abstenção ganha
  sentido concreto (o etanol cai em INDEFINIDA por estar fora do domínio; a
  cafeína, dentro do domínio, recebe um FRACO confiante e correto).

## Notas técnicas

- **Tempo de execução** do notebook inteiro: alguns minutos em CPU. A SVM (5.2) e
  o SHAP (6.3) são os trechos mais lentos; ambos já subamostram e avisam.
- **Plotly no Colab:** os gráficos aparecem inline; se não renderizarem após
  salvar/reabrir, reexecute a célula. Não há renderizador manual a configurar.
- **TensorBoard** (5b–5c): é a experiência ao vivo; como o painel não fica salvo
  no arquivo, **toda curva mostrada nele é replicada em Plotly** na célula
  seguinte. Registra a perda da rede (PyTorch e sklearn), a comparação de 3 taxas
  de aprendizado e também o MCC da floresta em função do número de árvores (5.4c) —
  para deixar claro que o TensorBoard não é só para redes. Limpe a pasta de logs
  entre execuções (a célula já faz isso).
- **ipywidgets** (Seção 10): o controle deslizante atualiza métricas ao vivo; o
  estado não é salvo — é preciso reexecutar a célula.
- **Reprodutibilidade:** uma única constante `SEMENTE` no topo é propagada a
  `random`, `numpy`, `torch` e a todo estimador.

## Trocar de alvo

O notebook resolve o alvo pelo símbolo do gene (`ALVO_GENE = "ACHE"`) via API do
ChEMBL. Para outro alvo, troque essa constante e reexecute a extração — ou gere um
novo `dados_alvo_bruto.csv` (na raiz do repositório) com `tools/extrair_chembl.py`
(ajustando o ChEMBL ID do alvo).
