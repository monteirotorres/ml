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
| `aula.ipynb` | O notebook completo da aula: perguntas com resposta e todas as células de código preenchidas (cada uma prefixada por `# Célula NN`). |
| `../data/dados_alvo_bruto.csv` | Extração pré-feita do ChEMBL (10.079 medidas de IC50), na pasta **`data/`** do repositório. O notebook lê direto do link público; este é o caminho alternativo quando a API do EBI não está acessível em sala. |
| `../data/world.csv` | Biblioteca de fármacos aprovados (ZINC15) usada na triagem virtual da Seção 9. |
| `01_ml_bioinformatica.md` | Página de apresentação da aula no site do curso. |

## Antes da aula

1. Abra o `aula.ipynb` no Colab (`File → Upload notebook` ou via GitHub).
2. Nada a subir: a célula 1.1 lê o `dados_alvo_bruto.csv` direto do link público
   do repositório. (Se preferir trabalhar offline, baixe o CSV da pasta `data/`
   do repositório e suba-o para a sessão do Colab — a célula cai no arquivo local
   se a URL falhar.)
3. Rode a Seção 0 uma vez para instalar `rdkit`, `plotly` e demais pacotes
   (cerca de 1 minuto). O restante já vem no Colab.

## Tempo estimado por seção

| Seção | Assunto | Tempo | Pode cortar? |
| --- | --- | --- | --- |
| 0 | Ambiente, semente, versões | 5 min | não |
| 1 | IC50/pIC50; carregar dados | 10 min | não |
| 2 | Curadoria + funil de proveniência | 15 min | não |
| 3 | Descritores, correlação e fingerprint de Morgan | 20 min | não |
| 4 | Partição aleatória × esqueleto; projeção do espaço químico | 20 min | não (é a lição central) |
| 5 | Modelos (logística, SVM, floresta, MLP e a mesma rede aberta em PyTorch) + comparação | 35 min | 5b (PyTorch) para 2ª aula |
| 6 | Interpretabilidade: confundimento por tamanho, permutação e SHAP | 20 min | SHAP (6.3) é o mais lento — pode cortar |
| 7 | Domínio de aplicabilidade | 10 min | não (é o que sustenta a abstenção) |
| 8 | `classificar()` em uso + galeria + triagem virtual | 10 min | não (é o pagamento da aula) |
| 9 | Persistência do modelo | 5 min | não |

**Total cheio:** ~2 h. Para encurtar, adie a subseção 5b (PyTorch) e o SHAP (6.3)
para uma segunda aula. O núcleo mínimo — Seções 0 a 4, os modelos sklearn da
Seção 5, a Seção 7 e a 9 — cabe em ~90 min.

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

- **Tempo de execução** do notebook inteiro: alguns minutos em CPU. A SVM (5.2),
  treinada no conjunto **completo** com kernel RBF, e o SHAP (6.3) são os trechos
  mais lentos (cerca de 1 minuto cada) e avisam antes de rodar.
- **Plotly no Colab:** os gráficos aparecem inline; se não renderizarem após
  salvar/reabrir, reexecute a célula. Não há renderizador manual a configurar.
- **PyTorch (5b):** a mesma rede do `MLPClassifier` é reimplementada à mão, com o
  laço de treino explícito, e entra na comparação final ao lado dos modelos do
  scikit-learn — o objetivo é abrir a caixa-preta do `fit`, não um modelo melhor.
- **Reprodutibilidade:** uma única constante `SEMENTE` no topo é propagada a
  `random`, `numpy`, `torch` e a todo estimador.

## Trocar de alvo

O notebook resolve o alvo pelo símbolo do gene (`ALVO_GENE = "ACHE"`) via API do
ChEMBL. Para outro alvo, troque essa constante e reexecute a extração — ou gere um
novo `data/dados_alvo_bruto.csv` com `tools/extrair_chembl.py`
(ajustando o ChEMBL ID do alvo).
