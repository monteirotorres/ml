# Aprendizagem de Máquina

Curso introdutório de Aprendizagem de Máquina do **IBCCF · UFRJ**, com foco
intuitivo, demonstrações interativas e notebooks em Python. O material é
publicado em **[monteirotorres.github.io/ml](https://monteirotorres.github.io/ml)**.

O curso segue o mesmo framework visual e técnico do curso de
[Bioestatística](https://monteirotorres.github.io/biostat).

## Conteúdo

1. **Fundamentos** — o que é ML, tipos de aprendizagem, representação dos dados,
   generalização (overfitting/underfitting), validação cruzada, métricas e
   pré-processamento.
2. **Regressão** — linear simples e múltipla, polinomial, regularização
   (Ridge/Lasso) e regressão logística.
3. **Classificação** — k-NN, árvores de decisão, Naive Bayes e SVM.
4. **Ensembles e Boosting** — bagging/Random Forests, Gradient Boosting,
   XGBoost/LightGBM e stacking.
5. **Aprendizagem Não Supervisionada** — k-means, clustering hierárquico, PCA,
   t-SNE e UMAP.
6. **Redes Neurais** — perceptron, backpropagation, funções de ativação e
   introdução ao deep learning.
7. **Exercícios** — listas resolvidas por tema, com soluções em Python.

## Estrutura do repositório

```
ml/
├── SUMMARY.md              índice de seções/tópicos (lido por gerar_site.py)
├── index.html, *.html      páginas geradas (não edite à mão)
├── assets/
│   ├── style.css           design system
│   ├── widgets.js          demos interativas em canvas
│   └── slides/             figuras dos decks reveal.js
├── tools/
│   ├── gerar_site.py       gera o site a partir de SUMMARY.md + markdown
│   ├── gerar_figuras.py    gera figuras dos slides (matplotlib)
│   └── gerar_notebooks.py  gera os notebooks .ipynb
├── 01_fundamentos/ ... 07_exercicios/   conteúdo em markdown + notebooks
└── *_slides.html           decks reveal.js (um por aula)
```

## Como regenerar o site

Requer Python 3 com `markdown`, `nbformat`, `numpy` e `matplotlib`:

```bash
pip install markdown nbformat numpy matplotlib

python tools/gerar_figuras.py     # gera as figuras dos slides
python tools/gerar_notebooks.py   # gera os notebooks .ipynb
python tools/gerar_site.py        # gera index.html e as páginas das seções
```

Depois de qualquer edição em um arquivo `.md`, rode `python tools/gerar_site.py`
novamente.

## Licença

Material didático de uso livre para fins educacionais.
