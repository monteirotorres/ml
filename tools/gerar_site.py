"""Gera o site (gitbook-style): sidebar à esquerda + uma página por seção.

- Lê SUMMARY.md (nomes de seção limpos).
- Converte markdown → HTML preservando LaTeX (KaTeX no cliente).
- Sidebar lista todas as seções/tópicos; links cross-page navegam, same-page alternam.
- Notebooks viram badges "Open in Colab".
- Injeta widgets interativos definidos no dict WIDGETS abaixo.

Uso:
    python tools/gerar_site.py
"""

import re
from pathlib import Path
import markdown

BASE = Path(__file__).parent.parent
SUMMARY = BASE / "SUMMARY.md"

GITHUB_REPO = "monteirotorres/ml"   # ← altere se o repo tiver outro nome
GITHUB_BRANCH = "main"

# cores usadas nos cards dos widgets (idênticas ao style.css)
PAL_BLUE = "#3266ad"
PAL_RED = "#c0392b"
PAL_GREEN = "#1a7a4a"

# Mapa: arquivo .md → (id base, função JS do widget)
# Adicione entradas aqui à medida que criar widgets interativos.
WIDGETS = {
    "01_fundamentos/04_generalizacao.md": ("overfit", "wOverfit"),
    "01_fundamentos/05_validacao_cruzada.md": ("kfold", "wKfold"),
    "01_fundamentos/09_ferramentas_sklearn_pytorch.md": ("skmap", "wSklearnMap"),
    "02_regressao/01_regressao_linear.md": ("linreg", "wLinReg"),
    "02_regressao/02_regressao_multipla.md": ("colin", "wColinear"),
    "02_regressao/03_regressao_polinomial.md": ("polyreg", "wOverfit"),
    "02_regressao/04_regularizacao.md": ("regul", "wRidgeLasso"),
    "02_regressao/05_regressao_logistica.md": ("logi", "wLogistic"),
    "03_classificacao/01_knn.md": ("knn", "wKnn"),
    "03_classificacao/02_arvores_decisao.md": ("tree", "wTree"),
    "03_classificacao/03_naive_bayes.md": ("bayes", "wBayes"),
    "03_classificacao/04_svm.md": ("svm", "wSvm"),
    "04_ensembles/01_random_forest.md": ("bag", "wBagging"),
    "04_ensembles/02_gradient_boosting.md": ("boost", "wBoosting"),
    "04_ensembles/04_stacking.md": ("vote", "wVoting"),
    "05_nao_supervisionado/01_kmeans.md": ("kmeans", "wKmeans"),
    "05_nao_supervisionado/02_hierarquico.md": ("dendro", "wDendro"),
    "05_nao_supervisionado/03_pca.md": ("pca", "wPca"),
}

# Mapa: nome da seção (idêntico ao ## do SUMMARY.md) → arquivo HTML de saída
SECTION_FILES = {
    "Fundamentos":                      "fundamentos.html",
    "Regressão":                        "regressao.html",
    "Classificação":                    "classificacao.html",
    "Ensembles e Boosting":             "ensembles.html",
    "Aprendizagem Não Supervisionada":  "nao_supervisionado.html",
    "Redes Neurais":                    "redes_neurais.html",
    "Exercícios":                       "exercicios.html",
    "Aula prática — Bioinformática":     "aula_bioinfo.html",
}


# ──────────────────────────────────────────────────────────────────────────────
# Widget HTML
# ──────────────────────────────────────────────────────────────────────────────
def widget_html(wid, fn):
    raw = _widget_body(wid, fn)
    return raw.replace('<div class="widget">',
                       f'<div class="widget" data-widget="{fn}" data-id="{wid}">', 1)


def _slider(wid, key, label, mn, mx, step, val):
    return (f'<div class="ctrl-row"><span class="ctrl-label">{label}</span>'
            f'<input type="range" min="{mn}" max="{mx}" step="{step}" value="{val}" id="{wid}-{key}">'
            f'<span class="ctrl-val" id="{wid}-{key}-v">{val}</span></div>')


def _card(wid, key, label, desc, color=""):
    style = f' style="color:{color};"' if color else ""
    return (f'<div class="stat-card"><div class="slabel">{label}</div>'
            f'<div class="sval" id="{wid}-{key}"{style}>—</div>'
            f'<div class="sdesc">{desc}</div></div>')


def _widget_body(wid, fn):
    # Widgets interativos por tópico. Cada um retorna HTML com ids
    # prefixados por {wid} e é implementado em assets/widgets.js.
    if fn == "wOverfit":
        return f"""<div class="widget">
  <div class="widget-title">Demonstração — grau do polinômio × generalização</div>
  <canvas id="{wid}-cv"></canvas>
  <div class="controls">
    {_slider(wid, "deg", "Grau do polinômio", 1, 12, 1, 3)}
  </div>
  <div class="btn-row">
    <button class="btn" id="{wid}-redraw">Nova amostra</button>
  </div>
  <div class="stat-grid">
    {_card(wid, "etr", "Erro de treino", "MSE nos 12 pontos vistos", PAL_BLUE)}
    {_card(wid, "ete", "Erro de teste", "MSE em dados novos", PAL_RED)}
    {_card(wid, "diag", "Diagnóstico", "regime do modelo")}
  </div>
</div>"""
    if fn == "wKfold":
        return f"""<div class="widget">
  <div class="widget-title">Demonstração — validação cruzada k-fold</div>
  <canvas id="{wid}-cv"></canvas>
  <div class="controls">
    {_slider(wid, "k", "Número de folds (k)", 2, 10, 1, 5)}
  </div>
  <div class="stat-grid">
    {_card(wid, "nmod", "Modelos treinados", "um por rodada", PAL_BLUE)}
    {_card(wid, "frac", "Fração de teste", "por rodada", PAL_RED)}
  </div>
</div>"""
    if fn == "wKmeans":
        return f"""<div class="widget">
  <div class="widget-title">k-means — agrupando ao vivo</div>
  <canvas id="{wid}-cv"></canvas>
  <div class="controls">
    {_slider(wid, "k", "Número de grupos (k)", 1, 6, 1, 3)}
  </div>
  <div class="btn-row">
    <button class="btn" id="{wid}-passo">Um passo (Lloyd)</button>
    <button class="btn" id="{wid}-conv">Convergir</button>
    <button class="btn" id="{wid}-nova">Nova amostra</button>
  </div>
  <div class="stat-grid">
    {_card(wid, "inercia", "Inércia (WCSS)", "soma das distâncias² aos centros", PAL_BLUE)}
    {_card(wid, "iter", "Iterações", "passos até estabilizar")}
  </div>
</div>"""
    if fn == "wDendro":
        return f"""<div class="widget">
  <div class="widget-title">Dendrograma — corte a árvore e conte os grupos</div>
  <canvas id="{wid}-cv"></canvas>
  <div class="controls">
    {_slider(wid, "corte", "Altura do corte", 0.05, 1, 0.01, 0.55)}
  </div>
  <div class="stat-grid">
    {_card(wid, "ngrupos", "Grupos formados", "abaixo da linha de corte", PAL_GREEN)}
  </div>
</div>"""
    if fn == "wPca":
        return f"""<div class="widget">
  <div class="widget-title">PCA — a direção de máxima variância</div>
  <canvas id="{wid}-cv"></canvas>
  <div class="controls">
    {_slider(wid, "ang", "Ângulo da direção de projeção (°)", 0, 180, 1, 20)}
  </div>
  <div class="btn-row">
    <button class="btn" id="{wid}-otimo">Ir para a 1ª componente</button>
  </div>
  <div class="stat-grid">
    {_card(wid, "vexp", "Variância capturada", "nesta direção", PAL_BLUE)}
    {_card(wid, "max", "Máximo possível (PC1)", "variância da 1ª componente", PAL_GREEN)}
  </div>
</div>"""
    if fn == "wBagging":
        return f"""<div class="widget">
  <div class="widget-title">Bagging — a média de muitas árvores reduz a variância</div>
  <canvas id="{wid}-cv"></canvas>
  <div class="controls">
    {_slider(wid, "b", "Número de árvores (B)", 1, 60, 1, 1)}
  </div>
  <div class="btn-row">
    <button class="btn" id="{wid}-nova">Nova amostra</button>
  </div>
  <div class="stat-grid">
    {_card(wid, "var", "Variabilidade da média", "dispersão do modelo combinado", PAL_BLUE)}
    {_card(wid, "reg", "Efeito", "o que B faz")}
  </div>
</div>"""
    if fn == "wBoosting":
        return f"""<div class="widget">
  <div class="widget-title">Gradient boosting — árvores que corrigem resíduos</div>
  <canvas id="{wid}-cv"></canvas>
  <div class="controls">
    {_slider(wid, "m", "Número de estágios (M)", 0, 40, 1, 0)}
    {_slider(wid, "nu", "Taxa de aprendizado ν", 0.05, 1, 0.05, 0.3)}
  </div>
  <div class="stat-grid">
    {_card(wid, "err", "Erro de treino", "resíduo quadrático médio", PAL_RED)}
    {_card(wid, "reg", "Estágio", "o modelo se aproxima")}
  </div>
</div>"""
    if fn == "wVoting":
        return f"""<div class="widget">
  <div class="widget-title">Soft voting — misturar dois modelos diferentes</div>
  <canvas id="{wid}-cv"></canvas>
  <div class="controls">
    {_slider(wid, "w", "Peso do modelo A na mistura", 0, 1, 0.05, 0.5)}
  </div>
  <div class="stat-grid">
    {_card(wid, "accA", "Acurácia — só A", "modelo de base A", PAL_BLUE)}
    {_card(wid, "accB", "Acurácia — só B", "modelo de base B", PAL_RED)}
    {_card(wid, "accM", "Acurácia da mistura", "combinação ponderada", PAL_GREEN)}
  </div>
</div>"""
    if fn == "wKnn":
        return f"""<div class="widget">
  <div class="widget-title">k-NN — como o k molda a fronteira</div>
  <canvas id="{wid}-cv"></canvas>
  <div class="controls">
    {_slider(wid, "k", "Número de vizinhos (k)", 1, 25, 2, 1)}
  </div>
  <div class="btn-row">
    <button class="btn" id="{wid}-nova">Nova amostra</button>
  </div>
  <div class="stat-grid">
    {_card(wid, "acc", "Acurácia no treino", "fração classificada certa", PAL_BLUE)}
    {_card(wid, "reg", "Regime", "efeito do k")}
  </div>
</div>"""
    if fn == "wTree":
        return f"""<div class="widget">
  <div class="widget-title">Árvore rasa — dois cortes que particionam o plano</div>
  <canvas id="{wid}-cv"></canvas>
  <div class="controls">
    {_slider(wid, "tx", "Corte vertical (x₁)", -3, 3, 0.1, 0)}
    {_slider(wid, "ty", "Corte horizontal (x₂)", -3, 3, 0.1, 0)}
  </div>
  <div class="stat-grid">
    {_card(wid, "gini", "Impureza de Gini média", "0 = regiões puras", PAL_RED)}
    {_card(wid, "acc", "Acurácia", "no limiar dos cortes", PAL_BLUE)}
  </div>
</div>"""
    if fn == "wBayes":
        return f"""<div class="widget">
  <div class="widget-title">Naive Bayes gaussiano — verossimilhança e posteriori</div>
  <canvas id="{wid}-cv"></canvas>
  <div class="controls">
    {_slider(wid, "m0", "Média da classe A", -4, 0, 0.1, -1.5)}
    {_slider(wid, "m1", "Média da classe B", 0, 4, 0.1, 1.5)}
    {_slider(wid, "pa", "Priori da classe A", 0.1, 0.9, 0.05, 0.5)}
  </div>
  <div class="stat-grid">
    {_card(wid, "fron", "Fronteira de decisão", "onde as posterioris se cruzam", PAL_GREEN)}
  </div>
</div>"""
    if fn == "wSvm":
        return f"""<div class="widget">
  <div class="widget-title">Margem máxima — encontre o melhor separador</div>
  <canvas id="{wid}-cv"></canvas>
  <div class="controls">
    {_slider(wid, "ang", "Ângulo do separador (°)", -90, 90, 1, 60)}
    {_slider(wid, "desl", "Deslocamento", -3, 3, 0.1, 0)}
  </div>
  <div class="btn-row">
    <button class="btn" id="{wid}-otimo">Aproximar da margem máxima</button>
  </div>
  <div class="stat-grid">
    {_card(wid, "margem", "Margem", "faixa até o ponto mais próximo", PAL_GREEN)}
    {_card(wid, "sep", "Separa as classes?", "sem pontos do lado errado")}
  </div>
</div>"""
    if fn == "wLinReg":
        return f"""<div class="widget">
  <div class="widget-title">Ajuste a reta na mão — mínimos quadrados</div>
  <canvas id="{wid}-cv"></canvas>
  <div class="controls">
    {_slider(wid, "b0", "Intercepto θ₀", -3, 3, 0.1, 0)}
    {_slider(wid, "b1", "Inclinação θ₁", -3, 3, 0.1, 1)}
  </div>
  <div class="btn-row">
    <button class="btn" id="{wid}-otimo">Mostrar reta ótima</button>
    <button class="btn" id="{wid}-nova">Nova amostra</button>
  </div>
  <div class="stat-grid">
    {_card(wid, "mse", "MSE da sua reta", "erro quadrático médio", PAL_BLUE)}
    {_card(wid, "mseo", "MSE da reta ótima", "o menor possível", PAL_GREEN)}
  </div>
</div>"""
    if fn == "wColinear":
        return f"""<div class="widget">
  <div class="widget-title">Colinearidade — quando dois preditores se confundem</div>
  <canvas id="{wid}-cv"></canvas>
  <div class="controls">
    {_slider(wid, "rho", "Correlação entre x₁ e x₂", 0, 0.99, 0.01, 0.0)}
  </div>
  <div class="stat-grid">
    {_card(wid, "vif", "VIF", "fator de inflação da variância", PAL_RED)}
    {_card(wid, "desvio", "Instabilidade dos coeficientes", "desvio entre 60 reajustes", PAL_BLUE)}
  </div>
</div>"""
    if fn == "wRidgeLasso":
        return f"""<div class="widget">
  <div class="widget-title">Caminho de regularização — Ridge × Lasso</div>
  <canvas id="{wid}-cv"></canvas>
  <div class="controls">
    {_slider(wid, "loga", "log₁₀(α) — força da penalidade", -3, 3, 0.1, 0)}
    <div class="ctrl-row"><span class="ctrl-label">Penalidade</span>
      <select class="ctrl-select" id="{wid}-tipo"><option value="ridge">Ridge (ℓ₂)</option><option value="lasso">Lasso (ℓ₁)</option></select></div>
  </div>
  <div class="stat-grid">
    {_card(wid, "alpha", "α atual", "força da penalidade", PAL_RED)}
    {_card(wid, "nz", "Coeficientes ≠ 0", "de 8 preditores", PAL_GREEN)}
  </div>
</div>"""
    if fn == "wLogistic":
        return f"""<div class="widget">
  <div class="widget-title">Regressão logística — sigmoide e fronteira de decisão</div>
  <canvas id="{wid}-cv"></canvas>
  <div class="controls">
    {_slider(wid, "b1", "Inclinação θ₁ (peso)", 0.2, 8, 0.1, 2)}
    {_slider(wid, "b0", "Deslocamento θ₀", -6, 6, 0.1, 0)}
  </div>
  <div class="btn-row">
    <button class="btn" id="{wid}-nova">Nova amostra</button>
  </div>
  <div class="stat-grid">
    {_card(wid, "acc", "Acurácia", "no limiar p = 0,5", PAL_BLUE)}
    {_card(wid, "loss", "Log-loss", "menor = melhor", PAL_RED)}
  </div>
</div>"""
    if fn == "wSklearnMap":
        return f"""<div class="widget">
  <div class="widget-title">Mapa de estimadores — por onde começar</div>
  <canvas id="{wid}-cv"></canvas>
  <div class="controls">
    {_slider(wid, "n", "Número de amostras", 1, 6, 1, 3)}
    <div class="ctrl-row"><span class="ctrl-label">Tenho rótulos (y conhecido)?</span>
      <select class="ctrl-select" id="{wid}-lab"><option value="sim">Sim</option><option value="nao">Não</option></select></div>
    <div class="ctrl-row"><span class="ctrl-label">Quero…</span>
      <select class="ctrl-select" id="{wid}-goal"><option value="cat">prever uma categoria</option><option value="num">prever uma quantidade</option><option value="exp">só explorar / reduzir dimensão</option></select></div>
  </div>
  <div class="stat-grid">
    {_card(wid, "fam", "Família recomendada", "para onde o mapa aponta", PAL_BLUE)}
    {_card(wid, "est", "Estimadores sugeridos", "por onde começar no scikit-learn", PAL_GREEN)}
    {_card(wid, "note", "Observação", "o porquê da recomendação")}
  </div>
</div>"""
    return ""


# ──────────────────────────────────────────────────────────────────────────────
# Markdown → HTML
# ──────────────────────────────────────────────────────────────────────────────
def md_to_html(text):
    blocks, inline = [], []
    text = re.sub(r"\$\$([\s\S]+?)\$\$",
                  lambda m: blocks.append(m.group(0)) or f"@@MB{len(blocks)-1}@@", text)
    text = re.sub(r"(?<!\\)\$([^\n$]+?)(?<!\\)\$",
                  lambda m: inline.append(m.group(0)) or f"@@MI{len(inline)-1}@@", text)
    html = markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists", "md_in_html"])
    html = re.sub(r'<pre><code class="language-mermaid">([\s\S]+?)</code></pre>',
                  lambda m: f'<div class="mermaid-container"><pre class="mermaid">{m.group(1)}</pre></div>', html)
    for i, b in enumerate(blocks):
        html = html.replace(f"@@MB{i}@@", b)
    for i, b in enumerate(inline):
        html = html.replace(f"@@MI{i}@@", b)
    html = re.sub(r"<p>\s*(\$\$[\s\S]+?\$\$)\s*</p>", r'<div class="math-display">\1</div>', html)
    return html


# ──────────────────────────────────────────────────────────────────────────────
# Parser do SUMMARY.md
# ──────────────────────────────────────────────────────────────────────────────
def parse_summary():
    section, sections, items = None, [], []
    for line in SUMMARY.read_text(encoding="utf-8").splitlines():
        h2 = re.match(r"^##\s+(.+)$", line)
        it = re.match(r"^\*\s+\[(.+?)\]\((.+?)\)$", line)
        if h2:
            if section is not None:
                sections.append((section, items))
            section, items = h2.group(1).strip(), []
        elif it and it.group(2).startswith("0"):
            items.append((it.group(1).strip(), it.group(2).strip()))
    if section is not None:
        sections.append((section, items))
    return sections


def slugify(path):
    return re.sub(r"[^a-z0-9]+", "-", path.lower()).strip("-")


# ──────────────────────────────────────────────────────────────────────────────
# Shell HTML
# ──────────────────────────────────────────────────────────────────────────────
COURSE_NAME = "Aprendizagem de Máquina"
COURSE_INST = "IBCCF · UFRJ"
COURSE_AUTHORS = "Pedro Torres"


def head(title):
    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title} — {COURSE_NAME}</title>
<script>(function(){{try{{var t=localStorage.getItem('tema');if(!t){{t=matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light';}}if(t==='dark')document.documentElement.setAttribute('data-theme','dark');}}catch(e){{}}}})();</script>
<link rel="stylesheet" href="assets/style.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"
  onload="renderMathInElement(document.body,{{delimiters:[{{left:'$$',right:'$$',display:true}},{{left:'$',right:'$',display:false}}],throwOnError:false}});"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
</head>
<body>
<button class="menu-toggle" id="menuToggle" aria-label="Menu">☰</button>
<button class="theme-toggle" id="themeToggle" aria-label="Alternar tema" title="Alternar tema claro/escuro"><span class="theme-icon"></span></button>
"""


FOOT = """
<script src="assets/widgets.js"></script>
<script>mermaid.initialize({startOnLoad:true,theme:document.documentElement.dataset.theme==='dark'?'dark':'neutral',securityLevel:'loose'});</script>
</body>
</html>
"""


def sidebar(sections, current_file):
    out = ['<nav class="sidebar" id="sidebar">']
    out.append(f'<a class="sidebar-brand" href="index.html">{COURSE_NAME}<span>{COURSE_INST}</span></a>')
    out.append('<button class="toggle-all" id="toggleAll" aria-label="Mostrar ou esconder todas as seções">− Esconder tudo</button>')
    for si, (sec, items) in enumerate(sections, 1):
        page = SECTION_FILES.get(sec, "index.html")
        out.append('<div class="nav-group">')
        out.append(f'<div class="nav-group-title">{si}. {sec}</div>')
        out.append('<ul>')
        for ti, (title, path) in enumerate(items, 1):
            slug = slugify(path)
            if page == current_file:
                out.append(f'<li><a class="nav-link" data-target="{slug}" href="#{slug}"><span class="n">{si}.{ti}</span>{title}</a></li>')
            else:
                out.append(f'<li><a href="{page}#{slug}"><span class="n">{si}.{ti}</span>{title}</a></li>')
        out.append('</ul></div>')
    out.append('</nav>')
    return "\n".join(out)


def footer():
    return (f'<div class="footer">'
            f'<div class="footer-text">{COURSE_INST} — Material didático de {COURSE_NAME}<br>{COURSE_AUTHORS}</div>'
            f'</div>')


def colab_link(ipynb):
    url = f"https://colab.research.google.com/github/{GITHUB_REPO}/blob/{GITHUB_BRANCH}/{ipynb}"
    return (f'<a class="colab-link" href="{url}" target="_blank" rel="noopener">'
            f'<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Abrir no Colab">'
            f'</a>')


def write_section_page(idx, sections):
    sec, items = sections[idx]
    page = SECTION_FILES[sec]
    out = [head(sec)]
    out.append(sidebar(sections, page))
    out.append('<main class="content"><div class="content-inner">')
    out.append(f'<div class="page-eyebrow">{idx+1}. {sec}</div>')
    for ti, (title, path) in enumerate(items, 1):
        slug = slugify(path)
        md_text = (BASE / path).read_text(encoding="utf-8")
        md_text = re.sub(r"^#\s+.+\n+", "", md_text, count=1)
        body = md_to_html(md_text)
        ipynb = path.replace(".md", ".ipynb")
        has_nb = (BASE / ipynb).exists()
        nb = colab_link(ipynb) if has_nb else ""
        widget = ""
        if path in WIDGETS:
            base_id, fn = WIDGETS[path]
            widget = widget_html(f"{base_id}_{idx+1}_{ti}", fn)
        out.append(f'<article id="{slug}" class="topic">')
        out.append(f'<div class="topic-num">{idx+1}.{ti}</div>')
        out.append(f'<h1 class="topic-title">{title}</h1>')
        if nb:
            out.append(f'<div class="nb-row">{nb}</div>')
        out.append(body)
        out.append(widget)
        out.append('</article>')
    out.append('</div>')
    out.append(footer())
    out.append('</main>')
    out.append(FOOT)
    (BASE / page).write_text("\n".join(out), encoding="utf-8")
    return page


# Descrições das seções para os cards da home
SECTION_DESCS = {
    "Fundamentos": "O que é ML, tipos de aprendizagem, overfitting, validação cruzada e métricas.",
    "Regressão": "Regressão linear, polinomial, regularização Ridge/Lasso e regressão logística.",
    "Classificação": "k-NN, árvores de decisão, SVM e Naive Bayes.",
    "Ensembles e Boosting": "Random Forests, Gradient Boosting, XGBoost e stacking.",
    "Aprendizagem Não Supervisionada": "k-means, clustering hierárquico, PCA, t-SNE e UMAP.",
    "Redes Neurais": "Perceptron, backpropagation, funções de ativação e deep learning.",
    "Exercícios": "Exercícios resolvidos por tema, com soluções em Python.",
    "Aula prática — Bioinformática": "Aula prática: classificar inibidores de um alvo (ChEMBL) como forte, fraco ou incerto.",
}


def write_index(sections):
    out = [head("Início")]
    out.append(sidebar(sections, "index.html"))
    out.append('<main class="content"><div class="content-inner">')
    out.append('<div class="hero">')
    out.append(f'<h1 class="hero-title">{COURSE_NAME}</h1>')
    out.append(f'<p class="hero-sub">{COURSE_INST}</p>')
    out.append('<p class="hero-desc">Curso introdutório com foco intuitivo, demonstrações interativas e notebooks em Python (scikit-learn, pandas, matplotlib). Navegue pelo menu à esquerda.</p>')
    out.append(f'<p class="hero-authors">{COURSE_AUTHORS}</p>')
    out.append('</div>')
    out.append('<div class="cards">')
    for si, (sec, items) in enumerate(sections, 1):
        page = SECTION_FILES.get(sec, "index.html")
        first = slugify(items[0][1]) if items else ""
        desc = SECTION_DESCS.get(sec, "")
        out.append(f'<a class="card" href="{page}#{first}">'
                   f'<div class="cnum">{si}</div>'
                   f'<div class="ctitle">{sec}</div>'
                   f'<div class="cdesc">{desc}</div>'
                   f'<div class="ccount">{len(items)} tópicos</div></a>')
    out.append('</div>')
    out.append(footer())
    out.append('</main>')
    out.append(FOOT)
    (BASE / "index.html").write_text("\n".join(out), encoding="utf-8")


def main():
    sections = parse_summary()
    write_index(sections)
    for i in range(len(sections)):
        page = write_section_page(i, sections)
        print(f"  {page}  ({(BASE/page).stat().st_size/1024:.1f} KB)")
    print(f"  index.html  ({(BASE/'index.html').stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
