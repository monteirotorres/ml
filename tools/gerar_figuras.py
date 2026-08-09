"""Figuras estáticas para os decks de slides do curso de ML.
Salva em assets/slides/<topico>/.

Convenção:
    - Cada capítulo/tema tem sua própria subpasta.
    - save(fig, subpasta, nome) escreve em assets/slides/<subpasta>/<nome>.png
    - Sempre dpi=150, bbox_inches="tight", facecolor=PAPER.
    - Paleta idêntica ao style.css / template_slides.html.

Uso:
    python tools/gerar_figuras.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from pathlib import Path

BASE = Path(__file__).parent.parent

# ── Paleta ───────────────────────────────────────────────────────────────────
INK    = "#1a1a1a"
PAPER  = "#fffdf8"
BLUE   = "#3266ad"
RED    = "#c0392b"
GREEN  = "#1a7a4a"
MUTED  = "#6b6457"
BLUEF  = "#dce7f4"
REDF   = "#f6dedb"
GREENF = "#dcefe4"
GREYF  = "#ece4d3"

plt.rcParams.update({
    "font.family": "serif", "font.size": 18,
    "axes.edgecolor": "#b9ad95", "axes.linewidth": 1.2,
    "axes.titlesize": 20, "figure.facecolor": PAPER,
    "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.grid": True, "grid.color": "#e2d9c4", "grid.linewidth": 0.7,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK,
    "legend.fontsize": 17, "axes.labelsize": 18,
    "xtick.labelsize": 16, "ytick.labelsize": 16,
})

RNG = np.random.default_rng(42)


def save(fig, subpasta, name):
    out = BASE / "assets" / "slides" / subpasta
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / name, dpi=150, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print("OK", (out / name).relative_to(BASE))


# ═════════════════════════════════════════════════════════════════════════════
# FUNDAMENTOS
# ═════════════════════════════════════════════════════════════════════════════

def fig_ml_vs_prog():
    """Programação clássica vs. aprendizagem de máquina (diagrama de blocos)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax in axes:
        ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 10)

    def box(ax, x, y, w, h, text, fc, ec):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                     facecolor=fc, edgecolor=ec, lw=1.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=13, color=INK)

    def arrow(ax, x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.8))

    # clássica
    ax = axes[0]
    ax.set_title("Programação clássica", fontsize=15)
    box(ax, 0.5, 6.8, 3.2, 1.6, "Dados", BLUEF, BLUE)
    box(ax, 0.5, 1.2, 3.2, 1.6, "Regras", REDF, RED)
    box(ax, 6.0, 4.0, 3.4, 1.8, "Respostas", GREYF, MUTED)
    arrow(ax, 3.7, 7.6, 6.0, 5.4)
    arrow(ax, 3.7, 2.0, 6.0, 4.4)

    # ML
    ax = axes[1]
    ax.set_title("Aprendizagem de máquina", fontsize=15)
    box(ax, 0.5, 6.8, 3.2, 1.6, "Dados", BLUEF, BLUE)
    box(ax, 0.5, 1.2, 3.2, 1.6, "Respostas", GREENF, GREEN)
    box(ax, 6.0, 4.0, 3.4, 1.8, "Regras\n(modelo)", REDF, RED)
    arrow(ax, 3.7, 7.6, 6.0, 5.4)
    arrow(ax, 3.7, 2.0, 6.0, 4.4)
    save(fig, "fundamentos", "ml_vs_prog.png")


def fig_supervisionada():
    """Mapa dos tipos de aprendizagem."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.axis("off"); ax.set_xlim(0, 12); ax.set_ylim(0, 8)

    def box(x, y, w, h, text, fc, ec, fs=13):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                     facecolor=fc, edgecolor=ec, lw=1.6))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=INK)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.6))

    box(4.3, 6.4, 3.4, 1.3, "Aprendizagem\nde máquina", GREYF, INK, 13)
    box(0.5, 3.6, 3.3, 1.3, "Supervisionada\n(tem rótulo y)", BLUEF, BLUE, 12)
    box(8.2, 3.6, 3.3, 1.3, "Não supervisionada\n(sem rótulo)", GREENF, GREEN, 12)
    arrow(5.4, 6.4, 2.6, 4.9); arrow(6.6, 6.4, 9.4, 4.9)

    box(0.2, 0.6, 1.9, 1.2, "Regressão\n(y contínuo)", REDF, RED, 11)
    box(2.4, 0.6, 1.9, 1.2, "Classificação\n(y categoria)", REDF, RED, 11)
    arrow(1.7, 3.6, 1.2, 1.8); arrow(2.5, 3.6, 3.2, 1.8)

    box(7.7, 0.6, 1.9, 1.2, "Clustering", BLUEF, BLUE, 11)
    box(9.9, 0.6, 1.9, 1.2, "Redução de\ndimensão", BLUEF, BLUE, 11)
    arrow(9.4, 3.6, 8.7, 1.8); arrow(10.2, 3.6, 10.8, 1.8)
    save(fig, "fundamentos", "tipos_aprendizagem.png")


def fig_overfit_panels():
    """Underfitting / equilíbrio / overfitting em três painéis."""
    f_true = lambda x: np.sin(2 * np.pi * x)
    xtr = np.sort(RNG.uniform(0, 1, 12))
    ytr = f_true(xtr) + RNG.normal(0, 0.22, xtr.size)
    xx = np.linspace(0, 1, 300)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, g, t in zip(axes, [1, 4, 12],
                        ["Grau 1 — underfitting", "Grau 4 — equilíbrio",
                         "Grau 12 — overfitting"]):
        coef = np.polyfit(xtr, ytr, g)
        ax.plot(xx, f_true(xx), "--", color=GREEN, lw=2, label="real")
        ax.plot(xx, np.polyval(coef, xx), color=BLUE, lw=2.4, label="modelo")
        ax.scatter(xtr, ytr, color=RED, s=32, zorder=5)
        ax.set_ylim(-1.8, 1.8); ax.set_title(t, fontsize=14)
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].legend(frameon=False, fontsize=12, loc="lower left")
    save(fig, "fundamentos", "overfit_panels.png")


def fig_bias_variance():
    """Curvas clássicas de viés-variância."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    c = np.linspace(1, 10, 200)
    treino = 1 / c + 0.05
    teste = 1 / c + 0.05 + 0.012 * (c - 3.5) ** 2
    ax.plot(c, treino, color=BLUE, lw=2.5, label="Erro de treino")
    ax.plot(c, teste, color=RED, lw=2.5, label="Erro de teste")
    ax.axvline(3.5, color=GREEN, lw=1.5, ls="--")
    ax.text(3.7, 0.62, "equilíbrio", color=GREEN, fontsize=14)
    ax.annotate("underfitting", xy=(1.4, 0.75), fontsize=12, color=MUTED)
    ax.annotate("overfitting", xy=(7.2, 0.75), fontsize=12, color=MUTED)
    ax.set_xlabel("Complexidade do modelo"); ax.set_ylabel("Erro")
    ax.set_ylim(0, 1.0); ax.set_xticks([])
    ax.legend(frameon=False); ax.set_title("Compromisso viés–variância")
    save(fig, "fundamentos", "bias_variance.png")


def fig_validacao_cruzada():
    """Diagrama de k-fold (k=5)."""
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.set_xlim(-0.5, 10.5); ax.set_ylim(-0.5, 5.5); ax.axis("off")
    for fold in range(5):
        for bloco in range(5):
            teste = bloco == fold
            fc = REDF if teste else BLUEF
            ec = RED if teste else BLUE
            ax.add_patch(FancyBboxPatch((bloco * 2, 4 - fold), 1.85, 0.82,
                         boxstyle="round,pad=0.05", facecolor=fc, edgecolor=ec, lw=1.4))
            ax.text(bloco * 2 + 0.925, 4 - fold + 0.41,
                    "teste" if teste else "treino", ha="center", va="center",
                    fontsize=11, color=ec, weight="bold" if teste else "normal")
        ax.text(-0.3, 4 - fold + 0.41, f"Rodada {fold+1}", ha="right",
                va="center", fontsize=13, color=MUTED)
    save(fig, "fundamentos", "validacao_cruzada.png")


def fig_confusao():
    """Matriz de confusão 2x2 anotada."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.set_xlim(0, 2); ax.set_ylim(0, 2); ax.axis("off")
    cells = [((0, 1), "VP", GREENF, GREEN), ((1, 1), "FN", REDF, RED),
             ((0, 0), "FP", REDF, RED), ((1, 0), "VN", GREENF, GREEN)]
    for (x, y), lab, fc, ec in cells:
        ax.add_patch(FancyBboxPatch((x + 0.04, y + 0.04), 0.92, 0.92,
                     boxstyle="round,pad=0.02", facecolor=fc, edgecolor=ec, lw=2))
        ax.text(x + 0.5, y + 0.5, lab, ha="center", va="center",
                fontsize=26, color=ec, weight="bold")
    ax.text(0.5, 2.12, "Previu +", ha="center", fontsize=14, color=INK)
    ax.text(1.5, 2.12, "Previu −", ha="center", fontsize=14, color=INK)
    ax.text(-0.12, 1.5, "Real +", va="center", ha="right", fontsize=14, color=INK)
    ax.text(-0.12, 0.5, "Real −", va="center", ha="right", fontsize=14, color=INK)
    save(fig, "fundamentos", "matriz_confusao.png")


def fig_roc():
    """Curva ROC com AUC ilustrativa."""
    fig, ax = plt.subplots(figsize=(7, 5))
    fpr = np.linspace(0, 1, 200)
    for a, col, lab in [(0.35, BLUE, "bom (AUC≈0,9)"),
                        (0.7, GREEN, "regular (AUC≈0,75)"),
                        (1.0, MUTED, "acaso (AUC=0,5)")]:
        tpr = fpr ** a
        ax.plot(fpr, tpr, color=col, lw=2.5, label=lab)
    ax.plot([0, 1], [0, 1], "--", color=MUTED, lw=1)
    ax.set_xlabel("Taxa de falsos positivos (1 − especificidade)")
    ax.set_ylabel("Recall (verdadeiros positivos)")
    ax.set_title("Curva ROC"); ax.legend(frameon=False, fontsize=13, loc="lower right")
    save(fig, "fundamentos", "roc.png")


def fig_outlier_metricas():
    """Efeito de um outlier sobre MAE vs RMSE."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(1, 11)
    erros = np.array([0.4, 0.3, 0.5, 0.2, 0.6, 0.4, 0.3, 0.5, 0.4, 5.0])
    ax.bar(x, erros, color=[BLUE] * 9 + [RED], edgecolor=INK, lw=0.8)
    mae = np.abs(erros).mean()
    rmse = np.sqrt((erros ** 2).mean())
    ax.axhline(mae, color=GREEN, ls="--", lw=2, label=f"MAE = {mae:.2f}")
    ax.axhline(rmse, color=RED, ls="--", lw=2, label=f"RMSE = {rmse:.2f}")
    ax.set_xlabel("Exemplo"); ax.set_ylabel("|erro|")
    ax.set_title("Um outlier infla o RMSE, não o MAE")
    ax.legend(frameon=False)
    save(fig, "fundamentos", "outlier_metricas.png")


def fig_escala():
    """Efeito da padronização em dados de escalas diferentes."""
    n = 120
    idade = RNG.uniform(20, 80, n)
    colesterol = RNG.normal(220, 40, n)
    grupo = ((0.03 * idade + 0.01 * colesterol) > 3.0).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(idade, colesterol, c=[BLUE if g else RED for g in grupo], s=22)
    axes[0].set_xlabel("idade (anos)"); axes[0].set_ylabel("colesterol (mg/dL)")
    axes[0].set_title("Antes: escalas incompatíveis")

    zi = (idade - idade.mean()) / idade.std()
    zc = (colesterol - colesterol.mean()) / colesterol.std()
    axes[1].scatter(zi, zc, c=[BLUE if g else RED for g in grupo], s=22)
    axes[1].set_xlabel("idade (z-score)"); axes[1].set_ylabel("colesterol (z-score)")
    axes[1].set_title("Depois: padronizado (μ=0, σ=1)")
    axes[1].set_xlim(-3, 3); axes[1].set_ylim(-3, 3)
    save(fig, "fundamentos", "escala.png")


if __name__ == "__main__":
    fig_ml_vs_prog()
    fig_supervisionada()
    fig_overfit_panels()
    fig_bias_variance()
    fig_validacao_cruzada()
    fig_confusao()
    fig_roc()
    fig_outlier_metricas()
    fig_escala()
    print("\nFiguras do capítulo 1 geradas.")
