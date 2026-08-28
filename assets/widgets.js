/* ============================================================
   Aprendizagem de Máquina — widgets interativos
   Princípio: mover um slider NUNCA sorteia novos valores.
   Só o botão "nova amostra" gera dados novos. Sem estado global:
   cada widget guarda seu estado no fecho da própria função.
   Canvas puro, sem bibliotecas externas.
   ============================================================ */

// ---------- PRNG determinístico (mulberry32) ----------
function makeRng(seed) {
  let s = seed >>> 0;
  return function () {
    s |= 0; s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function rngNormal(rng) {
  const u = 1 - rng(), v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// ---------- paleta (idêntica ao style.css) ----------
const PAL = {
  ink: '#1c1e15', blue: '#3266ad', red: '#c0392b', green: '#1a7a4a',
  muted: '#6b7050', line: '#d7dfbf', paper: '#fbfdf3',
  blueF: '#dce7f4', redF: '#f6dedb', greenF: '#dcefe4'
};
// no tema escuro, texto/eixos precisam clarear
function pal() {
  const dark = document.documentElement.getAttribute('data-theme') === 'dark';
  return {
    ...PAL,
    ink: dark ? '#e8eaec' : PAL.ink,
    muted: dark ? '#989ba1' : PAL.muted,
    line: dark ? '#3a3d42' : PAL.line,
    paper: dark ? '#1e2022' : PAL.paper,
  };
}

// ---------- helper: canvas com resolução de tela (HiDPI) ----------
function setupCanvas(canvas, wCss, hCss) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = wCss * dpr;
  canvas.height = hCss * dpr;
  canvas.style.width = '100%';
  canvas.style.height = 'auto';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}
const $ = id => document.getElementById(id);

// ============================================================
// Widget: overfitting / underfitting via grau do polinômio
// ------------------------------------------------------------
// Ajusta um polinômio de grau g a pontos ruidosos gerados a
// partir de uma função verdadeira (seno). Slider = grau; botão
// = nova amostra. Mostra erro de treino x erro de teste.
// ============================================================
function wOverfit(id) {
  const canvas = $(id + '-cv');
  if (!canvas) return;
  let seed = 42;

  const fTrue = x => Math.sin(2 * Math.PI * x);
  let train = [], test = [];

  function sample() {
    const rng = makeRng(seed);
    train = []; test = [];
    for (let i = 0; i < 12; i++) {
      const x = rng();
      train.push([x, fTrue(x) + 0.25 * rngNormal(rng)]);
    }
    for (let i = 0; i < 40; i++) {
      const x = rng();
      test.push([x, fTrue(x) + 0.25 * rngNormal(rng)]);
    }
  }

  // regressão polinomial por mínimos quadrados (equações normais)
  function fit(points, deg) {
    const n = points.length, m = deg + 1;
    const X = points.map(([x]) => Array.from({ length: m }, (_, j) => Math.pow(x, j)));
    const y = points.map(p => p[1]);
    // A = XᵀX (m×m), b = Xᵀy
    const A = Array.from({ length: m }, () => new Array(m).fill(0));
    const b = new Array(m).fill(0);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < m; j++) {
        b[j] += X[i][j] * y[i];
        for (let k = 0; k < m; k++) A[j][k] += X[i][j] * X[i][k];
      }
    // pequena regularização p/ estabilidade numérica em graus altos
    for (let j = 0; j < m; j++) A[j][j] += 1e-6;
    return solve(A, b);
  }
  function solve(A, b) {
    const n = b.length, M = A.map((r, i) => [...r, b[i]]);
    for (let c = 0; c < n; c++) {
      let piv = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
      [M[c], M[piv]] = [M[piv], M[c]];
      const d = M[c][c] || 1e-9;
      for (let k = c; k <= n; k++) M[c][k] /= d;
      for (let r = 0; r < n; r++) if (r !== c) {
        const f = M[r][c];
        for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
      }
    }
    return M.map(r => r[n]);
  }
  const predict = (coef, x) => coef.reduce((s, c, j) => s + c * Math.pow(x, j), 0);
  const mse = (pts, coef) => pts.reduce((s, [x, y]) => s + (y - predict(coef, x)) ** 2, 0) / pts.length;

  function draw() {
    const deg = +$(id + '-deg').value;
    $(id + '-deg-v').textContent = deg;
    const coef = fit(train, deg);
    const eTr = mse(train, coef), eTe = mse(test, coef);

    const p = pal();
    const W = 560, H = 300, pad = 34;
    const ctx = setupCanvas(canvas, W, H);
    ctx.clearRect(0, 0, W, H);
    const X0 = pad, X1 = W - pad, Y0 = H - pad, Y1 = pad;
    const sx = x => X0 + x * (X1 - X0);
    const sy = y => Y0 + (y + 1.6) / 3.2 * (Y1 - Y0);

    // eixos
    ctx.strokeStyle = p.line; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X0, Y1); ctx.lineTo(X0, Y0); ctx.lineTo(X1, Y0); ctx.stroke();

    // função verdadeira
    ctx.strokeStyle = p.green; ctx.lineWidth = 2; ctx.setLineDash([5, 4]);
    ctx.beginPath();
    for (let i = 0; i <= 120; i++) { const x = i / 120; i ? ctx.lineTo(sx(x), sy(fTrue(x))) : ctx.moveTo(sx(x), sy(fTrue(x))); }
    ctx.stroke(); ctx.setLineDash([]);

    // curva ajustada
    ctx.strokeStyle = p.blue; ctx.lineWidth = 2.4;
    ctx.beginPath();
    for (let i = 0; i <= 200; i++) {
      const x = i / 200, yv = Math.max(-1.6, Math.min(1.6, predict(coef, x)));
      i ? ctx.lineTo(sx(x), sy(yv)) : ctx.moveTo(sx(x), sy(yv));
    }
    ctx.stroke();

    // pontos de treino
    ctx.fillStyle = p.red;
    train.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(sx(x), sy(y), 4, 0, 2 * Math.PI); ctx.fill(); });

    // legenda
    ctx.font = '12px "Courier New", monospace';
    ctx.fillStyle = p.green; ctx.fillText('— função real', X1 - 130, Y1 + 6);
    ctx.fillStyle = p.blue;  ctx.fillText('— modelo (grau ' + deg + ')', X1 - 130, Y1 + 22);

    $(id + '-etr').textContent = eTr.toFixed(3);
    $(id + '-ete').textContent = eTe.toFixed(3);
    // diagnóstico
    let diag = 'equilíbrio';
    if (deg <= 1) diag = 'underfitting';
    else if (eTe > 3 * eTr && deg >= 6) diag = 'overfitting';
    $(id + '-diag').textContent = diag;
  }

  $(id + '-deg').addEventListener('input', draw);
  $(id + '-redraw').addEventListener('click', () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; sample(); draw(); });
  window.addEventListener('resize', draw);
  sample(); draw();
}

// ============================================================
// Widget: validação cruzada k-fold (diagrama animado)
// ------------------------------------------------------------
// Slider = k. Desenha o esquema de partição em k folds e o
// número de modelos treinados. Puramente ilustrativo.
// ============================================================
function wKfold(id) {
  const canvas = $(id + '-cv');
  if (!canvas) return;

  function draw() {
    const k = +$(id + '-k').value;
    $(id + '-k-v').textContent = k;
    const p = pal();
    const W = 560, H = 40 + k * 34;
    const ctx = setupCanvas(canvas, W, H);
    ctx.clearRect(0, 0, W, H);
    const labelW = 70, gap = 6;
    const cellW = (W - labelW - (5 * gap)) / 5 * 5 / k; // largura por bloco
    const blockW = (W - labelW - gap) / k;

    ctx.font = '12px "Courier New", monospace';
    for (let fold = 0; fold < k; fold++) {
      const y = 20 + fold * 34;
      ctx.fillStyle = p.muted;
      ctx.textAlign = 'right';
      ctx.fillText('Rodada ' + (fold + 1), labelW - 8, y + 18);
      ctx.textAlign = 'center';
      for (let b = 0; b < k; b++) {
        const x = labelW + b * blockW;
        const isTest = b === fold;
        ctx.fillStyle = isTest ? p.redF : p.blueF;
        ctx.strokeStyle = isTest ? p.red : p.blue;
        ctx.lineWidth = 1.4;
        roundRect(ctx, x, y, blockW - gap, 26, 5);
        ctx.fill(); ctx.stroke();
        ctx.fillStyle = isTest ? p.red : p.blue;
        if (blockW > 46) ctx.fillText(isTest ? 'teste' : 'treino', x + (blockW - gap) / 2, y + 17);
      }
    }
    $(id + '-nmod').textContent = k;
    $(id + '-frac').textContent = Math.round(100 / k) + '%';
  }
  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }
  $(id + '-k').addEventListener('input', draw);
  window.addEventListener('resize', draw);
  draw();
}

// ============================================================
// Widget: mapa de estimadores do scikit-learn (versão enxuta)
// ------------------------------------------------------------
// Três controles (nº de amostras, tenho rótulos?, o que quero)
// conduzem a uma família de modelos, no espírito do fluxograma
// oficial "choosing the right estimator". Desenha os quatro
// destinos e destaca o escolhido. Sem canvas de dados: é decisão.
// ============================================================
function wSklearnMap(id) {
  const cv = $(id + '-cv');
  const escala = [20, 100, 1000, 10000, 100000, 1000000];   // slider 1..6

  function recomendar() {
    const n = escala[(+$(id + '-n').value) - 1];
    const temRotulo = $(id + '-lab').value;      // 'sim' | 'nao'
    const objetivo = $(id + '-goal').value;      // 'cat' | 'num' | 'exp'
    let alvo, familia, estimadores, nota;

    if (n < 50) {
      alvo = 'nada';
      familia = 'Consiga mais dados';
      estimadores = '—';
      nota = 'Com menos de ~50 amostras o mapa recomenda coletar mais dados antes de tentar modelar.';
    } else if (objetivo === 'num') {
      alvo = 'reg';
      familia = 'Regressão';
      estimadores = n < 100000
        ? 'Ridge/Lasso/ElasticNet, SVR, floresta/boosting'
        : 'SGDRegressor (incremental)';
      nota = n < 100000
        ? 'Alvo numérico e contínuo: comece pelos modelos lineares regularizados e suba para floresta ou boosting.'
        : 'Muitas amostras: prefira um estimador incremental como o SGDRegressor.';
    } else if (objetivo === 'exp') {
      alvo = 'dim';
      familia = 'Redução de dimensionalidade';
      estimadores = 'PCA; se não bastar, Isomap/t-SNE/UMAP';
      nota = 'Sem alvo definido, o objetivo é enxergar estrutura: projete os dados em menos dimensões (PCA é o começo).';
    } else {                                   // objetivo === 'cat'
      if (temRotulo === 'sim') {
        alvo = 'cls';
        familia = 'Classificação';
        estimadores = n < 100000
          ? 'SVM linear, kNN, floresta aleatória, SVC'
          : 'SGDClassifier, aproximação de kernel';
        nota = n < 100000
          ? 'Categoria com rótulos: comece por modelos lineares/kNN e suba para floresta ou SVC.'
          : 'Muitas amostras rotuladas: prefira estimadores incrementais (SGD) ou aproximação de kernel.';
      } else {
        alvo = 'clu';
        familia = 'Clustering';
        estimadores = n < 10000
          ? 'KMeans, clustering espectral, mistura de gaussianas'
          : 'MiniBatchKMeans';
        nota = 'Você quer categorias, mas não tem rótulos: o caminho é agrupar (não supervisionado).';
      }
    }

    $(id + '-n-v').textContent = n.toLocaleString('pt-BR');
    $(id + '-fam').textContent = familia;
    $(id + '-est').textContent = estimadores;
    $(id + '-note').textContent = nota;

    // habilita/realça o controle de rótulos só quando faz diferença (categoria)
    $(id + '-lab').parentElement.style.opacity = (objetivo === 'cat') ? '1' : '0.45';

    desenhar(alvo);
  }

  function desenhar(alvo) {
    const p = pal();
    const W = 640, H = 150;
    const ctx = setupCanvas(cv, W, H);
    ctx.clearRect(0, 0, W, H);
    const nos = [
      { k: 'cls', rot: 'Classificação', cor: p.blue,  corF: p.blueF },
      { k: 'reg', rot: 'Regressão',     cor: p.green, corF: p.greenF },
      { k: 'clu', rot: 'Clustering',    cor: p.red,   corF: p.redF },
      { k: 'dim', rot: 'Redução dim.',  cor: p.muted, corF: p.line },
    ];
    const larg = 138, alt = 62, gap = (W - nos.length * larg) / (nos.length + 1);
    ctx.font = '600 14px Georgia, serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    nos.forEach((no, i) => {
      const x = gap + i * (larg + gap), y = (H - alt) / 2;
      const ativo = no.k === alvo;
      ctx.globalAlpha = (alvo === 'nada') ? 0.5 : (ativo ? 1 : 0.35);
      ctx.fillStyle = ativo ? no.corF : p.paper;
      ctx.strokeStyle = no.cor;
      ctx.lineWidth = ativo ? 3 : 1.2;
      const r = 10;
      ctx.beginPath();
      ctx.moveTo(x + r, y); ctx.arcTo(x + larg, y, x + larg, y + alt, r);
      ctx.arcTo(x + larg, y + alt, x, y + alt, r); ctx.arcTo(x, y + alt, x, y, r);
      ctx.arcTo(x, y, x + larg, y, r); ctx.closePath();
      ctx.fill(); ctx.stroke();
      ctx.fillStyle = no.cor;
      ctx.fillText(no.rot, x + larg / 2, y + alt / 2);
    });
    ctx.globalAlpha = 1;
    if (alvo === 'nada') {
      ctx.fillStyle = p.ink;
      ctx.font = 'italic 15px Georgia, serif';
      ctx.fillText('amostras insuficientes — colete mais dados', W / 2, 18);
    }
  }

  ['n', 'lab', 'goal'].forEach(k =>
    $(id + '-' + k).addEventListener('input', recomendar));
  window.addEventListener('resize', recomendar);
  recomendar();
}

// ============================================================
// Widget: ajuste de reta por mínimos quadrados (regressão linear)
// ============================================================
function wLinReg(id) {
  const cv = $(id + '-cv');
  let pts = [], semente = 7;
  function gerar() {
    const rng = makeRng(semente), a = 1.3, b = 0.4;
    pts = [];
    for (let i = 0; i < 14; i++) {
      const x = -2 + 4 * rng();
      pts.push([x, a * x + b + rngNormal(rng) * 0.9]);
    }
  }
  function otimo() {
    const n = pts.length; let mx = 0, my = 0;
    pts.forEach(p => { mx += p[0]; my += p[1]; }); mx /= n; my /= n;
    let sxy = 0, sxx = 0;
    pts.forEach(p => { sxy += (p[0] - mx) * (p[1] - my); sxx += (p[0] - mx) ** 2; });
    const b1 = sxy / sxx; return [my - b1 * mx, b1];
  }
  function mse(b0, b1) {
    let s = 0; pts.forEach(p => { const e = p[1] - (b0 + b1 * p[0]); s += e * e; });
    return s / pts.length;
  }
  function draw() {
    const p = pal(), W = 640, H = 340, ctx = setupCanvas(cv, W, H);
    ctx.clearRect(0, 0, W, H);
    const padL = 40, padR = 20, padT = 18, padB = 28, xmin = -2.5, xmax = 2.5, ymin = -4, ymax = 4;
    const X = x => padL + (x - xmin) / (xmax - xmin) * (W - padL - padR);
    const Y = y => H - padB - (y - ymin) / (ymax - ymin) * (H - padT - padB);
    ctx.strokeStyle = p.line; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X(xmin), Y(0)); ctx.lineTo(X(xmax), Y(0));
    ctx.moveTo(X(0), Y(ymin)); ctx.lineTo(X(0), Y(ymax)); ctx.stroke();
    const b0 = +$(id + '-b0').value, b1 = +$(id + '-b1').value;
    ctx.strokeStyle = p.red; ctx.lineWidth = 1;
    pts.forEach(pt => { const yh = b0 + b1 * pt[0]; ctx.beginPath(); ctx.moveTo(X(pt[0]), Y(pt[1])); ctx.lineTo(X(pt[0]), Y(yh)); ctx.stroke(); });
    ctx.strokeStyle = p.blue; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(X(xmin), Y(b0 + b1 * xmin)); ctx.lineTo(X(xmax), Y(b0 + b1 * xmax)); ctx.stroke();
    ctx.fillStyle = p.ink;
    pts.forEach(pt => { ctx.beginPath(); ctx.arc(X(pt[0]), Y(pt[1]), 4, 0, 7); ctx.fill(); });
    const o = otimo();
    $(id + '-mse').textContent = mse(b0, b1).toFixed(3);
    $(id + '-mseo').textContent = mse(o[0], o[1]).toFixed(3);
  }
  function sincroniza() {
    $(id + '-b0-v').textContent = (+$(id + '-b0').value).toFixed(1);
    $(id + '-b1-v').textContent = (+$(id + '-b1').value).toFixed(1);
  }
  ['b0', 'b1'].forEach(k => $(id + '-' + k).addEventListener('input', () => { sincroniza(); draw(); }));
  $(id + '-otimo').addEventListener('click', () => {
    const o = otimo();
    $(id + '-b0').value = o[0].toFixed(1); $(id + '-b1').value = o[1].toFixed(1);
    sincroniza(); draw();
  });
  $(id + '-nova').addEventListener('click', () => { semente = (semente * 1664525 + 1013904223) >>> 0; gerar(); draw(); });
  window.addEventListener('resize', draw);
  gerar(); sincroniza(); draw();
}

// ============================================================
// Widget: colinearidade — VIF e instabilidade dos coeficientes
// ============================================================
function wColinear(id) {
  const cv = $(id + '-cv');
  function draw() {
    const p = pal(), rho = +$(id + '-rho').value;
    $(id + '-rho-v').textContent = rho.toFixed(2);
    $(id + '-vif').textContent = (1 / (1 - rho * rho)).toFixed(1);
    // 60 reajustes: desvio-padrão do coeficiente de x1 (dispara quando rho -> 1)
    const rng = makeRng(2024), coefs = [];
    for (let b = 0; b < 60; b++) {
      const n = 40; let s11 = 0, s12 = 0, s22 = 0, s1y = 0, s2y = 0;
      for (let i = 0; i < n; i++) {
        const a = rngNormal(rng), c = rngNormal(rng);
        const u = a, v = rho * a + Math.sqrt(1 - rho * rho) * c;
        const y = u + v + rngNormal(rng) * 0.5;
        s11 += u * u; s12 += u * v; s22 += v * v; s1y += u * y; s2y += v * y;
      }
      const det = s11 * s22 - s12 * s12;
      coefs.push(Math.abs(det) < 1e-9 ? 0 : (s22 * s1y - s12 * s2y) / det);
    }
    const m = coefs.reduce((a, b) => a + b, 0) / coefs.length;
    const sd = Math.sqrt(coefs.reduce((a, b) => a + (b - m) * (b - m), 0) / coefs.length);
    $(id + '-desvio').textContent = '±' + sd.toFixed(2);
    // nuvem (x1, x2) colapsando numa reta
    const W = 640, H = 300, ctx = setupCanvas(cv, W, H);
    ctx.clearRect(0, 0, W, H);
    const padL = 36, padR = 18, padT = 18, padB = 28, lim = 3;
    const X = x => padL + (x + lim) / (2 * lim) * (W - padL - padR);
    const Y = y => H - padB - (y + lim) / (2 * lim) * (H - padT - padB);
    ctx.strokeStyle = p.line; ctx.beginPath();
    ctx.moveTo(X(-lim), Y(0)); ctx.lineTo(X(lim), Y(0));
    ctx.moveTo(X(0), Y(-lim)); ctx.lineTo(X(0), Y(lim)); ctx.stroke();
    const rng2 = makeRng(99); ctx.fillStyle = p.blue; ctx.globalAlpha = 0.6;
    for (let i = 0; i < 130; i++) {
      const a = rngNormal(rng2), c = rngNormal(rng2);
      ctx.beginPath(); ctx.arc(X(a), Y(rho * a + Math.sqrt(1 - rho * rho) * c), 3, 0, 7); ctx.fill();
    }
    ctx.globalAlpha = 1; ctx.fillStyle = p.muted; ctx.font = '12px Georgia, serif'; ctx.textAlign = 'center';
    ctx.fillText('x₁', X(lim) - 10, Y(0) - 8); ctx.fillText('x₂', X(0) + 14, Y(lim) + 6);
  }
  $(id + '-rho').addEventListener('input', draw);
  window.addEventListener('resize', draw); draw();
}

// ============================================================
// Widget: caminho de regularização (Ridge x Lasso), ilustrativo
// ------------------------------------------------------------
// Sobre coeficientes OLS fixos, aplica encolhimento estilizado:
// Ridge  theta = beta/(1+alpha);  Lasso  soft-threshold.
// Mostra a diferença qualitativa: Ridge encolhe, Lasso zera.
// ============================================================
function wRidgeLasso(id) {
  const cv = $(id + '-cv');
  const beta = [2.5, -1.8, 1.2, 0.0, 0.6, 0.0, -0.9, 0.05];
  const cores = [PAL.blue, PAL.red, PAL.green, '#8660a0', '#d68910', '#16a085', '#7f8c8d', '#b9770e'];
  function coef(b, alpha, tipo) {
    if (tipo === 'ridge') return b / (1 + alpha);
    const m = Math.abs(b) - alpha * 0.5; return m > 0 ? Math.sign(b) * m : 0;
  }
  function draw() {
    const p = pal(), loga = +$(id + '-loga').value, tipo = $(id + '-tipo').value, alpha = Math.pow(10, loga);
    $(id + '-loga-v').textContent = loga.toFixed(1);
    $(id + '-alpha').textContent = alpha < 1 ? alpha.toFixed(3) : alpha.toFixed(1);
    const th = beta.map(b => coef(b, alpha, tipo));
    $(id + '-nz').textContent = th.filter(t => Math.abs(t) > 1e-3).length + ' / 8';
    const W = 640, H = 320, ctx = setupCanvas(cv, W, H);
    ctx.clearRect(0, 0, W, H);
    const padL = 46, padR = 14, padT = 16, padB = 34, la0 = -3, la1 = 3, ymax = 3;
    const X = la => padL + (la - la0) / (la1 - la0) * (W - padL - padR);
    const Y = v => padT + (ymax - v) / (2 * ymax) * (H - padT - padB);
    ctx.strokeStyle = p.line; ctx.beginPath(); ctx.moveTo(padL, Y(0)); ctx.lineTo(W - padR, Y(0)); ctx.stroke();
    for (let j = 0; j < beta.length; j++) {
      ctx.strokeStyle = cores[j % cores.length]; ctx.lineWidth = 2; ctx.beginPath();
      for (let k = 0; k <= 60; k++) {
        const la = la0 + (la1 - la0) * k / 60;
        const v = Math.max(-ymax, Math.min(ymax, coef(beta[j], Math.pow(10, la), tipo)));
        k ? ctx.lineTo(X(la), Y(v)) : ctx.moveTo(X(la), Y(v));
      }
      ctx.stroke();
    }
    ctx.strokeStyle = p.muted; ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(X(loga), padT); ctx.lineTo(X(loga), H - padB); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = p.muted; ctx.font = '12px Georgia, serif'; ctx.textAlign = 'center';
    ctx.fillText('log₁₀(α)', W / 2, H - 8);
    ctx.save(); ctx.translate(13, H / 2); ctx.rotate(-Math.PI / 2); ctx.fillText('coeficiente', 0, 0); ctx.restore();
  }
  $(id + '-loga').addEventListener('input', draw);
  $(id + '-tipo').addEventListener('change', draw);
  window.addEventListener('resize', draw); draw();
}

// ============================================================
// Widget: regressão logística — sigmoide e fronteira de decisão
// ============================================================
function wLogistic(id) {
  const cv = $(id + '-cv');
  let pts = [], semente = 5;
  function gerar() {
    const rng = makeRng(semente); pts = [];
    for (let i = 0; i < 30; i++) {
      const c = i < 15 ? 0 : 1;
      pts.push([(c ? 1 : -1) + rngNormal(rng) * 0.8, c]);
    }
  }
  function draw() {
    const p = pal(), b0 = +$(id + '-b0').value, b1 = +$(id + '-b1').value;
    $(id + '-b0-v').textContent = b0.toFixed(1); $(id + '-b1-v').textContent = b1.toFixed(1);
    const W = 640, H = 320, ctx = setupCanvas(cv, W, H);
    ctx.clearRect(0, 0, W, H);
    const padL = 40, padR = 20, padT = 20, padB = 30, xmin = -4, xmax = 4;
    const X = x => padL + (x - xmin) / (xmax - xmin) * (W - padL - padR);
    const Y = pp => H - padB - pp * (H - padT - padB);
    ctx.strokeStyle = p.line; ctx.lineWidth = 1;
    [0, 0.5, 1].forEach(v => { ctx.beginPath(); ctx.moveTo(padL, Y(v)); ctx.lineTo(W - padR, Y(v)); ctx.stroke(); });
    ctx.strokeStyle = p.blue; ctx.lineWidth = 2.5; ctx.beginPath();
    for (let k = 0; k <= 120; k++) {
      const x = xmin + (xmax - xmin) * k / 120, pr = 1 / (1 + Math.exp(-(b0 + b1 * x)));
      k ? ctx.lineTo(X(x), Y(pr)) : ctx.moveTo(X(x), Y(pr));
    }
    ctx.stroke();
    if (Math.abs(b1) > 1e-3) {
      const xb = -b0 / b1;
      if (xb > xmin && xb < xmax) {
        ctx.strokeStyle = p.muted; ctx.setLineDash([4, 4]);
        ctx.beginPath(); ctx.moveTo(X(xb), padT); ctx.lineTo(X(xb), H - padB); ctx.stroke(); ctx.setLineDash([]);
      }
    }
    let acertos = 0, loss = 0;
    pts.forEach(pt => {
      const pr = 1 / (1 + Math.exp(-(b0 + b1 * pt[0]))), yv = pt[1];
      ctx.fillStyle = yv ? p.red : p.blue; ctx.globalAlpha = 0.8;
      ctx.beginPath(); ctx.arc(X(pt[0]), Y(yv), 4, 0, 7); ctx.fill();
      if ((pr > 0.5 ? 1 : 0) === yv) acertos++;
      loss += -(yv * Math.log(pr + 1e-9) + (1 - yv) * Math.log(1 - pr + 1e-9));
    });
    ctx.globalAlpha = 1;
    $(id + '-acc').textContent = (100 * acertos / pts.length).toFixed(0) + '%';
    $(id + '-loss').textContent = (loss / pts.length).toFixed(3);
  }
  ['b0', 'b1'].forEach(k => $(id + '-' + k).addEventListener('input', draw));
  $(id + '-nova').addEventListener('click', () => { semente = (semente * 1664525 + 1013904223) >>> 0; gerar(); draw(); });
  window.addEventListener('resize', draw);
  gerar(); draw();
}

// ============================================================
// Widget: k-NN — o k molda a fronteira de decisão
// ============================================================
function wKnn(id) {
  const cv = $(id + '-cv'); let pts = [], semente = 11;
  function gerar() {
    const rng = makeRng(semente); pts = [];
    for (let i = 0; i < 40; i++) {
      const c = i < 20 ? 0 : 1, cx = c ? 1.1 : -1.1, cy = c ? 0.6 : -0.4;
      pts.push([cx + rngNormal(rng) * 0.9, cy + rngNormal(rng) * 0.9, c]);
    }
  }
  function voto(x, y, k, excl) {
    const ds = [];
    for (let i = 0; i < pts.length; i++) {
      if (i === excl) continue;
      const dx = x - pts[i][0], dy = y - pts[i][1];
      ds.push([dx * dx + dy * dy, pts[i][2]]);
    }
    ds.sort((a, b) => a[0] - b[0]);
    let v = 0; for (let i = 0; i < k && i < ds.length; i++) v += ds[i][1] ? 1 : -1;
    return v >= 0 ? 1 : 0;
  }
  function draw() {
    const p = pal(), k = +$(id + '-k').value, W = 640, H = 360, ctx = setupCanvas(cv, W, H);
    ctx.clearRect(0, 0, W, H);
    const padL = 10, padR = 10, padT = 10, padB = 10, xmin = -4, xmax = 4, ymin = -3, ymax = 3;
    const X = x => padL + (x - xmin) / (xmax - xmin) * (W - padL - padR);
    const Y = y => H - padB - (y - ymin) / (ymax - ymin) * (H - padT - padB);
    const nx = 48, ny = 28;
    for (let gx = 0; gx < nx; gx++) for (let gy = 0; gy < ny; gy++) {
      const x0 = xmin + (xmax - xmin) * gx / nx, x1 = xmin + (xmax - xmin) * (gx + 1) / nx;
      const y0 = ymin + (ymax - ymin) * gy / ny, y1 = ymin + (ymax - ymin) * (gy + 1) / ny;
      ctx.fillStyle = voto((x0 + x1) / 2, (y0 + y1) / 2, k, -1) ? p.redF : p.blueF;
      ctx.fillRect(X(x0), Y(y1), X(x1) - X(x0) + 1, Y(y0) - Y(y1) + 1);
    }
    pts.forEach(pt => {
      ctx.fillStyle = pt[2] ? p.red : p.blue; ctx.strokeStyle = p.paper; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(X(pt[0]), Y(pt[1]), 5, 0, 7); ctx.fill(); ctx.stroke();
    });
    let ok = 0;
    for (let i = 0; i < pts.length; i++) if (voto(pts[i][0], pts[i][1], k, i) === pts[i][2]) ok++;
    $(id + '-acc').textContent = (100 * ok / pts.length).toFixed(0) + '%';
    $(id + '-reg').textContent = k <= 3 ? 'fronteira recortada (variância alta)'
      : k >= 15 ? 'fronteira suave (viés alto)' : 'equilíbrio';
  }
  $(id + '-k').addEventListener('input', () => { $(id + '-k-v').textContent = $(id + '-k').value; draw(); });
  $(id + '-nova').addEventListener('click', () => { semente = (semente * 1664525 + 1013904223) >>> 0; gerar(); draw(); });
  window.addEventListener('resize', draw);
  gerar(); $(id + '-k-v').textContent = '1'; draw();
}

// ============================================================
// Widget: árvore rasa — dois cortes particionam o plano
// ============================================================
function wTree(id) {
  const cv = $(id + '-cv'); let pts = [];
  (function () {
    const rng = makeRng(3);
    for (let i = 0; i < 60; i++) {
      const x = rngNormal(rng) * 1.3, y = rngNormal(rng) * 1.3;
      const c = ((x > 0) === (y > 0)) ? 1 : 0;
      pts.push([x, y, rng() < 0.1 ? 1 - c : c]);
    }
  })();
  const regiao = (x, y, tx, ty) => (x > tx ? 1 : 0) * 2 + (y > ty ? 1 : 0);
  function draw() {
    const p = pal(), tx = +$(id + '-tx').value, ty = +$(id + '-ty').value;
    const W = 640, H = 360, ctx = setupCanvas(cv, W, H); ctx.clearRect(0, 0, W, H);
    const padL = 10, padR = 10, padT = 10, padB = 10, xmin = -4, xmax = 4, ymin = -4, ymax = 4;
    const X = x => padL + (x - xmin) / (xmax - xmin) * (W - padL - padR);
    const Y = y => H - padB - (y - ymin) / (ymax - ymin) * (H - padT - padB);
    const cnt = [[0, 0], [0, 0], [0, 0], [0, 0]];
    pts.forEach(pt => cnt[regiao(pt[0], pt[1], tx, ty)][pt[2]]++);
    const maj = cnt.map(c => c[1] >= c[0] ? 1 : 0);
    const paint = (x0, x1, y0, y1, ri) => { ctx.fillStyle = maj[ri] ? p.redF : p.blueF; ctx.fillRect(X(x0), Y(y1), X(x1) - X(x0), Y(y0) - Y(y1)); };
    paint(xmin, tx, ymin, ty, 0); paint(xmin, tx, ty, ymax, 1);
    paint(tx, xmax, ymin, ty, 2); paint(tx, xmax, ty, ymax, 3);
    ctx.strokeStyle = p.ink; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(X(tx), Y(ymin)); ctx.lineTo(X(tx), Y(ymax));
    ctx.moveTo(X(xmin), Y(ty)); ctx.lineTo(X(xmax), Y(ty)); ctx.stroke();
    pts.forEach(pt => {
      ctx.fillStyle = pt[2] ? p.red : p.blue; ctx.strokeStyle = p.paper; ctx.lineWidth = 1.2;
      ctx.beginPath(); ctx.arc(X(pt[0]), Y(pt[1]), 4.5, 0, 7); ctx.fill(); ctx.stroke();
    });
    let gsum = 0, acc = 0, ntot = pts.length;
    for (let r = 0; r < 4; r++) {
      const n = cnt[r][0] + cnt[r][1]; if (!n) continue;
      const p1 = cnt[r][1] / n; gsum += (1 - (p1 * p1 + (1 - p1) * (1 - p1))) * n / ntot;
      acc += Math.max(cnt[r][0], cnt[r][1]);
    }
    $(id + '-gini').textContent = gsum.toFixed(3);
    $(id + '-acc').textContent = (100 * acc / ntot).toFixed(0) + '%';
  }
  ['tx', 'ty'].forEach(k => $(id + '-' + k).addEventListener('input', () => { $(id + '-' + k + '-v').textContent = (+$(id + '-' + k).value).toFixed(1); draw(); }));
  window.addEventListener('resize', draw);
  $(id + '-tx-v').textContent = '0.0'; $(id + '-ty-v').textContent = '0.0'; draw();
}

// ============================================================
// Widget: Naive Bayes gaussiano — densidades e fronteira
// ============================================================
function wBayes(id) {
  const cv = $(id + '-cv');
  const dens = (x, m) => Math.exp(-0.5 * (x - m) * (x - m));   // normal (std=1), sem a constante
  function draw() {
    const p = pal(), m0 = +$(id + '-m0').value, m1 = +$(id + '-m1').value, pa = +$(id + '-pa').value;
    $(id + '-m0-v').textContent = m0.toFixed(1); $(id + '-m1-v').textContent = m1.toFixed(1); $(id + '-pa-v').textContent = pa.toFixed(2);
    const W = 640, H = 320, ctx = setupCanvas(cv, W, H); ctx.clearRect(0, 0, W, H);
    const padL = 24, padR = 14, padT = 16, padB = 26, xmin = -6, xmax = 6;
    const X = x => padL + (x - xmin) / (xmax - xmin) * (W - padL - padR);
    const top = padT, bot = H - padB, hgt = bot - top, N = 200;
    for (let i = 0; i < N; i++) {
      const x = xmin + (xmax - xmin) * i / N;
      ctx.fillStyle = (pa * dens(x, m0) >= (1 - pa) * dens(x, m1)) ? p.blueF : p.redF;
      ctx.fillRect(X(x), top, (W - padL - padR) / N + 1, hgt);
    }
    const escala = hgt * 0.9; ctx.lineWidth = 2.5;
    [[m0, pa, p.blue], [m1, 1 - pa, p.red]].forEach(g => {
      ctx.strokeStyle = g[2]; ctx.beginPath();
      for (let k = 0; k <= 180; k++) {
        const x = xmin + (xmax - xmin) * k / 180, d = g[1] * dens(x, g[0]);
        k ? ctx.lineTo(X(x), bot - d * escala) : ctx.moveTo(X(x), bot - d * escala);
      }
      ctx.stroke();
    });
    let xb = null;
    if (Math.abs(m1 - m0) > 1e-6) xb = (Math.log(pa / (1 - pa)) - 0.5 * (m0 * m0 - m1 * m1)) / (m1 - m0);
    if (xb !== null && xb > xmin && xb < xmax) {
      ctx.strokeStyle = p.ink; ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(X(xb), top); ctx.lineTo(X(xb), bot); ctx.stroke(); ctx.setLineDash([]);
    }
    $(id + '-fron').textContent = xb === null ? 'indefinida' : ('x = ' + xb.toFixed(2));
  }
  ['m0', 'm1', 'pa'].forEach(k => $(id + '-' + k).addEventListener('input', draw));
  window.addEventListener('resize', draw); draw();
}

// ============================================================
// Widget: SVM — margem máxima encontrada à mão
// ============================================================
function wSvm(id) {
  const cv = $(id + '-cv'); let pts = [];
  (function () {
    const rng = makeRng(8);
    for (let i = 0; i < 24; i++) {
      const c = i < 12 ? 0 : 1, cx = c ? 1.6 : -1.6, cy = c ? 1.0 : -1.0;
      pts.push([cx + rngNormal(rng) * 0.6, cy + rngNormal(rng) * 0.6, c]);
    }
  })();
  function draw() {
    const p = pal(), a = (+$(id + '-ang').value) * Math.PI / 180, off = +$(id + '-desl').value;
    $(id + '-ang-v').textContent = $(id + '-ang').value; $(id + '-desl-v').textContent = off.toFixed(1);
    const nx = Math.cos(a), ny = Math.sin(a);
    const W = 640, H = 360, ctx = setupCanvas(cv, W, H); ctx.clearRect(0, 0, W, H);
    const padL = 10, padR = 10, padT = 10, padB = 10, xmin = -4, xmax = 4, ymin = -3, ymax = 3;
    const X = x => padL + (x - xmin) / (xmax - xmin) * (W - padL - padR);
    const Y = y => H - padB - (y - ymin) / (ymax - ymin) * (H - padT - padB);
    let margem = 1e9, ok = true;
    pts.forEach(pt => {
      const s = nx * pt[0] + ny * pt[1] - off;
      if (Math.abs(s) < margem) margem = Math.abs(s);
      if ((s >= 0 ? 1 : 0) !== pt[2]) ok = false;
    });
    const px = off * nx, py = off * ny, dx = -ny, dy = nx, t = 7;
    ctx.strokeStyle = p.ink; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(X(px - dx * t), Y(py - dy * t)); ctx.lineTo(X(px + dx * t), Y(py + dy * t)); ctx.stroke();
    if (ok) {
      ctx.setLineDash([5, 4]); ctx.strokeStyle = p.green;
      [margem, -margem].forEach(mm => {
        const qx = (off + mm) * nx, qy = (off + mm) * ny;
        ctx.beginPath(); ctx.moveTo(X(qx - dx * t), Y(qy - dy * t)); ctx.lineTo(X(qx + dx * t), Y(qy + dy * t)); ctx.stroke();
      });
      ctx.setLineDash([]);
    }
    pts.forEach(pt => {
      ctx.fillStyle = pt[2] ? p.red : p.blue; ctx.strokeStyle = p.paper; ctx.lineWidth = 1.4;
      ctx.beginPath(); ctx.arc(X(pt[0]), Y(pt[1]), 5, 0, 7); ctx.fill(); ctx.stroke();
    });
    $(id + '-margem').textContent = ok ? margem.toFixed(2) : '—';
    $(id + '-sep').textContent = ok ? 'sim' : 'não (há pontos do lado errado)';
  }
  $(id + '-otimo').addEventListener('click', () => {
    let m0 = [0, 0], m1 = [0, 0], n0 = 0, n1 = 0;
    pts.forEach(pt => { if (pt[2]) { m1[0] += pt[0]; m1[1] += pt[1]; n1++; } else { m0[0] += pt[0]; m0[1] += pt[1]; n0++; } });
    m0 = [m0[0] / n0, m0[1] / n0]; m1 = [m1[0] / n1, m1[1] / n1];
    let a = Math.atan2(m1[1] - m0[1], m1[0] - m0[0]) * 180 / Math.PI;
    const nx = Math.cos(a * Math.PI / 180), ny = Math.sin(a * Math.PI / 180);
    const off = nx * (m0[0] + m1[0]) / 2 + ny * (m0[1] + m1[1]) / 2;
    $(id + '-ang').value = Math.round(a); $(id + '-desl').value = off.toFixed(1); draw();
  });
  ['ang', 'desl'].forEach(k => $(id + '-' + k).addEventListener('input', draw));
  window.addEventListener('resize', draw); draw();
}

// ============================================================
// Widget: bagging — a média de muitas árvores reduz a variância
// ============================================================
function wBagging(id) {
  const cv = $(id + '-cv'); let xs = [], ys = [], semente = 13;
  const K = 6, G = 80, POOL = 40;
  function gerar() {
    const rng = makeRng(semente); xs = []; ys = [];
    for (let i = 0; i < 30; i++) { const x = rng(); xs.push(x); ys.push(Math.sin(2 * Math.PI * x) + rngNormal(rng) * 0.35); }
  }
  function arvore(rng) {
    const s = new Array(K).fill(0), c = new Array(K).fill(0);
    for (let i = 0; i < xs.length; i++) { const j = Math.floor(rng() * xs.length); const b = Math.min(K - 1, Math.floor(xs[j] * K)); s[b] += ys[j]; c[b]++; }
    const m = []; for (let b = 0; b < K; b++) m.push(c[b] ? s[b] / c[b] : (b > 0 ? m[b - 1] : 0)); return m;
  }
  const prever = (m, x) => m[Math.min(K - 1, Math.floor(x * K))];
  function draw() {
    const p = pal(), B = +$(id + '-b').value, rng = makeRng(semente ^ 0x9e3779b9);
    const pool = []; for (let t = 0; t < POOL; t++) pool.push(arvore(rng));
    const W = 640, H = 340, ctx = setupCanvas(cv, W, H); ctx.clearRect(0, 0, W, H);
    const padL = 30, padR = 16, padT = 14, padB = 24;
    const X = x => padL + x * (W - padL - padR), Y = y => H - padB - ((y + 1.7) / 3.4) * (H - padT - padB);
    ctx.lineWidth = 1; ctx.strokeStyle = p.line;
    for (let t = 0; t < Math.min(B, 12); t++) {
      ctx.beginPath(); for (let g = 0; g <= G; g++) { const x = g / G, yy = prever(pool[t], x); g ? ctx.lineTo(X(x), Y(yy)) : ctx.moveTo(X(x), Y(yy)); } ctx.stroke();
    }
    let seSum = 0;
    ctx.lineWidth = 3; ctx.strokeStyle = p.blue; ctx.beginPath();
    for (let g = 0; g <= G; g++) {
      const x = g / G;
      let sB = 0; for (let t = 0; t < B; t++) sB += prever(pool[t], x); const mediaB = sB / B;
      let s = 0, s2 = 0; for (let t = 0; t < POOL; t++) { const v = prever(pool[t], x); s += v; s2 += v * v; }
      const varInd = Math.max(0, s2 / POOL - (s / POOL) * (s / POOL));
      seSum += Math.sqrt(varInd) / Math.sqrt(B);
      g ? ctx.lineTo(X(x), Y(mediaB)) : ctx.moveTo(X(x), Y(mediaB));
    }
    ctx.stroke();
    ctx.strokeStyle = p.ink; ctx.setLineDash([4, 4]); ctx.lineWidth = 1.5; ctx.beginPath();
    for (let g = 0; g <= G; g++) { const x = g / G; g ? ctx.lineTo(X(x), Y(Math.sin(2 * Math.PI * x))) : ctx.moveTo(X(x), Y(Math.sin(2 * Math.PI * x))); } ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = p.muted; ctx.globalAlpha = 0.5;
    for (let i = 0; i < xs.length; i++) { ctx.beginPath(); ctx.arc(X(xs[i]), Y(ys[i]), 3, 0, 7); ctx.fill(); } ctx.globalAlpha = 1;
    $(id + '-var').textContent = (seSum / (G + 1)).toFixed(3);
    $(id + '-reg').textContent = B <= 2 ? 'poucas árvores: média instável' : B >= 25 ? 'muitas árvores: média estável' : 'a média se firma';
  }
  $(id + '-b').addEventListener('input', () => { $(id + '-b-v').textContent = $(id + '-b').value; draw(); });
  $(id + '-nova').addEventListener('click', () => { semente = (semente * 1664525 + 1013904223) >>> 0; gerar(); draw(); });
  window.addEventListener('resize', draw);
  gerar(); $(id + '-b-v').textContent = '1'; draw();
}

// ============================================================
// Widget: gradient boosting — stumps ajustados aos resíduos
// ============================================================
function wBoosting(id) {
  const cv = $(id + '-cv'); let xs = [], ys = []; const G = 100;
  (function () {
    const rng = makeRng(21);
    for (let i = 0; i < 40; i++) { const x = i / 39; xs.push(x); ys.push(Math.sin(2 * Math.PI * x) + rngNormal(rng) * 0.15); }
  })();
  function stump(res) {
    let best = null;
    for (let ti = 1; ti < xs.length; ti++) {
      const t = (xs[ti - 1] + xs[ti]) / 2;
      let sl = 0, cl = 0, sr = 0, cr = 0;
      for (let i = 0; i < xs.length; i++) { if (xs[i] < t) { sl += res[i]; cl++; } else { sr += res[i]; cr++; } }
      if (!cl || !cr) continue;
      const ml = sl / cl, mr = sr / cr; let sse = 0;
      for (let i = 0; i < xs.length; i++) { const pr = xs[i] < t ? ml : mr; sse += (res[i] - pr) ** 2; }
      if (!best || sse < best.sse) best = { t, ml, mr, sse };
    }
    return best;
  }
  const preverStump = (s, x) => x < s.t ? s.ml : s.mr;
  function draw() {
    const p = pal(), M = +$(id + '-m').value, nu = +$(id + '-nu').value;
    $(id + '-m-v').textContent = M; $(id + '-nu-v').textContent = nu.toFixed(2);
    const media0 = ys.reduce((a, b) => a + b, 0) / ys.length;
    const F = new Array(xs.length).fill(media0), arvores = [];
    for (let m = 0; m < M; m++) {
      const res = ys.map((y, i) => y - F[i]);
      const tr = stump(res); if (!tr) break; arvores.push(tr);
      for (let i = 0; i < xs.length; i++) F[i] += nu * preverStump(tr, xs[i]);
    }
    const pred = x => { let v = media0; for (let m = 0; m < arvores.length; m++) v += nu * preverStump(arvores[m], x); return v; };
    const W = 640, H = 340, ctx = setupCanvas(cv, W, H); ctx.clearRect(0, 0, W, H);
    const padL = 30, padR = 16, padT = 14, padB = 24;
    const X = x => padL + x * (W - padL - padR), Y = y => H - padB - ((y + 1.6) / 3.2) * (H - padT - padB);
    ctx.strokeStyle = p.ink; ctx.setLineDash([4, 4]); ctx.lineWidth = 1.5; ctx.beginPath();
    for (let g = 0; g <= G; g++) { const x = g / G; g ? ctx.lineTo(X(x), Y(Math.sin(2 * Math.PI * x))) : ctx.moveTo(X(x), Y(Math.sin(2 * Math.PI * x))); } ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = p.blue; ctx.lineWidth = 3; ctx.beginPath();
    for (let g = 0; g <= G; g++) { const x = g / G; g ? ctx.lineTo(X(x), Y(pred(x))) : ctx.moveTo(X(x), Y(pred(x))); } ctx.stroke();
    ctx.fillStyle = p.muted; ctx.globalAlpha = 0.5;
    for (let i = 0; i < xs.length; i++) { ctx.beginPath(); ctx.arc(X(xs[i]), Y(ys[i]), 3, 0, 7); ctx.fill(); } ctx.globalAlpha = 1;
    let err = 0; for (let i = 0; i < xs.length; i++) err += (ys[i] - F[i]) ** 2; err /= xs.length;
    $(id + '-err').textContent = err.toFixed(3);
    $(id + '-reg').textContent = M === 0 ? 'só a média (viés máximo)' : M >= 25 ? 'bem ajustado' : 'aprendendo os resíduos';
  }
  ['m', 'nu'].forEach(k => $(id + '-' + k).addEventListener('input', draw));
  window.addEventListener('resize', draw);
  $(id + '-m-v').textContent = '0'; $(id + '-nu-v').textContent = '0.30'; draw();
}

// ============================================================
// Widget: soft voting — misturar dois modelos complementares
// ============================================================
function wVoting(id) {
  const cv = $(id + '-cv'); const pts = [];
  for (let i = 0; i < 24; i++) {
    const t = i % 2, region = i < 8 ? 0 : (i < 16 ? 1 : 2);
    const pA = region === 1 ? (t ? 0.35 : 0.65) : (t ? 0.82 : 0.18);
    const pB = region === 2 ? (t ? 0.35 : 0.65) : (t ? 0.82 : 0.18);
    pts.push({ t, pA, pB });
  }
  const acc = w => { let ok = 0; pts.forEach(pt => { if (((w * pt.pA + (1 - w) * pt.pB) > 0.5 ? 1 : 0) === pt.t) ok++; }); return ok / pts.length; };
  function draw() {
    const p = pal(), w = +$(id + '-w').value;
    $(id + '-w-v').textContent = w.toFixed(2);
    $(id + '-accA').textContent = (100 * acc(1)).toFixed(0) + '%';
    $(id + '-accB').textContent = (100 * acc(0)).toFixed(0) + '%';
    $(id + '-accM').textContent = (100 * acc(w)).toFixed(0) + '%';
    const W = 640, H = 210, ctx = setupCanvas(cv, W, H); ctx.clearRect(0, 0, W, H);
    const cols = 12, mx = 40, my = 46, gw = (W - 2 * mx) / (cols - 1), gh = 74;
    pts.forEach((pt, i) => {
      const r = Math.floor(i / cols), c = i % cols, x = mx + c * gw, y = my + r * gh;
      const correct = (((w * pt.pA + (1 - w) * pt.pB) > 0.5 ? 1 : 0) === pt.t);
      ctx.fillStyle = pt.t ? p.red : p.blue; ctx.beginPath(); ctx.arc(x, y, 9, 0, 7); ctx.fill();
      ctx.lineWidth = 3; ctx.strokeStyle = correct ? p.green : p.red;
      ctx.beginPath(); ctx.arc(x, y, 13, 0, 7); ctx.stroke();
    });
    ctx.fillStyle = p.muted; ctx.font = '12px Georgia, serif'; ctx.textAlign = 'left';
    ctx.fillText('preenchimento = classe verdadeira · anel verde = a mistura acertou', mx, H - 12);
  }
  $(id + '-w').addEventListener('input', draw);
  window.addEventListener('resize', draw); draw();
}

// ============================================================
// Widget: k-means ao vivo (algoritmo de Lloyd)
// ============================================================
function wKmeans(id) {
  const cv = $(id + '-cv'); let pts = [], cent = [], iter = 0, semente = 7;
  const cores = ['#3266ad', '#c0392b', '#1a7a4a', '#8660a0', '#d68910', '#16a085'];
  function gerar() {
    const rng = makeRng(semente); pts = [];
    const centros = [[-1.5, -1], [1.5, -1.2], [0, 1.6], [2.2, 1.4]];
    for (let i = 0; i < 120; i++) { const c = centros[i % 4]; pts.push([c[0] + rngNormal(rng) * 0.5, c[1] + rngNormal(rng) * 0.5]); }
  }
  function initCent() {
    const k = +$(id + '-k').value, rng = makeRng((semente * 2654435761) >>> 0); cent = []; const usados = new Set();
    while (cent.length < k) { const j = Math.floor(rng() * pts.length); if (usados.has(j)) continue; usados.add(j); cent.push([pts[j][0], pts[j][1]]); }
    iter = 0;
  }
  function rotular() {
    return pts.map(p => { let bi = 0, bd = 1e9; for (let c = 0; c < cent.length; c++) { const dx = p[0] - cent[c][0], dy = p[1] - cent[c][1], d = dx * dx + dy * dy; if (d < bd) { bd = d; bi = c; } } return bi; });
  }
  function passo() {
    const lab = rotular(), s = cent.map(() => [0, 0, 0]);
    for (let i = 0; i < pts.length; i++) { const c = lab[i]; s[c][0] += pts[i][0]; s[c][1] += pts[i][1]; s[c][2]++; }
    let moveu = false;
    for (let c = 0; c < cent.length; c++) if (s[c][2]) { const nx = s[c][0] / s[c][2], ny = s[c][1] / s[c][2]; if (Math.abs(nx - cent[c][0]) > 1e-6 || Math.abs(ny - cent[c][1]) > 1e-6) moveu = true; cent[c] = [nx, ny]; }
    iter++; return moveu;
  }
  function draw() {
    const p = pal(), lab = rotular(), W = 640, H = 360, ctx = setupCanvas(cv, W, H);
    ctx.clearRect(0, 0, W, H);
    const padL = 10, padR = 10, padT = 10, padB = 10, xmin = -3.5, xmax = 4, ymin = -3, ymax = 3.5;
    const X = x => padL + (x - xmin) / (xmax - xmin) * (W - padL - padR);
    const Y = y => H - padB - (y - ymin) / (ymax - ymin) * (H - padT - padB);
    ctx.globalAlpha = 0.65;
    for (let i = 0; i < pts.length; i++) { ctx.fillStyle = cores[lab[i] % cores.length]; ctx.beginPath(); ctx.arc(X(pts[i][0]), Y(pts[i][1]), 4, 0, 7); ctx.fill(); }
    ctx.globalAlpha = 1;
    let inercia = 0;
    for (let i = 0; i < pts.length; i++) { const c = lab[i], dx = pts[i][0] - cent[c][0], dy = pts[i][1] - cent[c][1]; inercia += dx * dx + dy * dy; }
    for (let c = 0; c < cent.length; c++) {
      ctx.fillStyle = cores[c % cores.length]; ctx.strokeStyle = p.ink; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.arc(X(cent[c][0]), Y(cent[c][1]), 9, 0, 7); ctx.fill(); ctx.stroke();
      ctx.strokeStyle = p.paper; ctx.lineWidth = 2; const cx = X(cent[c][0]), cy = Y(cent[c][1]);
      ctx.beginPath(); ctx.moveTo(cx - 4, cy - 4); ctx.lineTo(cx + 4, cy + 4); ctx.moveTo(cx + 4, cy - 4); ctx.lineTo(cx - 4, cy + 4); ctx.stroke();
    }
    $(id + '-inercia').textContent = inercia.toFixed(1); $(id + '-iter').textContent = iter;
  }
  $(id + '-k').addEventListener('input', () => { $(id + '-k-v').textContent = $(id + '-k').value; initCent(); draw(); });
  $(id + '-passo').addEventListener('click', () => { passo(); draw(); });
  $(id + '-conv').addEventListener('click', () => { let n = 0; while (passo() && n < 50) n++; draw(); });
  $(id + '-nova').addEventListener('click', () => { semente = (semente * 1664525 + 1013904223) >>> 0; gerar(); initCent(); draw(); });
  window.addEventListener('resize', draw);
  gerar(); $(id + '-k-v').textContent = '3'; initCent(); draw();
}

// ============================================================
// Widget: dendrograma com linha de corte
// ============================================================
function wDendro(id) {
  const cv = $(id + '-cv'); const leaves = 8;
  const merges = [
    { a: 0, b: 1, h: 0.12 }, { a: 2, b: 3, h: 0.15 }, { a: 8, b: 9, h: 0.35 },
    { a: 4, b: 5, h: 0.18 }, { a: 6, b: 7, h: 0.22 }, { a: 11, b: 12, h: 0.45 },
    { a: 10, b: 13, h: 0.85 },
  ];
  const nx = {}, nh = {};
  for (let i = 0; i < leaves; i++) { nx[i] = i; nh[i] = 0; }
  merges.forEach((m, k) => { const id2 = leaves + k; nx[id2] = (nx[m.a] + nx[m.b]) / 2; nh[id2] = m.h; });
  const cores = ['#3266ad', '#c0392b', '#1a7a4a', '#8660a0', '#d68910', '#16a085', '#2c7fb8', '#b9770e'];
  function draw() {
    const p = pal(), cut = +$(id + '-corte').value;
    $(id + '-corte-v').textContent = cut.toFixed(2);
    const parent = {}; for (let i = 0; i < leaves + merges.length; i++) parent[i] = i;
    const find = x => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
    merges.forEach((m, k) => { if (m.h <= cut) { parent[find(m.a)] = find(leaves + k); parent[find(m.b)] = find(leaves + k); } });
    const rootColor = {}; let ci = 0; const leafRoot = [];
    for (let i = 0; i < leaves; i++) { const r = find(i); if (!(r in rootColor)) { rootColor[r] = cores[ci % cores.length]; ci++; } leafRoot.push(r); }
    $(id + '-ngrupos').textContent = new Set(leafRoot).size;
    const W = 640, H = 320, ctx = setupCanvas(cv, W, H); ctx.clearRect(0, 0, W, H);
    const padL = 20, padR = 20, padT = 20, padB = 34, hmax = 1.0;
    const X = x => padL + x / (leaves - 1) * (W - padL - padR);
    const Y = h => H - padB - (h / hmax) * (H - padT - padB);
    merges.forEach((m, k) => {
      const id2 = leaves + k, below = m.h <= cut;
      ctx.strokeStyle = below ? (rootColor[find(id2)] || p.ink) : p.line; ctx.lineWidth = below ? 2.5 : 1.2;
      ctx.beginPath();
      ctx.moveTo(X(nx[m.a]), Y(nh[m.a])); ctx.lineTo(X(nx[m.a]), Y(m.h));
      ctx.lineTo(X(nx[m.b]), Y(m.h)); ctx.lineTo(X(nx[m.b]), Y(nh[m.b])); ctx.stroke();
    });
    ctx.strokeStyle = p.red; ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(padL, Y(cut)); ctx.lineTo(W - padR, Y(cut)); ctx.stroke(); ctx.setLineDash([]);
    ctx.fillStyle = p.red; ctx.font = '12px Georgia, serif'; ctx.textAlign = 'right'; ctx.fillText('corte', W - padR, Y(cut) - 6);
    for (let i = 0; i < leaves; i++) { ctx.fillStyle = rootColor[leafRoot[i]]; ctx.beginPath(); ctx.arc(X(i), Y(0), 4, 0, 7); ctx.fill(); }
  }
  $(id + '-corte').addEventListener('input', draw);
  window.addEventListener('resize', draw); draw();
}

// ============================================================
// Widget: PCA — direção de máxima variância
// ============================================================
function wPca(id) {
  const cv = $(id + '-cv'); let pts = [], mx = 0, my = 0, C = [0, 0, 0];
  (function () {
    const rng = makeRng(15), ang = Math.PI / 6;
    for (let i = 0; i < 120; i++) {
      const a = rngNormal(rng) * 1.7, b = rngNormal(rng) * 0.55;
      pts.push([a * Math.cos(ang) - b * Math.sin(ang), a * Math.sin(ang) + b * Math.cos(ang)]);
    }
    for (const q of pts) { mx += q[0]; my += q[1]; } mx /= pts.length; my /= pts.length;
    for (const q of pts) { C[0] += (q[0] - mx) ** 2; C[1] += (q[0] - mx) * (q[1] - my); C[2] += (q[1] - my) ** 2; }
    C = C.map(v => v / pts.length);
  })();
  const trace = C[0] + C[2];
  const varDir = th => { const c = Math.cos(th), s = Math.sin(th); return c * c * C[0] + 2 * c * s * C[1] + s * s * C[2]; };
  const pc1 = 0.5 * Math.atan2(2 * C[1], C[0] - C[2]);
  const varPC1 = varDir(pc1);
  function draw() {
    const p = pal(), th = (+$(id + '-ang').value) * Math.PI / 180;
    $(id + '-ang-v').textContent = $(id + '-ang').value;
    $(id + '-vexp').textContent = (100 * varDir(th) / trace).toFixed(1) + '%';
    $(id + '-max').textContent = (100 * varPC1 / trace).toFixed(1) + '%';
    const W = 640, H = 340, ctx = setupCanvas(cv, W, H); ctx.clearRect(0, 0, W, H);
    const lim = 5, s = Math.min((W - 20) / (2 * lim), (H - 20) / (2 * lim)), cxp = W / 2, cyp = H / 2;
    const X = x => cxp + (x - mx) * s, Y = y => cyp - (y - my) * s;
    const u = [Math.cos(th), Math.sin(th)], t = 6;
    ctx.strokeStyle = p.green; ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(X(mx - u[0] * t), Y(my - u[1] * t)); ctx.lineTo(X(mx + u[0] * t), Y(my + u[1] * t)); ctx.stroke();
    for (const pt of pts) {
      const dx = pt[0] - mx, dy = pt[1] - my, proj = dx * u[0] + dy * u[1];
      const fx = mx + proj * u[0], fy = my + proj * u[1];
      ctx.strokeStyle = p.line; ctx.lineWidth = 0.7;
      ctx.beginPath(); ctx.moveTo(X(pt[0]), Y(pt[1])); ctx.lineTo(X(fx), Y(fy)); ctx.stroke();
      ctx.fillStyle = p.blue; ctx.globalAlpha = 0.6; ctx.beginPath(); ctx.arc(X(pt[0]), Y(pt[1]), 3.5, 0, 7); ctx.fill(); ctx.globalAlpha = 1;
      ctx.fillStyle = p.green; ctx.beginPath(); ctx.arc(X(fx), Y(fy), 2.5, 0, 7); ctx.fill();
    }
  }
  $(id + '-ang').addEventListener('input', draw);
  $(id + '-otimo').addEventListener('click', () => { let d = pc1 * 180 / Math.PI; if (d < 0) d += 180; $(id + '-ang').value = Math.round(d); draw(); });
  window.addEventListener('resize', draw); draw();
}

// ============================================================
// Widget: perceptron — uma reta resolve E/OU, mas não o XOR
// ============================================================
function wPerceptron(id) {
  const cv = $(id + '-cv'); const pontos = [[0, 0], [0, 1], [1, 0], [1, 1]];
  function rotulos(prob) {
    return pontos.map(pt => { const a = pt[0], b = pt[1]; if (prob === 'and') return (a && b) ? 1 : 0; if (prob === 'or') return (a || b) ? 1 : 0; return (a ^ b) ? 1 : 0; });
  }
  function draw() {
    const p = pal(), prob = $(id + '-prob').value, w1 = +$(id + '-w1').value, w2 = +$(id + '-w2').value, b = +$(id + '-b').value;
    ['w1', 'w2', 'b'].forEach(k => $(id + '-' + k + '-v').textContent = (+$(id + '-' + k).value).toFixed(1));
    const lab = rotulos(prob);
    const W = 640, H = 340, ctx = setupCanvas(cv, W, H); ctx.clearRect(0, 0, W, H);
    const padL = 40, padR = 40, padT = 20, padB = 26, xmin = -0.6, xmax = 1.6, ymin = -0.6, ymax = 1.6;
    const X = x => padL + (x - xmin) / (xmax - xmin) * (W - padL - padR);
    const Y = y => H - padB - (y - ymin) / (ymax - ymin) * (H - padT - padB);
    const N = 60;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
      const x0 = xmin + (xmax - xmin) * i / N, x1 = xmin + (xmax - xmin) * (i + 1) / N;
      const y0 = ymin + (ymax - ymin) * j / N, y1 = ymin + (ymax - ymin) * (j + 1) / N;
      ctx.fillStyle = (w1 * (x0 + x1) / 2 + w2 * (y0 + y1) / 2 + b) >= 0 ? p.redF : p.blueF;
      ctx.fillRect(X(x0), Y(y1), X(x1) - X(x0) + 1, Y(y0) - Y(y1) + 1);
    }
    ctx.strokeStyle = p.ink; ctx.lineWidth = 2; ctx.beginPath();
    if (Math.abs(w2) > 1e-6) { ctx.moveTo(X(xmin), Y(-(w1 * xmin + b) / w2)); ctx.lineTo(X(xmax), Y(-(w1 * xmax + b) / w2)); }
    else if (Math.abs(w1) > 1e-6) { const xv = -b / w1; ctx.moveTo(X(xv), Y(ymin)); ctx.lineTo(X(xv), Y(ymax)); }
    ctx.stroke();
    let acc = 0;
    pontos.forEach((pt, i) => {
      const pred = (w1 * pt[0] + w2 * pt[1] + b) >= 0 ? 1 : 0; if (pred === lab[i]) acc++;
      ctx.fillStyle = lab[i] ? p.red : p.blue; ctx.beginPath(); ctx.arc(X(pt[0]), Y(pt[1]), 10, 0, 7); ctx.fill();
      ctx.lineWidth = 3; ctx.strokeStyle = pred === lab[i] ? p.green : p.red;
      ctx.beginPath(); ctx.arc(X(pt[0]), Y(pt[1]), 15, 0, 7); ctx.stroke();
    });
    $(id + '-acc').textContent = acc + ' / 4';
    $(id + '-sep').textContent = prob === 'xor' ? 'Não — nenhuma reta separa' : 'Sim';
  }
  ['prob', 'w1', 'w2', 'b'].forEach(k => $(id + '-' + k).addEventListener('input', draw));
  window.addEventListener('resize', draw); draw();
}

// ============================================================
// Widget: gradiente descendente numa superfície de perda 1D
// ============================================================
function wGradDescent(id) {
  const cv = $(id + '-cv'); let x = -2.5, rodando = false, timer = null;
  const L = z => 0.15 * z * z * z * z - 0.5 * z * z + 0.2 * z + 1;
  const dL = z => 0.6 * z * z * z - z + 0.2;
  function passo() { const lr = +$(id + '-lr').value; x = x - lr * dL(x); if (!isFinite(x) || Math.abs(x) > 6) x = Math.sign(x || 1) * 6; }
  function draw() {
    const p = pal(), lr = +$(id + '-lr').value; $(id + '-lr-v').textContent = lr.toFixed(2);
    const W = 640, H = 320, ctx = setupCanvas(cv, W, H); ctx.clearRect(0, 0, W, H);
    const padL = 30, padR = 16, padT = 16, padB = 26, xmin = -3, xmax = 3, ymin = 0, ymax = 3;
    const X = z => padL + (z - xmin) / (xmax - xmin) * (W - padL - padR);
    const Y = y => H - padB - (y - ymin) / (ymax - ymin) * (H - padT - padB);
    ctx.strokeStyle = p.blue; ctx.lineWidth = 2.5; ctx.beginPath();
    for (let g = 0; g <= 160; g++) { const z = xmin + (xmax - xmin) * g / 160, y = Math.min(ymax, L(z)); g ? ctx.lineTo(X(z), Y(y)) : ctx.moveTo(X(z), Y(y)); } ctx.stroke();
    const bx = Math.max(xmin, Math.min(xmax, x));
    ctx.fillStyle = p.red; ctx.strokeStyle = p.paper; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.arc(X(bx), Y(Math.min(ymax, L(x))), 8, 0, 7); ctx.fill(); ctx.stroke();
    $(id + '-perda').textContent = L(x).toFixed(3);
    $(id + '-reg').textContent = lr <= 0.15 ? 'passos pequenos: desce devagar' : lr >= 0.9 ? 'passo grande: pode saltar/divergir' : 'taxa equilibrada';
  }
  function anim() { if (!rodando) return; passo(); draw(); if (Math.abs(dL(x)) < 1e-3 || Math.abs(x) >= 6) { rodando = false; return; } timer = setTimeout(anim, 70); }
  $(id + '-lr').addEventListener('input', draw);
  $(id + '-passo').addEventListener('click', () => { passo(); draw(); });
  $(id + '-run').addEventListener('click', () => { if (rodando) return; rodando = true; anim(); });
  $(id + '-reset').addEventListener('click', () => { rodando = false; if (timer) clearTimeout(timer); x = -2.5; draw(); });
  window.addEventListener('resize', draw); draw();
}

// ============================================================
// Widget: funções de ativação e suas derivadas
// ============================================================
function wActivation(id) {
  const cv = $(id + '-cv');
  function f(z, k) { if (k === 'sigmoid') return 1 / (1 + Math.exp(-z)); if (k === 'tanh') return Math.tanh(z); if (k === 'relu') return Math.max(0, z); return Math.max(0.1 * z, z); }
  function df(z, k) { if (k === 'sigmoid') { const s = 1 / (1 + Math.exp(-z)); return s * (1 - s); } if (k === 'tanh') { const t = Math.tanh(z); return 1 - t * t; } if (k === 'relu') return z > 0 ? 1 : 0; return z > 0 ? 1 : 0.1; }
  function draw() {
    const p = pal(), k = $(id + '-fn').value;
    const W = 640, H = 320, ctx = setupCanvas(cv, W, H); ctx.clearRect(0, 0, W, H);
    const padL = 34, padR = 16, padT = 16, padB = 26, zmin = -6, zmax = 6, ymin = -1.2, ymax = 2.2;
    const X = z => padL + (z - zmin) / (zmax - zmin) * (W - padL - padR);
    const Y = y => H - padB - (y - ymin) / (ymax - ymin) * (H - padT - padB);
    ctx.strokeStyle = p.line; ctx.beginPath(); ctx.moveTo(X(zmin), Y(0)); ctx.lineTo(X(zmax), Y(0)); ctx.moveTo(X(0), Y(ymin)); ctx.lineTo(X(0), Y(ymax)); ctx.stroke();
    ctx.strokeStyle = p.blue; ctx.lineWidth = 2.5; ctx.beginPath();
    for (let g = 0; g <= 200; g++) { const z = zmin + (zmax - zmin) * g / 200, y = Math.max(ymin, Math.min(ymax, f(z, k))); g ? ctx.lineTo(X(z), Y(y)) : ctx.moveTo(X(z), Y(y)); } ctx.stroke();
    ctx.strokeStyle = p.red; ctx.lineWidth = 2; ctx.setLineDash([5, 3]); ctx.beginPath();
    for (let g = 0; g <= 200; g++) { const z = zmin + (zmax - zmin) * g / 200, y = Math.max(ymin, Math.min(ymax, df(z, k))); g ? ctx.lineTo(X(z), Y(y)) : ctx.moveTo(X(z), Y(y)); } ctx.stroke(); ctx.setLineDash([]);
    ctx.font = '13px Georgia, serif'; ctx.textAlign = 'left';
    ctx.fillStyle = p.blue; ctx.fillText('f(z)', X(zmax) - 66, Y(ymax) + 8);
    ctx.fillStyle = p.red; ctx.fillText("f '(z)", X(zmax) - 66, Y(ymax) + 26);
    let dmax = 0; for (let g = 0; g <= 200; g++) { const z = zmin + (zmax - zmin) * g / 200; dmax = Math.max(dmax, df(z, k)); }
    $(id + '-dmax').textContent = dmax.toFixed(2);
    $(id + '-satura').textContent = (k === 'sigmoid' || k === 'tanh') ? 'Sim — deriv. → 0 nas pontas' : 'Não — deriv. constante p/ z>0';
  }
  $(id + '-fn').addEventListener('change', draw);
  window.addEventListener('resize', draw); draw();
}

// ============================================================
// Tabs + sidebar runtime  (infra do site — NÃO alterar)
// ============================================================
function showTopic(target) {
  document.querySelectorAll('.topic').forEach(t => t.classList.toggle('active', t.id === target));
  document.querySelectorAll('.nav-link').forEach(a => a.classList.toggle('active', a.dataset.target === target));
  window.dispatchEvent(new Event('resize'));
}

window.addEventListener('DOMContentLoaded', () => {
  // inicializa widgets
  document.querySelectorAll('[data-widget]').forEach(el => {
    const fn = window[el.dataset.widget];
    if (typeof fn === 'function') { try { fn(el.dataset.id); } catch (e) { console.error('widget', el.dataset.widget, e); } }
  });

  // navegação por links da sidebar (mesma página)
  document.querySelectorAll('.nav-link[data-target]').forEach(a => {
    a.addEventListener('click', e => {
      e.preventDefault();
      const target = a.dataset.target;
      showTopic(target);
      history.replaceState(null, '', '#' + target);
      const el = document.getElementById(target);
      if (el) window.scrollTo({ top: 0, behavior: 'smooth' });
      document.querySelector('.sidebar')?.classList.remove('open');
    });
  });

  // ativa pelo hash, senão o primeiro tópico
  const hash = window.location.hash.slice(1);
  if (hash && document.getElementById(hash)) showTopic(hash);
  else { const first = document.querySelector('.topic'); if (first) showTopic(first.id); }

  // toggle sidebar mobile
  document.getElementById('menuToggle')?.addEventListener('click', () => {
    document.querySelector('.sidebar')?.classList.toggle('open');
  });

  // estado das guias (expandida/colapsada) preservado entre páginas
  const GUIAS_KEY = 'guias-estado';
  function lerGuias() {
    try { return JSON.parse(localStorage.getItem(GUIAS_KEY)) || {}; } catch (e) { return {}; }
  }
  function salvarGuias(estado) {
    try { localStorage.setItem(GUIAS_KEY, JSON.stringify(estado)); } catch (e) {}
  }
  function registrar(g) {
    const estado = lerGuias();
    if (g.dataset.sec) estado[g.dataset.sec] = g.classList.contains('collapsed') ? 'c' : 'o';
    salvarGuias(estado);
  }
  // aplica a preferência salva; guias sem preferência mantêm o padrão da página
  (function aplicarGuias() {
    const estado = lerGuias();
    document.querySelectorAll('.nav-group').forEach(g => {
      const s = estado[g.dataset.sec];
      if (s === 'o') g.classList.remove('collapsed');
      else if (s === 'c') g.classList.add('collapsed');
    });
  })();

  // colapsar/expandir grupos da sidebar (individual)
  document.querySelectorAll('.nav-group-title').forEach(t => {
    t.addEventListener('click', () => {
      const g = t.parentElement;
      g.classList.toggle('collapsed');
      registrar(g);
      syncToggleAll();
    });
  });

  // botão "mostrar/esconder tudo"
  const toggleAll = document.getElementById('toggleAll');
  function anyOpen() {
    return [...document.querySelectorAll('.nav-group')].some(g => !g.classList.contains('collapsed'));
  }
  function syncToggleAll() {
    if (toggleAll) toggleAll.textContent = anyOpen() ? '− Esconder tudo' : '+ Mostrar tudo';
  }
  if (toggleAll) {
    toggleAll.addEventListener('click', () => {
      const collapse = anyOpen();   // se algum aberto -> fecha todos; senão abre todos
      document.querySelectorAll('.nav-group').forEach(g => {
        g.classList.toggle('collapsed', collapse);
        registrar(g);
      });
      syncToggleAll();
    });
    syncToggleAll();
  }

  // alternar tema claro/escuro
  document.getElementById('themeToggle')?.addEventListener('click', () => {
    const dark = document.documentElement.getAttribute('data-theme') === 'dark';
    if (dark) document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', 'dark');
    try { localStorage.setItem('tema', dark ? 'light' : 'dark'); } catch (e) {}
    window.dispatchEvent(new Event('resize'));
  });
});
