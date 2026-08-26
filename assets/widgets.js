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

  // colapsar/expandir grupos da sidebar (individual)
  document.querySelectorAll('.nav-group-title').forEach(t => {
    t.addEventListener('click', () => { t.parentElement.classList.toggle('collapsed'); syncToggleAll(); });
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
      document.querySelectorAll('.nav-group').forEach(g => g.classList.toggle('collapsed', collapse));
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
