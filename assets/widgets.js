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
  ink: '#1b1c12', blue: '#3266ad', red: '#c0392b', green: '#1a7a4a',
  muted: '#66693f', line: '#cbd598', paper: '#f3f6e0',
  blueF: '#dce7f4', redF: '#f6dedb', greenF: '#dcefe4'
};
// no tema escuro, texto/eixos precisam clarear
function pal() {
  const dark = document.documentElement.getAttribute('data-theme') === 'dark';
  return {
    ...PAL,
    ink: dark ? '#ebedd6' : PAL.ink,
    muted: dark ? '#a4a87e' : PAL.muted,
    line: dark ? '#474f30' : PAL.line,
    paper: dark ? '#1f2113' : PAL.paper,
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

  // colapsar/expandir grupos da sidebar
  document.querySelectorAll('.nav-group-title').forEach(t => {
    t.addEventListener('click', () => t.parentElement.classList.toggle('collapsed'));
  });

  // alternar tema claro/escuro
  document.getElementById('themeToggle')?.addEventListener('click', () => {
    const dark = document.documentElement.getAttribute('data-theme') === 'dark';
    if (dark) document.documentElement.removeAttribute('data-theme');
    else document.documentElement.setAttribute('data-theme', 'dark');
    try { localStorage.setItem('tema', dark ? 'light' : 'dark'); } catch (e) {}
    window.dispatchEvent(new Event('resize'));
  });
});
