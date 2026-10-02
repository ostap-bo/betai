/* BetAI dashboard — читає data/latest.json і data/picks.json */
(() => {
  "use strict";

  const TZ = "Europe/Kyiv";
  const $ = (s) => document.querySelector(s);
  const esc = (v) => String(v ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  const pct = (v, d = 1) => (v == null ? "—" : `${(v * 100).toFixed(d)}%`);
  const sgn = (v, d = 2) => (v == null ? "—" : `${v > 0 ? "+" : ""}${v.toFixed(d)}`);
  const cls = (v) => (v == null || v === 0 ? "" : v > 0 ? "pos" : "neg");
  const safeUrl = (u) => (/^https?:\/\//i.test(u || "") ? u : null);

  const fmtDate = (iso, opts) => new Intl.DateTimeFormat("uk-UA", { timeZone: TZ, ...opts }).format(new Date(iso));
  const kickoff = (iso) => fmtDate(iso, { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });

  const SIDE = { home: "П1", draw: "Нічия", away: "П2", over: "ТБ", under: "ТМ" };
  const STATUS = { won: "Виграш", lost: "Програш", pending: "Очікує", void: "Повернення" };

  async function load(path) {
    const r = await fetch(`${path}?t=${Date.now()}`);
    if (!r.ok) throw new Error(`${path}: ${r.status}`);
    return r.json();
  }

  /* ── KPI ─────────────────────────────────────────────── */
  function renderKpis(s) {
    const tiles = [
      ["ROI", pct(s.roi), cls(s.roi), `${s.settled} розрах.`],
      ["Прибуток", `${sgn(s.profit)} од.`, cls(s.profit), `поставлено ${s.staked} од.`],
      ["Вінрейт", pct(s.hit_rate, 0), "", `${s.won}–${s.lost}`],
      ["Сер. коефіцієнт", s.avg_odds ?? "—", "", "розраховані"],
      ["CLV", pct(s.avg_clv), cls(s.avg_clv), "vs. останній коеф."],
      ["Очікують", s.pending, "", `усього ${s.total}`],
    ];
    $("#kpis").innerHTML = tiles.map(([l, v, c, sub]) => `
      <div class="kpi"><div class="label">${l}</div>
        <div class="value num ${c}">${esc(v)}</div><div class="sub">${esc(sub)}</div></div>`).join("");
  }

  /* ── Profit curve (SVG) ──────────────────────────────── */
  function renderCurve(curve) {
    const el = $("#curve");
    if (!curve.length) {
      el.innerHTML = `<div class="empty muted">Графік з'явиться після перших розрахованих ставок.</div>`;
      return;
    }
    const pts = [{ profit: 0 }, ...curve];
    const W = Math.max(300, el.clientWidth - 16), H = 200, P = { l: 44, r: 12, t: 12, b: 24 };
    const ys = pts.map((p) => p.profit);
    let min = Math.min(0, ...ys), max = Math.max(0, ...ys);
    if (max - min < 1) { max += 1; min -= 1; }
    const x = (i) => P.l + (i / (pts.length - 1)) * (W - P.l - P.r);
    const y = (v) => P.t + (1 - (v - min) / (max - min)) * (H - P.t - P.b);
    const line = pts.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.profit).toFixed(1)}`).join("");
    const area = `${line}L${x(pts.length - 1)},${y(0)}L${x(0)},${y(0)}Z`;
    const last = ys[ys.length - 1];
    const color = last >= 0 ? "var(--accent)" : "var(--neg)";
    const ticks = [min, (min + max) / 2, max].map((v) =>
      `<text x="${P.l - 6}" y="${y(v) + 4}" text-anchor="end" font-size="11" fill="var(--muted)">${v.toFixed(1)}</text>`).join("");
    el.innerHTML = `
      <svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Крива прибутку">
        <line x1="${P.l}" x2="${W - P.r}" y1="${y(0)}" y2="${y(0)}" stroke="var(--border)" stroke-dasharray="4 4"/>
        <path d="${area}" fill="${color}" opacity=".08"/>
        <path d="${line}" fill="none" stroke="${color}" stroke-width="2" vector-effect="non-scaling-stroke"/>
        ${ticks}
        <text x="${P.l}" y="${H - 6}" font-size="11" fill="var(--muted)">${esc(curve[0].date)}</text>
        <text x="${W - P.r}" y="${H - 6}" font-size="11" fill="var(--muted)" text-anchor="end">${esc(curve[curve.length - 1].date)}</text>
      </svg>`;
    $("#curve-note").textContent = `${curve.length} ставок · ${sgn(last)} од.`;
  }

  /* ── Деталі матчу (спільні для карток і списку) ──────── */
  function probTable(ev) {
    const rows = [["Модель", ev.model_probs], ["Ринок", ev.market_probs], ["AI", ev.ai ? ev.final_probs : null]]
      .filter(([, p]) => p && Object.keys(p).length);
    const line = ev.odds?.totals?.line ?? 2.5;
    const cols = [["home", "П1"], ["draw", "X"], ["away", "П2"], ["over", `ТБ ${line}`], ["under", `ТМ ${line}`]];
    const odds = { home: ev.odds?.h2h?.home, draw: ev.odds?.h2h?.draw, away: ev.odds?.h2h?.away,
      over: ev.odds?.totals?.over, under: ev.odds?.totals?.under };
    return `<table class="probs"><thead><tr><th></th>${cols.map(([, l]) => `<th>${l}</th>`).join("")}</tr></thead><tbody>
      <tr><td>Коеф.</td>${cols.map(([k]) => `<td class="num">${odds[k]?.best ?? "—"}</td>`).join("")}</tr>
      ${rows.map(([n, p]) => `<tr class="${n === "AI" ? "ai" : ""}"><td>${n}</td>${cols.map(([k]) => `<td class="num">${pct(p[k], 0)}</td>`).join("")}</tr>`).join("")}
    </tbody></table>`;
  }

  const formRow = (f) => `<div class="form">${(f || []).map((g) =>
    `<span class="${g.result}" title="${esc(`${g.date} ${g.venue === "H" ? "vs" : "@"} ${g.opponent} ${g.score}`)}">${g.result === "W" ? "В" : g.result === "D" ? "Н" : "П"}</span>`).join("")}</div>`;

  function details(ev) {
    const ai = ev.ai || {};
    const parts = [];
    parts.push(`<div>${probTable(ev)}</div>`);
    if (ev.expected_goals) {
      const scores = (ev.likely_scores || []).map((s) => `${s.score} (${pct(s.p, 0)})`).join(", ");
      parts.push(`<div><h3>Очікувані голи</h3><span class="num">${ev.expected_goals.home} – ${ev.expected_goals.away}</span>
        <span class="muted small"> · ймовірні рахунки: ${esc(scores)}</span></div>`);
    }
    if (ai.key_factors?.length) parts.push(`<div><h3>Ключові фактори</h3><ul>${ai.key_factors.map((f) => `<li>${esc(f)}</li>`).join("")}</ul></div>`);
    const abs = ai.absences?.length ? ai.absences : (ev.injuries || []).map((i) => ({ team: i.team, player: i.player, status: i.type, impact: "" }));
    if (abs.length) parts.push(`<div><h3>Відсутні гравці</h3>${abs.map((a) =>
      `<span class="chip ${esc(a.impact)}">${esc(a.player)} · ${esc(a.team)}${a.status ? ` · ${esc(a.status)}` : ""}</span>`).join("")}</div>`);
    if (ev.form) parts.push(`<div class="two-col"><h3>Форма (останні матчі)</h3>
      <div class="small muted">${esc(ev.home)}</div>${formRow(ev.form.home)}
      <div class="small muted" style="margin-top:6px">${esc(ev.away)}</div>${formRow(ev.form.away)}</div>`);
    if (ev.h2h?.length) parts.push(`<div><h3>Особисті зустрічі</h3>${ev.h2h.slice().reverse().map((g) =>
      `<span class="chip">${esc(g.date.slice(0, 7))} ${esc(g.home)} ${esc(g.score)} ${esc(g.away)}</span>`).join("")}</div>`);
    if (ai.risks?.length) parts.push(`<div><h3>Ризики</h3><ul>${ai.risks.map((r) => `<li>${esc(r)}</li>`).join("")}</ul></div>`);
    if (ev.news?.length) parts.push(`<div><h3>Новини</h3><ul>${ev.news.slice(0, 5).map((n) => {
      const u = safeUrl(n.link);
      return `<li>${u ? `<a href="${esc(u)}" target="_blank" rel="noopener noreferrer">${esc(n.title)}</a>` : esc(n.title)} <span class="muted small">${esc(n.source)}</span></li>`;
    }).join("")}</ul></div>`);
    if (ai.sources?.length) parts.push(`<div><h3>Джерела AI</h3>${ai.sources.map((s) => {
      const u = safeUrl(s);
      return u ? `<a class="chip" href="${esc(u)}" target="_blank" rel="noopener noreferrer">${esc(new URL(u).hostname)}</a>` : `<span class="chip">${esc(s)}</span>`;
    }).join("")}</div>`);
    if (ev.ai_error) parts.push(`<div class="neg small">AI-помилка: ${esc(ev.ai_error)}</div>`);
    return `<details><summary>Детальний аналіз</summary><div class="detail">${parts.join("")}</div></details>`;
  }

  const confBar = (c) => c == null ? "" :
    `<div class="conf" aria-label="Впевненість ${c}/10">${Array.from({ length: 10 }, (_, i) => `<i class="${i < c ? "on" : ""}"></i>`).join("")}</div>`;

  /* ── Рекомендації ────────────────────────────────────── */
  function renderPicks(picks, events) {
    const now = Date.now();
    const open = picks.filter((p) => p.status === "pending" && new Date(p.commence_time) > now - 2 * 3600e3)
      .sort((a, b) => a.commence_time.localeCompare(b.commence_time));
    const evById = Object.fromEntries(events.map((e) => [e.id, e]));
    $("#c-picks").textContent = open.length;
    if (!open.length) {
      $("#tab-picks").innerHTML = `<div class="card empty-state">Зараз немає ставок з достатнім value.
        Це нормально: платформа пропускає матчі, де перевага не підтверджена.</div>`;
      return;
    }
    $("#tab-picks").innerHTML = `<div class="grid">${open.map((p) => {
      const ev = evById[p.event_id];
      const summary = ev?.ai?.summary;
      return `<article class="card pick">
        <div class="pick-top"><span>${esc(p.league_name)}</span><span>${kickoff(p.commence_time)}</span></div>
        <div class="teams">${esc(p.home)}<span class="vs">—</span>${esc(p.away)}</div>
        <div class="bet">
          <div><div class="sel">${esc(p.label)}</div><div class="small muted">ставка ${p.stake} од.</div></div>
          <div><div class="price num">${p.price.toFixed(2)}</div><div class="book">${esc(p.book)}</div></div>
        </div>
        <div class="metrics">
          <div class="metric"><div class="label">Ймовірність</div><div class="v num">${pct(p.prob)}</div></div>
          <div class="metric"><div class="label">Value</div><div class="v num pos">${sgn(p.edge * 100, 1)}%</div></div>
          <div class="metric"><div class="label">Впевненість</div><div class="v num">${p.confidence ?? "—"}/10</div>${confBar(p.confidence)}</div>
        </div>
        ${summary ? `<p class="reason">${esc(summary)}</p>` : ""}
        ${p.reasoning ? `<p class="reason muted">${esc(p.reasoning)}</p>` : ""}
        ${p.clv != null ? `<div class="small muted">CLV зараз: <span class="num ${cls(p.clv)}">${pct(p.clv)}</span></div>` : ""}
        ${ev ? details(ev) : ""}
      </article>`;
    }).join("")}</div>`;
  }

  /* ── Усі матчі ───────────────────────────────────────── */
  function renderMatches(events) {
    $("#c-matches").textContent = events.length;
    if (!events.length) { $("#tab-matches").innerHTML = `<div class="card empty-state">Немає матчів у найближчому вікні.</div>`; return; }
    $("#tab-matches").innerHTML = `<div class="card">${events.map((ev) => {
      const best = (ev.final_candidates || [])[0];
      const verdict = ev.verdict === "bet" ? `<span class="verdict bet">Ставка</span>` :
        ev.ai ? `<span class="verdict skip">Пропуск</span>` : `<span class="verdict skip">Без AI</span>`;
      return `<div class="match-row">
        <div class="match-line">
          <div><div class="small muted">${esc(ev.league_name)} · ${kickoff(ev.commence_time)}</div>
            <strong>${esc(ev.home)} — ${esc(ev.away)}</strong></div>
          <div class="small">${verdict}${best ? ` · найкраще: ${esc(best.label)} @ <span class="num">${best.price}</span>
            (<span class="num ${cls(best.edge)}">${sgn(best.edge * 100, 1)}%</span>)` : ""}</div>
        </div>
        ${ev.ai?.summary ? `<p class="reason small" style="margin:6px 0 0">${esc(ev.ai.summary)}</p>` : ""}
        ${details(ev)}
      </div>`;
    }).join("")}</div>`;
  }

  /* ── Історія ─────────────────────────────────────────── */
  function renderHistory(picks) {
    const done = picks.filter((p) => p.status !== "pending").sort((a, b) => b.commence_time.localeCompare(a.commence_time));
    $("#c-history").textContent = done.length;
    if (!done.length) { $("#tab-history").innerHTML = `<div class="card empty-state">Історія порожня — ставки ще не розраховані.</div>`; return; }
    $("#tab-history").innerHTML = `<div class="card table-wrap"><table class="data">
      <thead><tr><th>Дата</th><th>Матч</th><th>Ставка</th><th class="r">Коеф.</th><th class="r">Сума</th>
        <th>Рахунок</th><th>Результат</th><th class="r">P/L</th><th class="r">CLV</th></tr></thead>
      <tbody>${done.map((p) => `<tr>
        <td class="small muted">${fmtDate(p.commence_time, { day: "2-digit", month: "2-digit", year: "2-digit" })}</td>
        <td>${esc(p.home)} — ${esc(p.away)}</td><td>${esc(p.label)}</td>
        <td class="r num">${p.price.toFixed(2)}</td><td class="r num">${p.stake}</td>
        <td class="num">${esc(p.score ?? "—")}</td>
        <td><span class="status ${p.status}">${STATUS[p.status] || p.status}</span></td>
        <td class="r num ${cls(p.profit)}">${sgn(p.profit)}</td>
        <td class="r num ${cls(p.clv)}">${pct(p.clv)}</td></tr>`).join("")}</tbody></table></div>`;
  }

  /* ── Статистика ──────────────────────────────────────── */
  function breakdownTable(title, rows) {
    return `<div class="card"><h2>${title}</h2>${rows.length ? `<div class="table-wrap"><table class="data">
      <thead><tr><th></th><th class="r">Ставок</th><th class="r">Виграно</th><th class="r">P/L</th><th class="r">ROI</th></tr></thead>
      <tbody>${rows.map((r) => `<tr><td>${esc(r.name)}</td><td class="r num">${r.n}</td><td class="r num">${r.won}</td>
        <td class="r num ${cls(r.profit)}">${sgn(r.profit)}</td><td class="r num ${cls(r.roi)}">${pct(r.roi)}</td></tr>`).join("")}</tbody>
      </table></div>` : `<div class="empty-state">Немає даних</div>`}</div>`;
  }

  function renderStats(s, latest) {
    const ai = latest.ai || {};
    const aiInfo = ai.mode === "ai" || ai.mode === "claude"
      ? `${esc(ai.provider || "Claude")} · модель: <span class="mono">${esc(ai.model)}</span> · викликів: ${ai.calls ?? 0} · токенів: ${(ai.input_tokens ?? 0) + (ai.output_tokens ?? 0)} · пошуків: ${ai.web_searches ?? 0}`
      : ai.mode === "demo" ? "Демо-режим (без AI)" : "AI вимкнено — лише статистична модель";
    const st = latest.settings || {};
    $("#tab-stats").innerHTML = `<div class="two">${breakdownTable("За ринком", s.by_market)}${breakdownTable("За лігою", s.by_league)}</div>
      <div class="card" style="margin-top:14px;padding:14px 16px">
        <h2>Останній запуск</h2>
        <p class="small">${aiInfo}</p>
        <p class="small muted">Фільтри: value ≥ ${pct(st.min_edge, 0)}, коефіцієнт ${st.min_odds}–${st.max_odds}, впевненість ≥ ${st.min_confidence}/10 · новин зібрано: ${latest.news_count ?? 0} · тривалість: ${latest.duration_sec ?? "—"} с</p>
        <p class="small muted">CLV (closing line value) показує, чи кращий наш коефіцієнт за пізніший ринковий. Стабільно позитивний CLV — найнадійніша ознака реальної переваги, навіть коли результати на короткій дистанції коливаються.</p>
      </div>`;
  }

  /* ── Tabs ────────────────────────────────────────────── */
  function initTabs() {
    const btns = document.querySelectorAll(".tabs button");
    const show = (name) => {
      btns.forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
      document.querySelectorAll(".tab").forEach((t) => (t.hidden = t.id !== `tab-${name}`));
      try { localStorage.setItem("betai-tab", name); } catch { /* ignore */ }
    };
    btns.forEach((b) => b.addEventListener("click", () => show(b.dataset.tab)));
    let saved = null;
    try { saved = localStorage.getItem("betai-tab"); } catch { /* ignore */ }
    if (saved && document.getElementById(`tab-${saved}`)) show(saved);
  }

  async function main() {
    initTabs();
    try {
      const [latest, store] = await Promise.all([load("data/latest.json"), load("data/picks.json")]);
      $("#updated").textContent = `Оновлено: ${fmtDate(latest.updated_at, { day: "numeric", month: "long", hour: "2-digit", minute: "2-digit" })} (Київ)`;
      $("#demo-banner").hidden = !latest.demo;
      const ai = latest.ai || {};
      $("#badges").innerHTML = [
        latest.demo ? `<span class="badge warn">ДЕМО</span>` : "",
        (ai.mode === "ai" || ai.mode === "claude") ? `<span class="badge ok">${esc(ai.provider || "Claude")} · ${esc(ai.model)}</span>` : ai.mode === "off" ? `<span class="badge">AI вимкнено</span>` : "",
        `<span class="badge">${latest.events.length} матчів</span>`,
      ].join("");
      renderKpis(store.stats);
      renderCurve(store.stats.curve || []);
      let rt; window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => renderCurve(store.stats.curve || []), 150); });
      renderPicks(store.picks || [], latest.events || []);
      renderMatches(latest.events || []);
      renderHistory(store.picks || []);
      renderStats(store.stats, latest);
    } catch (err) {
      console.error(err);
      $("#updated").textContent = "Дані ще не згенеровані";
      $("#tab-picks").innerHTML = `<div class="card empty-state">Не вдалося завантажити дані. Запустіть <span class="mono">python -m betai --demo</span> або GitHub Action.</div>`;
    }
  }

  main();
})();
