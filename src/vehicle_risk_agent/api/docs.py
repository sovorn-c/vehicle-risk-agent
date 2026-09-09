# ruff: noqa: E501
"""Dependency-free branded API documentation page."""

DOCS_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Auditable Vehicle Risk Assessment Agent API documentation">
  <title>Vehicle Risk / API docs</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #102131;
      --muted: #637184;
      --line: #dce3e8;
      --paper: #f4f7f8;
      --surface: #ffffff;
      --deep: #102131;
      --deep-soft: #193b51;
      --mint: #b8f397;
      --mint-ink: #19301b;
      --cyan: #83d9e5;
      --coral: #f8785d;
      --amber: #e7ae4e;
      --code: #0b1722;
      --shadow: 0 18px 50px rgba(16, 33, 49, .09);
    }

    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body {
      position: relative;
      margin: 0;
      overflow-x: hidden;
      background: var(--paper);
      color: var(--ink);
      font: 15px/1.55 "Avenir Next", "Segoe UI", sans-serif;
    }
    body::before {
      position: fixed;
      z-index: -1;
      inset: 0;
      background-image: linear-gradient(rgba(16, 33, 49, .025) 1px, transparent 1px), linear-gradient(90deg, rgba(16, 33, 49, .025) 1px, transparent 1px);
      background-size: 32px 32px;
      content: "";
      mask-image: linear-gradient(to bottom, black, transparent 72%);
    }
    a { color: inherit; }
    button, input { font: inherit; }
    button { cursor: pointer; }
    code, pre, .mono { font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace; }
    :focus-visible { outline: 3px solid var(--cyan); outline-offset: 3px; }

    .skip-link {
      position: fixed;
      z-index: 5;
      top: 10px;
      left: 10px;
      padding: 8px 12px;
      background: var(--mint);
      color: var(--mint-ink);
      font-weight: 800;
      transform: translateY(-150%);
    }
    .skip-link:focus { transform: translateY(0); }

    .app {
      display: grid;
      grid-template-columns: 258px minmax(0, 1fr);
      min-height: 100vh;
    }
    .sidebar {
      position: sticky;
      top: 0;
      align-self: start;
      height: 100vh;
      padding: 28px 20px;
      background: var(--deep);
      color: #edf4f3;
      display: flex;
      flex-direction: column;
      gap: 34px;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 11px;
      text-decoration: none;
    }
    .brand-mark {
      display: grid;
      width: 36px;
      height: 36px;
      place-items: center;
      background: var(--mint);
      color: var(--mint-ink);
      font-weight: 900;
      letter-spacing: -.1em;
      transform: rotate(-5deg);
    }
    .brand-copy strong { display: block; font-size: 14px; letter-spacing: -.02em; }
    .brand-copy span { color: #9db0bb; font-size: 11px; }
    .nav-label {
      margin: 0 0 9px 10px;
      color: #8299a7;
      font-size: 10px;
      font-weight: 700;
      letter-spacing: .16em;
      text-transform: uppercase;
    }
    .nav { display: grid; gap: 3px; }
    .nav a {
      display: block;
      padding: 9px 10px;
      border-radius: 7px;
      color: #bbcad1;
      font-size: 13px;
      text-decoration: none;
    }
    .nav a:hover, .nav a:focus-visible { background: var(--deep-soft); color: white; outline: none; }
    .sidebar-foot {
      margin-top: auto;
      padding: 14px 10px;
      border: 1px solid #345064;
      color: #a6b6bd;
      font-size: 12px;
    }
    .sidebar-foot strong { display: block; color: var(--mint); font-size: 12px; }

    main { min-width: 0; }
    .topbar {
      position: sticky;
      z-index: 2;
      top: 0;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      padding: 16px clamp(20px, 4vw, 62px);
      border-bottom: 1px solid rgba(220, 227, 232, .9);
      background: rgba(244, 247, 248, .92);
      backdrop-filter: blur(14px);
    }
    .crumb { color: var(--muted); font-size: 12px; }
    .crumb b { color: var(--ink); }
    .top-actions { display: flex; align-items: center; gap: 14px; }
    .ready {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 6px 10px;
      color: var(--muted);
      font-size: 12px;
      background: var(--surface);
    }
    .ready::before {
      width: 7px;
      height: 7px;
      border-radius: 50%;
      background: var(--coral);
      content: "";
    }
    .ready.online::before { background: #50a96a; }
    .top-link {
      color: var(--ink);
      font-size: 12px;
      font-weight: 800;
      text-decoration: none;
    }
    .top-link:hover { text-decoration: underline; }

    .content { width: min(1200px, 100%); margin: 0 auto; padding: 62px clamp(20px, 4vw, 62px) 90px; }
    .hero {
      position: relative;
      display: grid;
      grid-template-columns: minmax(0, 1.18fr) minmax(280px, .82fr);
      gap: 44px;
      align-items: end;
      padding-bottom: 62px;
    }
    .hero::after {
      position: absolute;
      right: 12%;
      bottom: 28px;
      width: 90px;
      height: 9px;
      background: var(--coral);
      content: "";
      transform: rotate(-4deg);
    }
    .eyebrow {
      margin: 0 0 16px;
      color: var(--coral);
      font-size: 11px;
      font-weight: 800;
      letter-spacing: .16em;
      text-transform: uppercase;
    }
    h1, h2, h3 { margin: 0; text-wrap: balance; }
    h1 {
      max-width: 760px;
      font: 500 clamp(44px, 7vw, 88px)/.93 Georgia, "Times New Roman", serif;
      letter-spacing: -.07em;
    }
    .hero-copy { max-width: 650px; margin: 23px 0 0; color: #536476; font-size: 17px; }
    .hero-aside {
      padding: 22px;
      border: 1px solid var(--ink);
      background: var(--deep);
      color: #edf4f3;
      box-shadow: var(--shadow);
    }
    .hero-aside .mono { color: var(--mint); font-size: 12px; }
    .hero-aside p { margin: 13px 0 0; color: #c1cfd3; font-size: 13px; }
    .signal-row { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 17px; }
    .signal { padding: 5px 8px; border: 1px solid #3d5b6b; color: #c9d9dd; font: 10px/1 "SFMono-Regular", Consolas, monospace; text-transform: uppercase; }
    .signal strong { color: var(--cyan); }

    .section { scroll-margin-top: 90px; padding: 40px 0; border-top: 1px solid var(--line); }
    .section-heading { display: flex; align-items: end; justify-content: space-between; gap: 20px; margin-bottom: 21px; }
    .section-heading h2 { font-size: 25px; letter-spacing: -.045em; }
    .section-heading p { max-width: 475px; margin: 0; color: var(--muted); font-size: 13px; }

    .journey { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
    .journey-card {
      position: relative;
      min-height: 163px;
      padding: 18px;
      border: 1px solid var(--line);
      background: var(--surface);
      transition: border-color .2s ease, box-shadow .2s ease, transform .2s ease;
    }
    .journey-card::after { position: absolute; top: 27px; right: -13px; width: 16px; height: 1px; background: var(--coral); content: ""; }
    .journey-card:last-child::after { display: none; }
    .journey-card:hover { border-color: var(--ink); box-shadow: var(--shadow); transform: translateY(-3px); }
    .journey-index { color: var(--coral); font: 800 10px/1 "SFMono-Regular", Consolas, monospace; letter-spacing: .12em; }
    .journey-card h3 { margin-top: 26px; font-size: 16px; }
    .journey-card p { margin: 6px 0 0; color: var(--muted); font-size: 12px; }
    .journey-card code { display: block; margin-top: 14px; color: #466176; font-size: 10px; }
    .journey-card:nth-child(2) { background: var(--cyan); border-color: var(--cyan); }
    .journey-card:nth-child(2) p, .journey-card:nth-child(2) code { color: #244c5a; }
    .journey-card:nth-child(3) { background: var(--mint); border-color: var(--mint); }
    .journey-card:nth-child(3) p, .journey-card:nth-child(3) code { color: #365235; }

    .principles { display: grid; grid-template-columns: 1.1fr .9fr; gap: 14px; }
    .principle { padding: 23px; border: 1px solid var(--line); background: var(--surface); }
    .principle h3 { font-size: 17px; }
    .principle p { margin: 9px 0 0; color: var(--muted); font-size: 13px; }
    .principle.primary { grid-row: span 2; background: var(--deep); border-color: var(--deep); color: #edf4f3; }
    .principle.primary p { color: #bbcad1; }
    .principle.primary .kicker { color: var(--mint); }
    .kicker { display: block; margin-bottom: 25px; color: var(--coral); font: 800 10px/1 "SFMono-Regular", Consolas, monospace; letter-spacing: .12em; text-transform: uppercase; }

    .endpoint-tools { display: flex; align-items: center; gap: 10px; margin-bottom: 18px; }
    .search {
      width: min(410px, 100%);
      border: 1px solid var(--line);
      padding: 11px 12px;
      background: var(--surface);
      color: var(--ink);
      outline: none;
    }
    .search:focus { border-color: var(--ink); box-shadow: 0 0 0 3px rgba(131, 217, 229, .35); }
    .endpoint-count { color: var(--muted); font-size: 12px; }
    .endpoint-group { margin-top: 29px; }
    .endpoint-group h3 { margin-bottom: 9px; font-size: 14px; }
    .endpoint-list { display: grid; gap: 10px; }
    .endpoint { overflow: hidden; border: 1px solid var(--line); background: var(--surface); }
    .endpoint summary { display: flex; align-items: center; gap: 12px; padding: 15px 17px; cursor: pointer; list-style: none; }
    .endpoint summary::-webkit-details-marker { display: none; }
    .endpoint summary:hover { background: #fbfcfd; }
    .method {
      display: inline-grid;
      min-width: 52px;
      padding: 4px 6px;
      place-items: center;
      border-radius: 4px;
      background: var(--mint);
      color: var(--mint-ink);
      font: 800 10px/1 "SFMono-Regular", Consolas, monospace;
    }
    .method.post, .method.put, .method.patch, .method.delete { background: var(--coral); color: #351710; }
    .path { overflow: hidden; font-size: 13px; font-weight: 700; text-overflow: ellipsis; white-space: nowrap; }
    .summary-text { margin-left: auto; color: var(--muted); font-size: 12px; text-align: right; }
    .operation-badges { display: flex; flex-wrap: wrap; gap: 5px; margin-left: auto; }
    .badge { border: 1px solid var(--line); padding: 3px 6px; color: var(--muted); font: 9px/1 "SFMono-Regular", Consolas, monospace; text-transform: uppercase; white-space: nowrap; }
    .badge.safe { border-color: #acd997; color: #3b6b3a; }
    .badge.review { border-color: #f2bfad; color: #9e4838; }
    .endpoint-body { display: grid; grid-template-columns: minmax(0, 1fr) minmax(280px, .8fr); gap: 19px; padding: 0 17px 17px; }
    .endpoint-description { color: var(--muted); font-size: 13px; }
    .endpoint-description p { margin: 0 0 14px; }
    .try-form { display: flex; flex-wrap: wrap; gap: 8px; }
    .param-input { min-width: 200px; flex: 1; border: 1px solid var(--line); padding: 9px 10px; color: var(--ink); background: var(--paper); }
    .run, .copy {
      border: 1px solid var(--ink);
      padding: 9px 12px;
      background: var(--ink);
      color: white;
      font-size: 12px;
      font-weight: 700;
    }
    .copy { border-color: var(--line); background: var(--surface); color: var(--ink); }
    .run:hover, .run:focus-visible { background: var(--deep-soft); }
    .copy:hover, .copy:focus-visible { border-color: var(--ink); }
    .response { min-height: 111px; margin: 0; overflow: auto; padding: 13px; background: var(--code); color: #d4e5e7; font-size: 11px; white-space: pre-wrap; word-break: break-word; }
    .response[data-state="error"] { color: #ffb9aa; }
    .response-label { display: block; margin-bottom: 6px; color: var(--muted); font-size: 10px; letter-spacing: .1em; text-transform: uppercase; }
    .reference-note { margin-top: 17px; padding: 14px 16px; border-left: 4px solid var(--amber); background: #fffaf1; color: #6a593c; font-size: 13px; }
    .reference-note a { color: var(--ink); font-weight: 800; }
    .no-results { padding: 15px; border: 1px dashed var(--line); color: var(--muted); font-size: 13px; }

    .footer { display: flex; justify-content: space-between; gap: 20px; padding-top: 28px; border-top: 1px solid var(--line); color: var(--muted); font-size: 12px; }
    .footer a { color: var(--ink); font-weight: 800; }

    @media (max-width: 1000px) {
      .app { grid-template-columns: 1fr; }
      .sidebar { position: static; height: auto; padding: 16px 20px; gap: 15px; }
      .nav-label, .sidebar-foot { display: none; }
      .nav { display: flex; overflow-x: auto; gap: 3px; }
      .nav a { white-space: nowrap; }
      .hero { grid-template-columns: 1fr; }
      .journey { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .journey-card:nth-child(2)::after { display: none; }
    }
    @media (max-width: 680px) {
      .topbar { align-items: flex-start; flex-direction: column; gap: 8px; }
      .top-actions { width: 100%; justify-content: space-between; }
      .content { padding-top: 40px; }
      h1 { font-size: clamp(42px, 15vw, 68px); }
      .section-heading, .footer { align-items: flex-start; flex-direction: column; }
      .principles { grid-template-columns: 1fr; }
      .principle.primary { grid-row: auto; }
      .journey { grid-template-columns: 1fr; }
      .journey-card::after { display: none; }
      .endpoint-body { grid-template-columns: 1fr; }
      .endpoint summary { align-items: flex-start; flex-wrap: wrap; }
      .summary-text, .operation-badges { width: 100%; margin-left: 64px; text-align: left; }
      .operation-badges { margin-top: 3px; }
    }
    @media (prefers-reduced-motion: reduce) {
      html { scroll-behavior: auto; }
      *, *::before, *::after { transition-duration: .01ms !important; animation-duration: .01ms !important; }
    }
  </style>
</head>
<body>
  <a class="skip-link" href="#top">Skip to content</a>
  <div class="app">
    <aside class="sidebar">
      <a class="brand" href="#top" aria-label="Vehicle Risk Agent home">
        <span class="brand-mark">VR</span>
        <span class="brand-copy"><strong>Vehicle Risk</strong><span>API documentation</span></span>
      </a>
      <div>
        <p class="nav-label">Explore</p>
        <nav class="nav" aria-label="Documentation sections">
          <a href="#top">Overview</a>
          <a href="#journey">Workflow</a>
          <a href="#principles">Guardrails</a>
          <a href="#contract">Contract explorer</a>
          <a href="#operations">Run locally</a>
        </nav>
      </div>
      <div class="sidebar-foot"><strong>Decisions you can defend.</strong>Evidence remains traceable from MCP through policy, risk, report, and review.</div>
    </aside>

    <main id="top">
      <header class="topbar">
        <div class="crumb"><b>Developer docs</b> / v0.1.0</div>
        <div class="top-actions"><span class="ready" id="ready-status">Checking API…</span><a class="top-link" href="/openapi.json">OpenAPI JSON ↗</a><a class="top-link" href="/reference">Full reference ↗</a></div>
      </header>

      <div class="content">
        <section class="hero" aria-labelledby="page-title">
          <div>
            <p class="eyebrow">Vehicle Risk Assessment Agent</p>
            <h1 id="page-title">Make risk decisions you can defend.</h1>
            <p class="hero-copy">An auditable workflow for evidence-aware vehicle assessment. Explore how MCP evidence becomes a grounded report, why uncertainty stays visible, and where a human Reviewer must decide.</p>
          </div>
          <div class="hero-aside">
            <span class="mono">Evidence → Policy → Risk → Review</span>
            <p>The API orchestrates a typed workflow. It does not turn missing evidence into a clean result, or approval into purchase authorization.</p>
            <div class="signal-row"><span class="signal"><strong>MCP-only evidence</strong></span><span class="signal"><strong>Policy citations</strong></span><span class="signal"><strong>Deterministic risk</strong></span><span class="signal"><strong>Human approval</strong></span></div>
          </div>
        </section>

        <section class="section" id="journey" aria-labelledby="journey-title">
          <div class="section-heading"><h2 id="journey-title">Follow the decision trail</h2><p>Four bounded stages make the system inspectable instead of magical.</p></div>
          <div class="journey">
            <article class="journey-card"><span class="journey-index">01 / EVIDENCE</span><h3>Gather through MCP</h3><p>Vehicle facts, provenance, conflicts, confidence, and synthetic notices stay attached.</p><code>lookup_vehicle → explain_field</code></article>
            <article class="journey-card"><span class="journey-index">02 / POLICY</span><h3>Retrieve citations</h3><p>One pinned policy corpus supports each policy claim, or the system abstains.</p><code>retrieve → cite passage</code></article>
            <article class="journey-card"><span class="journey-index">03 / RISK</span><h3>Calculate outside prompts</h3><p>Versioned Python rules produce reproducible factors, bands, and sufficiency outcomes.</p><code>score → classify</code></article>
            <article class="journey-card"><span class="journey-index">04 / REVIEW</span><h3>Pause for a Reviewer</h3><p>A human approves release, rejects, or requests bounded reinvestigation.</p><code>draft → decision</code></article>
          </div>
        </section>

        <section class="section" id="principles" aria-labelledby="principles-title">
          <div class="section-heading"><h2 id="principles-title">Guardrails are part of the contract</h2><p>The useful answer is not always a score. This API keeps the limits visible.</p></div>
          <div class="principles">
            <article class="principle primary"><span class="kicker">Operating thesis</span><h3>Uncertainty is a first-class result.</h3><p>UNKNOWN, UNRESOLVED, ABSENT, synthetic, and unavailable evidence retain their meaning all the way to the report. The workflow abstains instead of filling gaps with confidence.</p></article>
            <article class="principle"><span class="kicker">01 / Boundary</span><h3>MCP-only evidence.</h3><p>The Agent never reaches into the upstream Pipeline database or API. Evidence enters through the typed Vehicle Intelligence MCP Server.</p></article>
            <article class="principle"><span class="kicker">02 / Decision</span><h3>Approval releases a report.</h3><p><strong>Synthetic demonstration data</strong> remains clearly labelled, and only a Reviewer can release a draft. That action does not authorize a purchase or certify legal, financial, mechanical, or insurance safety.</p></article>
          </div>
        </section>

        <section class="section" id="contract" aria-labelledby="contract-title">
          <div class="section-heading"><h2 id="contract-title">Contract explorer</h2><p>Generated from the same OpenAPI document used by clients and tooling. Safe GET examples run here; complete authenticated operations live in the reference.</p></div>
          <div class="endpoint-tools"><label class="mono" for="endpoint-search" style="position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0)">Filter endpoints</label><input class="search" id="endpoint-search" type="search" placeholder="Filter endpoints…"><span class="endpoint-count" id="endpoint-count">Loading endpoints…</span></div>
          <div id="endpoint-list" aria-live="polite"><div class="no-results">Loading the API contract…</div></div>
          <div class="reference-note">State-changing, reviewer-only, policy-maintainer, operator, and server-sent event operations are documented but not executed from this page. Use the <a href="/reference">full reference UI ↗</a> when you need to authenticate and exercise the complete contract.</div>
        </section>

        <section class="section" id="operations" aria-labelledby="operations-title">
          <div class="section-heading"><h2 id="operations-title">Run it locally</h2><p>The supported demonstration is local and deterministic. Hosted vehicle evidence is optional and never a silent fallback.</p></div>
          <div class="hero-aside"><span class="mono">docker compose up -d --build</span><p>Then open <a href="/docs" style="color:var(--mint)">localhost:8001/docs</a>. Seeded evidence is synthetic demonstration data. It is not live NZTA, PPSR, Police, insurer, owner, or dealer data.</p></div>
        </section>

        <footer class="footer"><span>Vehicle Risk Assessment Agent · API v0.1.0</span><span><a href="/openapi.json">OpenAPI JSON</a> · <a href="/reference">Full reference</a></span></footer>
      </div>
    </main>
  </div>

  <script>
    const examples = {
      assessment_id: 'assessment-id',
      policy_id: 'policy-id',
      source_id: 'source-id',
      snapshot_id: 'snapshot-id',
      corpus_id: 'corpus-id',
      run_number: '1',
      observation_id: 'observation-id'
    };
    const endpointRoot = document.getElementById('endpoint-list');
    const search = document.getElementById('endpoint-search');
    const endpointCount = document.getElementById('endpoint-count');
    let operations = [];

    const escapeHtml = (value) => String(value ?? '').replace(/[&<>"']/g, (char) => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
    }[char]));

    const isSafeRead = (operation) => operation.method.toLowerCase() === 'get' && !operation.path.endsWith('/events') && !operation.path.includes('/evidence/observations/');

    const groupFor = (operation) => {
      const tag = operation.tags?.[0];
      if (tag) return tag;
      if (operation.path === '/health' || operation.path === '/ready') return 'Operational';
      return 'Other';
    };

    const renderResponse = (node, status, payload) => {
      node.dataset.state = status >= 400 || status === 0 ? 'error' : 'ok';
      node.textContent = `${status || 'network error'}\n\n${JSON.stringify(payload, null, 2)}`;
    };

    const operationUrl = (operation, card) => {
      let path = operation.path;
      card.querySelectorAll('[data-param]').forEach((input) => {
        const value = input.value.trim();
        if (input.dataset.location === 'query') {
          if (value) path += `${path.includes('?') ? '&' : '?'}${encodeURIComponent(input.dataset.param)}=${encodeURIComponent(value)}`;
        } else {
          path = path.replace(`{${input.dataset.param}}`, encodeURIComponent(value));
        }
      });
      return path;
    };

    const runReadOnly = async (button) => {
      const operation = operations[Number(button.dataset.operation)];
      if (!isSafeRead(operation)) return;
      const card = button.closest('.endpoint');
      const response = card.querySelector('.response');
      response.textContent = 'Requesting…';
      response.dataset.state = 'loading';
      try {
        const result = await fetch(operationUrl(operation, card), { headers: { Accept: 'application/json' } });
        const text = await result.text();
        let payload;
        try { payload = JSON.parse(text); } catch { payload = text; }
        renderResponse(response, result.status, payload);
      } catch (error) {
        renderResponse(response, 0, { error: 'The read-only request could not be completed.' });
      }
    };

    const copyCurl = async (button) => {
      const operation = operations[Number(button.dataset.operation)];
      const card = button.closest('.endpoint');
      const command = `curl -i -X ${operation.method.toUpperCase()} "${window.location.origin}${operationUrl(operation, card)}"`;
      try {
        await navigator.clipboard.writeText(command);
        const original = button.textContent;
        button.textContent = 'Copied';
        setTimeout(() => { button.textContent = original; }, 1200);
      } catch { button.textContent = 'Copy unavailable'; }
    };

    const renderEndpoints = () => {
      const query = search.value.trim().toLowerCase();
      const visible = operations.filter((operation) => `${operation.path} ${operation.summary} ${operation.description} ${groupFor(operation)}`.toLowerCase().includes(query));
      endpointCount.textContent = `${visible.length} endpoint${visible.length === 1 ? '' : 's'}`;
      if (!visible.length) {
        endpointRoot.innerHTML = '<div class="no-results">No endpoints match that filter.</div>';
        return;
      }
      const groups = new Map();
      visible.forEach((operation) => {
        const group = groupFor(operation);
        if (!groups.has(group)) groups.set(group, []);
        groups.get(group).push(operation);
      });
      endpointRoot.innerHTML = [...groups.entries()].map(([group, items]) => `
        <div class="endpoint-group"><h3>${escapeHtml(group)}</h3><div class="endpoint-list">
          ${items.map((operation) => {
            const index = operations.indexOf(operation);
            const method = operation.method.toLowerCase();
            const safe = isSafeRead(operation);
            const params = (operation.parameters || []).filter((parameter) => parameter.in === 'path' || parameter.in === 'query');
            const inputs = params.map((parameter) => `<input class="param-input" data-param="${escapeHtml(parameter.name)}" data-location="${escapeHtml(parameter.in)}" value="${escapeHtml(examples[parameter.name] || '')}" placeholder="${escapeHtml(parameter.name)}" aria-label="${escapeHtml(parameter.name)}">`).join('');
            const badges = safe ? '<span class="badge safe">read-only</span>' : `<span class="badge review">${method === 'get' ? 'stream / restricted' : 'authenticated'}</span>`;
            const controls = safe ? `<button class="run" type="button" data-operation="${index}">Try request</button>` : '<span class="badge review">Use full reference</span>';
            return `<details class="endpoint"><summary><span class="method ${escapeHtml(method)}">${escapeHtml(method.toUpperCase())}</span><span class="path mono">${escapeHtml(operation.path)}</span><span class="operation-badges">${badges}</span><span class="summary-text">${escapeHtml(operation.summary || 'API operation')}</span></summary><div class="endpoint-body"><div class="endpoint-description"><p>${escapeHtml(operation.description || operation.summary || 'API operation.')}</p><div class="try-form">${inputs}${controls}<button class="copy" type="button" data-copy-operation="${index}">Copy cURL</button></div></div><div><span class="response-label">Response preview</span><pre class="response">${safe ? 'Run a read-only request to inspect the response.' : 'This operation is documented here and executable in the full reference UI.'}</pre></div></div></details>`;
          }).join('')}
        </div></div>`).join('');
      endpointRoot.querySelectorAll('.run').forEach((button) => button.addEventListener('click', () => runReadOnly(button)));
      endpointRoot.querySelectorAll('[data-copy-operation]').forEach((button) => button.addEventListener('click', () => copyCurl(button)));
    };

    const loadContract = async () => {
      try {
        const response = await fetch('/openapi.json', { headers: { Accept: 'application/json' } });
        if (!response.ok) throw new Error('contract unavailable');
        const schema = await response.json();
        const methods = ['get', 'post', 'put', 'patch', 'delete', 'head', 'options'];
        operations = Object.entries(schema.paths).flatMap(([path, pathItem]) => methods.filter((method) => pathItem[method]).map((method) => ({ ...pathItem[method], method, path })));
        renderEndpoints();
      } catch {
        endpointCount.textContent = 'Unavailable';
        endpointRoot.innerHTML = '<div class="no-results">The API contract is unavailable. Use the full reference link or try again when the service is ready.</div>';
      }
    };

    const checkReady = async () => {
      const status = document.getElementById('ready-status');
      try {
        const response = await fetch('/ready', { headers: { Accept: 'application/json' } });
        status.textContent = response.ok ? 'API ready' : 'API unavailable';
        status.classList.toggle('online', response.ok);
      } catch { status.textContent = 'API unavailable'; }
    };

    search.addEventListener('input', renderEndpoints);
    loadContract();
    checkReady();
  </script>
</body>
</html>"""
