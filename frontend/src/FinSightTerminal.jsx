import React, { useState } from 'react';
import './FinSight.css';

const API_URL = 'http://localhost:8000/api/analyze';

const FinSightTerminal = () => {
  const [appTab, setAppTab] = useState('decision');
  const [ticker, setTicker] = useState('AAPL');
  const [strictness, setStrictness] = useState(70);

  // Decision state
  const [decLoading, setDecLoading] = useState(false);
  const [decProgress, setDecProgress] = useState('');
  const [decResult, setDecResult] = useState(null);
  const [decError, setDecError] = useState(null);
  const [activeComponent, setActiveComponent] = useState('risk');

  // Research state
  const [resMode, setResMode] = useState('auto');
  const [resQuery, setResQuery] = useState('');
  const [resLoading, setResLoading] = useState(false);
  const [resResult, setResResult] = useState(null);
  const [resError, setResError] = useState(null);

  // Modal
  const [selectedEvidence, setSelectedEvidence] = useState(null);

  // ── Decision Pipeline ──
  const runDecision = async () => {
    setDecLoading(true);
    setDecError(null);
    setDecResult(null);
    setDecProgress('Ingesting raw text feed & retrieving documents...');
    setActiveComponent('risk');

    const interval = setInterval(() => {
      const msgs = [
        'Executing Financial Orchestrator...',
        'Scanning SEC filings for risk language...',
        'Analyzing management tone via FinBERT...',
        'Computing DCF valuation gap...',
        'Scoring multi-dimensional signals...',
        'Fetching live news sentiment...',
        'Formulating deterministic decision rule...',
      ];
      setDecProgress(msgs[Math.floor(Math.random() * msgs.length)]);
    }, 2200);

    try {
      const resp = await fetch('http://localhost:8000/api/decision', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          ticker,
          fiscal_year: 2024,
          strictness: parseInt(strictness, 10),
        }),
      });
      if (!resp.ok) throw new Error(`Server ${resp.status}: ${await resp.text()}`);
      const raw = await resp.json();

      let signal = null;
      if (raw.hackathon_signal_report) {
        const rep = raw.hackathon_signal_report;
        const sc = raw.hackathon_signal_score || {};
        signal = {
          decision: raw.hackathon_signal_decision?.action || 'NO_ACT',
          recommendation: (rep.recommendation || 'HOLD').toUpperCase(),
          strength: rep.signal_strength || 0,
          confidence: rep.confidence || 0,
          components: sc.component_scores || {},
          findings: rep.key_findings || [],
          risks: rep.top_risks || [],
          tone: rep.tone_trend || {},
          valuation: rep.valuation_summary || {},
          news: rep.news_summary || [],
        };
      }

      // Parse evidence chunks
      let evidence = [];
      if (raw.evidence?.chunks) evidence = raw.evidence.chunks;
      else if (raw.packed_context) evidence = [{ text: raw.packed_context.substring(0, 800), source: 'Raw Context' }];

      setDecResult({ signal, markdown: raw.hackathon_signal_markdown || '', evidence });
    } catch (e) {
      setDecError(e.message);
    } finally {
      clearInterval(interval);
      setDecLoading(false);
    }
  };

  // ── Research Pipeline ──
  const runResearch = async () => {
    if (!resQuery.trim()) return;
    setResLoading(true);
    setResError(null);
    setResResult(null);
    try {
      const resp = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: resQuery, ticker, fiscal_year: 2024,
          mode: resMode, strictness: parseInt(strictness, 10),
        }),
      });
      if (!resp.ok) throw new Error(`Server ${resp.status}: ${await resp.text()}`);
      const raw = await resp.json();
      let answer = raw.result?.final_answer || `Abstained. Reason: ${raw.reason || 'Unknown'}`;
      const sig = answer.indexOf('--- Investment Signal:');
      if (sig > -1) answer = answer.substring(0, sig).trim();
      let evidence = [];
      if (raw.evidence?.chunks) evidence = raw.evidence.chunks.slice(0, 12);
      else if (raw.packed_context) evidence = [{ text: raw.packed_context.substring(0, 600) + '...', source: 'Raw Context' }];
      setResResult({ answer, evidence, mode: raw.mode || resMode, action: raw.action || 'abstain', metric: raw.result?.claims?.[0]?.value_or_summary || '', math: raw.result?.claims?.[0]?.formula || '' });
    } catch (e) { setResError(e.message); } finally { setResLoading(false); }
  };

  // ── Component tab data builder ──
  const getComponentData = (sig) => {
    if (!sig) return {};
    return {
      risk: {
        title: 'Risk Analysis',
        score: sig.components.risk || 0,
        color: (sig.components.risk || 0) >= 0 ? 'var(--accent-green)' : 'var(--accent-red)',
        kpis: sig.risks.length > 0
          ? sig.risks.map(r => ({ label: r.category || r, value: typeof r === 'object' ? `Severity: ${(r.severity || 0).toFixed(2)} | Count: ${r.count || 0}` : 'Flagged' }))
          : [{ label: 'Risk Status', value: Math.abs(sig.components.risk || 0) > 0.5 ? 'Elevated Risk' : 'Moderate' }],
        description: 'Scans SEC Filing Item 1A for risk keywords (regulatory, litigation, supply chain, cyber, macro). Higher severity → more negative score.',
      },
      tone: {
        title: 'Tone Trend',
        score: sig.components.tone || 0,
        color: (sig.components.tone || 0) >= 0 ? 'var(--accent-green)' : 'var(--accent-red)',
        kpis: [
          { label: 'Direction', value: sig.tone.direction || 'Stable' },
          { label: 'Delta', value: (sig.tone.delta || 0).toFixed(3) },
        ],
        description: 'Measures the change in management sentiment between earnings calls using FinBERT NLP. Positive delta = improving outlook.',
      },
      valuation: {
        title: 'Valuation Gap',
        score: sig.components.valuation || 0,
        color: (sig.components.valuation || 0) >= 0 ? 'var(--accent-green)' : 'var(--accent-red)',
        kpis: [
          { label: 'Valuation Gap %', value: sig.valuation.valuation_gap_pct != null ? `${(sig.valuation.valuation_gap_pct * 100).toFixed(1)}%` : 'N/A' },
          { label: 'Intrinsic Value', value: sig.valuation.intrinsic_value != null ? `$${sig.valuation.intrinsic_value.toFixed(2)}` : 'N/A' },
          { label: 'Market Price', value: sig.valuation.current_price != null ? `$${sig.valuation.current_price.toFixed(2)}` : 'N/A' },
          { label: 'Status', value: (sig.components.valuation || 0) > 0 ? 'Undervalued' : (sig.components.valuation || 0) < 0 ? 'Overvalued' : 'Fair Value' },
        ],
        description: 'Compares intrinsic value (from DCF model using revenue × 12% FCF proxy) vs current market price. Positive = undervalued opportunity.',
        tool: 'run_dcf + yfinance',
      },
      growth: {
        title: 'Growth Metrics',
        score: sig.components.growth || 0,
        color: (sig.components.growth || 0) >= 0 ? 'var(--accent-green)' : 'var(--accent-red)',
        kpis: [
          { label: 'YoY Revenue Growth', value: sig.components.growth !== 0 ? `${(sig.components.growth * 40).toFixed(1)}%` : 'N/A' },
          { label: 'Normalized Score', value: (sig.components.growth || 0).toFixed(3) },
        ],
        description: 'Year-over-year revenue growth, normalized (40%+ YoY caps at +1.0).',
      },
      news: {
        title: 'News Sentiment',
        score: sig.components.news || 0,
        color: (sig.components.news || 0) >= 0 ? 'var(--accent-green)' : 'var(--accent-red)',
        kpis: [
          { label: 'Direction', value: (sig.components.news || 0) > 0 ? 'Bullish' : (sig.components.news || 0) < 0 ? 'Bearish' : 'Neutral' },
          { label: 'Score', value: (sig.components.news || 0).toFixed(3) },
        ],
        description: 'Live news sentiment from NewsAPI headlines. Positive = bullish, negative = bearish.',
      },
    };
  };

  const componentKeys = ['risk', 'tone', 'valuation', 'growth', 'news'];

  return (
    <div className="finsight-container">
      {/* HEADER */}
      <header className="finsight-header glass-panel">
        <div className="header-brand">
          <h1>FinSight</h1>
          <span className="brand-sub">Intelligence Terminal</span>
        </div>
        <nav className="app-tabs">
          <button className={`app-tab ${appTab === 'decision' ? 'active' : ''}`} onClick={() => setAppTab('decision')}>
            Decision Analysis
          </button>
          <button className={`app-tab ${appTab === 'research' ? 'active' : ''}`} onClick={() => setAppTab('research')}>
            Research Mode
          </button>
        </nav>
      </header>

      {/* ═══════════ TAB 1: DECISION ANALYSIS ═══════════ */}
      {appTab === 'decision' && (
        <div className="tab-view fade-in-up">
          <div className="decision-controls glass-panel">
            <div className="control-row">
              <div className="control-item lg">
                <label>TICKER</label>
                <input value={ticker} onChange={e => setTicker(e.target.value.toUpperCase())} placeholder="AAPL" />
              </div>
              <div className="control-item xl">
                <label>EVIDENCE STRICTNESS: {strictness}%</label>
                <input type="range" min="0" max="100" value={strictness} onChange={e => setStrictness(e.target.value)} className="strictness-slider" />
                <div className="strictness-presets">
                  <button className={strictness == 30 ? 'preset active' : 'preset'} onClick={() => setStrictness(30)}>Lenient</button>
                  <button className={strictness == 50 ? 'preset active' : 'preset'} onClick={() => setStrictness(50)}>Balanced</button>
                  <button className={strictness == 70 ? 'preset active' : 'preset'} onClick={() => setStrictness(70)}>Strict</button>
                  <button className={strictness == 90 ? 'preset active' : 'preset'} onClick={() => setStrictness(90)}>Maximum</button>
                </div>
              </div>
              <button className="run-btn decision-run" onClick={runDecision} disabled={decLoading}>
                {decLoading ? <span className="spinner"></span> : 'GENERATE SIGNAL'}
              </button>
            </div>
          </div>

          {decError && <div className="error-banner">⚠️ {decError}</div>}

          {decLoading && (
            <div className="thinking-container glass-panel fade-in-up">
              <div className="radar-spinner"></div>
              <h2>System is analyzing {ticker}...</h2>
              <p className="progress-text">{decProgress}</p>
            </div>
          )}

          {decResult?.signal && !decLoading && (
            <div className="decision-results fade-in-up">

              {/* ── HERO KPI STRIP ── */}
              <div className="kpi-strip">
                <div className={`kpi-card kpi-decision kpi-${decResult.signal.decision.toLowerCase()}`}>
                  <span className="kpi-label">DECISION</span>
                  <span className="kpi-value">{decResult.signal.decision}</span>
                </div>
                <div className="kpi-card">
                  <span className="kpi-label">SIGNAL STRENGTH</span>
                  <span className="kpi-value" style={{ color: decResult.signal.strength >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                    {decResult.signal.strength > 0 ? '+' : ''}{decResult.signal.strength.toFixed(3)}
                  </span>
                </div>
                <div className="kpi-card">
                  <span className="kpi-label">RECOMMENDATION</span>
                  <span className="kpi-value rec-value">{decResult.signal.recommendation}</span>
                </div>
                <div className="kpi-card">
                  <span className="kpi-label">CONFIDENCE</span>
                  <span className="kpi-value">{(decResult.signal.confidence * 100).toFixed(0)}%</span>
                </div>
              </div>

              {/* ── ASSUMPTIONS ── */}
              <div className="assumptions-bar glass-panel">
                <span className="assumptions-label">ASSUMPTIONS</span>
                <span>Strictness: {strictness}% · Fiscal Year: 2024 · Ticker: {ticker} · Decision Threshold: ACT ≥ 0.35 & Conf ≥ 55%</span>
              </div>

              {/* ── COMPONENT SUB-TABS ── */}
              <div className="component-tabs">
                {componentKeys.map(key => {
                  const sc = decResult.signal.components[key] || 0;
                  return (
                    <button
                      key={key}
                      className={`comp-tab ${activeComponent === key ? 'active' : ''}`}
                      onClick={() => setActiveComponent(key)}
                    >
                      <span className="comp-tab-name">{key}</span>
                      <span className="comp-tab-score" style={{ color: sc >= 0 ? 'var(--accent-green)' : 'var(--accent-red)' }}>
                        {sc > 0 ? '+' : ''}{sc.toFixed(2)}
                      </span>
                    </button>
                  );
                })}
              </div>

              {/* ── ACTIVE COMPONENT CONTENT ── */}
              {(() => {
                const data = getComponentData(decResult.signal);
                const comp = data[activeComponent];
                if (!comp) return null;
                return (
                  <div className="component-detail glass-panel fade-in-up" key={activeComponent}>
                    <div className="comp-detail-header">
                      <h3>{comp.title}</h3>
                      <div className="comp-score-big" style={{ color: comp.color }}>
                        {comp.score > 0 ? '+' : ''}{comp.score.toFixed(3)}
                      </div>
                    </div>
                    <p className="comp-description">{comp.description}</p>

                    <div className="comp-kpi-grid">
                      {comp.kpis.map((kpi, i) => (
                        <div key={i} className="comp-kpi-card">
                          <span className="comp-kpi-label">{kpi.label}</span>
                          <span className="comp-kpi-value">{kpi.value}</span>
                        </div>
                      ))}
                    </div>

                    <div className="progress-track lg-track">
                      <div className="progress-fill" style={{
                        width: `${Math.max(3, Math.abs(comp.score) * 100)}%`,
                        backgroundColor: comp.color,
                      }}></div>
                    </div>

                    {/* Evidence for this component */}
                    {decResult.evidence && decResult.evidence.length > 0 && (
                      <div className="comp-evidence">
                        <h4 className="section-label" style={{ marginTop: '1.5rem' }}>Supporting Evidence</h4>
                        <div className="evidence-grid compact">
                          {decResult.evidence.slice(0, 4).map((item, idx) => (
                            <div key={idx} className="evidence-card clickable" onClick={() => setSelectedEvidence(item)}>
                              <div className="evidence-meta">
                                <span className="source-badge">{item.source_type && item.source_type !== 'unknown' ? item.source_type.toUpperCase() : 'DOCUMENT'}</span>
                              </div>
                              <p className="evidence-text">
                                "{item.text?.length > 120 ? item.text.substring(0, 120) + '...' : item.text}"
                              </p>
                              <div className="evidence-footer">
                                <span>{item.source || 'Knowledge Base'}</span>
                                <span className="click-hint">Click to expand →</span>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                );
              })()}
            </div>
          )}
        </div>
      )}

      {/* ═══════════ TAB 2: RESEARCH MODE ═══════════ */}
      {appTab === 'research' && (
        <div className="tab-view fade-in-up">
          <div className="research-controls glass-panel">
            <div className="control-row">
              <div className="control-item">
                <label>TICKER</label>
                <input value={ticker} onChange={e => setTicker(e.target.value.toUpperCase())} placeholder="AAPL" />
              </div>
              <div className="control-item">
                <label>TOOL MODE</label>
                <select value={resMode} onChange={e => setResMode(e.target.value)}>
                  <option value="auto">Auto-Detect</option>
                  <option value="valuation">DCF Valuation</option>
                  <option value="risk_analysis">Risk Analysis</option>
                  <option value="relative_valuation">Multiples (Relative)</option>
                  <option value="compute_metric">Fact Extraction</option>
                  <option value="mba_framework">MBA Framework</option>
                  <option value="comparative_analysis">Comparative</option>
                  <option value="lookup_numeric">Data Search</option>
                  <option value="lookup_text_management">Filing/Transcript Search</option>
                  <option value="lookup_text_news">News Search</option>
                  <option value="explanatory_reasoning">Financial Reasoning</option>
                </select>
              </div>
              <div className="control-item xl">
                <label>STRICTNESS: {strictness}%</label>
                <input type="range" min="0" max="100" value={strictness} onChange={e => setStrictness(e.target.value)} className="strictness-slider" />
              </div>
            </div>
          </div>

          <section className="query-section">
            <input type="text" className="main-search-input glass-input"
              placeholder="Ask a question (e.g., 'What is Apple's intrinsic value based on DCF?')"
              value={resQuery} onChange={e => setResQuery(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && runResearch()}
            />
            <button className="run-btn" onClick={runResearch} disabled={resLoading}>
              {resLoading ? <span className="spinner"></span> : 'RUN QUERY'}
            </button>
          </section>

          {resError && <div className="error-banner">⚠️ {resError}</div>}
          {resLoading && (
            <div className="thinking-container glass-panel fade-in-up">
              <div className="radar-spinner"></div>
              <h2>Researching...</h2>
              <p className="progress-text">Executing {resMode} pipeline for {ticker}...</p>
            </div>
          )}

          {resResult && !resLoading && (
            <div className="research-results fade-in-up">
              <div className="answer-panel glass-panel">
                <div className="panel-header">
                  <h3 className="section-label">Answer — {resResult.mode}</h3>
                  <span className={`status-indicator indicator-${resResult.action}`}>{resResult.action.toUpperCase()}</span>
                </div>
                {resResult.metric && <div className="primary-metric">{resResult.metric}</div>}
                <p className="answer-text">{resResult.answer}</p>
                {resResult.math && <div className="math-trace"><code>{resResult.math}</code></div>}
              </div>
              {resResult.evidence.length > 0 && (
                <div className="evidence-section">
                  <h3 className="section-label">Evidence Used ({resResult.evidence.length})</h3>
                  <div className="evidence-grid">
                    {resResult.evidence.map((item, idx) => (
                      <div key={idx} className="evidence-card clickable" onClick={() => setSelectedEvidence(item)}>
                        <div className="evidence-meta">
                          <span className="source-badge">{item.source_type && item.source_type !== 'unknown' ? item.source_type.toUpperCase() : 'DOCUMENT'}</span>
                          {item.ticker && <span className="source-ticker">{item.ticker}</span>}
                        </div>
                        <p className="evidence-text">"{item.text?.length > 150 ? item.text.substring(0, 150) + '...' : item.text}"</p>
                        <div className="evidence-footer">
                          <span>{item.source || 'Knowledge Base'}</span>
                          <span className="click-hint">Click to expand →</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* ═══════════ EVIDENCE MODAL ═══════════ */}
      {selectedEvidence && (
        <div className="modal-overlay" onClick={() => setSelectedEvidence(null)}>
          <div className="modal-content glass-panel" onClick={e => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setSelectedEvidence(null)}>×</button>
            <div className="modal-header">
              <h3>Evidence Source</h3>
              <div className="evidence-meta" style={{ justifyContent: 'flex-start', gap: '10px' }}>
                <span className="source-badge">{selectedEvidence.source_type && selectedEvidence.source_type !== 'unknown' ? selectedEvidence.source_type.toUpperCase() : 'DOCUMENT'}</span>
                {selectedEvidence.ticker && <span className="source-ticker">{selectedEvidence.ticker}</span>}
              </div>
            </div>
            <div className="modal-body">
              <p className="full-evidence-text">{selectedEvidence.text}</p>
            </div>
            <div className="modal-footer">
              <span className="modal-source-path">{selectedEvidence.source || 'Knowledge Base'}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FinSightTerminal;
