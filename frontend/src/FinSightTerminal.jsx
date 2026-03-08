import React, { useMemo, useState } from 'react';
import './FinSight.css';

const API_BASE = 'http://localhost:8000';
const TICKERS = ['AAPL', 'NVDA', 'TSLA', 'META', 'GOOGL'];
const RESEARCH_MODES = [
  'auto',
  'lookup_numeric',
  'lookup_text',
  'lookup_text_filing',
  'lookup_text_management',
  'lookup_text_news',
  'compute_metric',
  'comparative_analysis',
  'risk_analysis',
  'valuation',
  'relative_valuation',
  'explanatory_reasoning',
  'mba_framework',
  'multi_period_analysis',
  'scenario_analysis',
  'peer_analysis',
];

const signalClass = (value) => {
  if (value > 0.08) return 'signal-positive';
  if (value < -0.08) return 'signal-negative';
  return 'signal-neutral';
};

const actionClass = (action) => {
  const k = (action || '').toUpperCase();
  if (k === 'ACT' || k === 'BUY') return 'signal-positive';
  if (k === 'WATCH' || k === 'HOLD') return 'signal-neutral';
  return 'signal-negative';
};

const sourceMeta = (ev = {}) => {
  const s = (ev.source_type || '').toLowerCase();
  if (s.includes('filing')) return { icon: 'SEC', ref: '10-K Item 7/8' };
  if (s.includes('news')) return { icon: 'NEWS', ref: 'News Catalyst' };
  if (s.includes('transcript')) return { icon: 'CALL', ref: 'Earnings Call' };
  if (s.includes('table')) return { icon: 'TBL', ref: 'Financial Table' };
  return { icon: 'DOC', ref: ev.source || 'Document' };
};

function FinSightTerminal() {
  const [tab, setTab] = useState('decision');
  const [ticker, setTicker] = useState('AAPL');
  const [strictness, setStrictness] = useState(70);

  const [decLoading, setDecLoading] = useState(false);
  const [decError, setDecError] = useState('');
  const [decResult, setDecResult] = useState(null);
  const [activeToolId, setActiveToolId] = useState('risk');

  const [resMode, setResMode] = useState('auto');
  const [resQuery, setResQuery] = useState('What is the latest investment signal and why?');
  const [resLoading, setResLoading] = useState(false);
  const [resError, setResError] = useState('');
  const [resResult, setResResult] = useState(null);

  const [selectedEvidence, setSelectedEvidence] = useState(null);
  const [evidenceTab, setEvidenceTab] = useState('summary');

  const runDecision = async () => {
    setDecLoading(true);
    setDecError('');
    try {
      const resp = await fetch(`${API_BASE}/api/decision`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ticker, fiscal_year: 2024, strictness }),
      });
      if (!resp.ok) throw new Error(`Server ${resp.status}: ${await resp.text()}`);
      const raw = await resp.json();
      const scoreObj = raw.hackathon_signal_score || {};
      const report = raw.hackathon_signal_report || {};
      const toolsUsed = raw.tools_used || {};
      const evidence = raw.evidence?.chunks || [];

      const tools = [
        {
          id: 'risk',
          name: 'Risk Extraction',
          metric: 'Risk Severity',
          score: Number(scoreObj.component_scores?.risk || 0),
          confidence: 0.68,
          formula: 'score = -1 * severity_avg (top risk buckets)',
          calc: 'NLP extraction from SEC Item 1A risk language with severity weighting.',
          factors: report.top_risks || [],
          sourceReliability: 'High (SEC filing text)',
          uncertainty: 'Medium (keyword/context extraction)',
          evidenceAgreement: evidence.length > 0 ? 0.8 : 0.45,
          evidence: evidence.slice(0, 4),
          toolMeta: toolsUsed.risk,
        },
        {
          id: 'tone',
          name: 'Management Tone',
          metric: 'Tone Delta',
          score: Number(scoreObj.component_scores?.tone || 0),
          confidence: 0.58,
          formula: 'score = FinBERT(current transcript) - FinBERT(prior transcript)',
          calc: 'Compares current vs prior earnings call sentiment trend.',
          factors: [report.tone_trend || {}],
          sourceReliability: 'Medium (transcripts)',
          uncertainty: 'Medium-High (sentiment model noise)',
          evidenceAgreement: evidence.length > 1 ? 0.65 : 0.4,
          evidence: evidence.slice(1, 5),
          toolMeta: toolsUsed.tone,
        },
        {
          id: 'valuation',
          name: 'DCF Valuation',
          metric: 'Valuation Gap',
          score: Number(scoreObj.component_scores?.valuation || 0),
          confidence: 0.61,
          formula: 'gap = (intrinsic_value - market_price) / market_price',
          calc: 'Projects FCF, applies discounting, compares to market price.',
          factors: [report.valuation_summary || {}],
          sourceReliability: 'Medium-High (market + filing inputs)',
          uncertainty: 'High (WACC/terminal assumptions)',
          evidenceAgreement: evidence.length > 2 ? 0.7 : 0.5,
          evidence: evidence.slice(0, 3),
          toolMeta: toolsUsed.valuation,
        },
        {
          id: 'growth',
          name: 'Growth Signal',
          metric: 'Revenue YoY',
          score: Number(scoreObj.component_scores?.growth || 0),
          confidence: 0.62,
          formula: 'score = normalize(revenue_growth_yoy)',
          calc: 'Normalizes YoY growth from financial statements.',
          factors: [{ yoy: toolsUsed.growth?.yoy, summary: 'Topline trend contribution.' }],
          sourceReliability: 'High (reported financial data)',
          uncertainty: 'Low-Medium',
          evidenceAgreement: evidence.length > 0 ? 0.75 : 0.5,
          evidence: evidence.slice(2, 6),
          toolMeta: toolsUsed.growth,
        },
        {
          id: 'news',
          name: 'News Catalyst',
          metric: 'Catalyst Sentiment',
          score: Number(scoreObj.component_scores?.news || 0),
          confidence: 0.42,
          formula: 'score = avg(catalyst sentiment over recent articles)',
          calc: 'Classifies market-moving news events and direction.',
          factors: report.news_summary || [],
          sourceReliability: 'Medium (external news feed)',
          uncertainty: 'High (headline volatility)',
          evidenceAgreement: evidence.length > 0 ? 0.55 : 0.35,
          evidence: evidence.slice(0, 4),
          toolMeta: toolsUsed.news,
        },
      ];

      setDecResult({
        raw,
        score: Number(scoreObj.signal_score || 0),
        confidence: Number(scoreObj.confidence || 0),
        action: raw.hackathon_signal_decision?.action || 'NO_ACT',
        policy: raw.hackathon_signal_decision?.policy || '',
        tools,
      });
      setActiveToolId('risk');
    } catch (err) {
      setDecError(err.message || String(err));
    } finally {
      setDecLoading(false);
    }
  };

  const runResearch = async () => {
    if (!resQuery.trim()) return;
    setResLoading(true);
    setResError('');
    try {
      const resp = await fetch(`${API_BASE}/api/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: resQuery,
          ticker,
          fiscal_year: 2024,
          mode: resMode,
          strictness,
        }),
      });
      if (!resp.ok) throw new Error(`Server ${resp.status}: ${await resp.text()}`);
      const raw = await resp.json();

      const answer = raw.result?.final_answer || `Abstained. Reason: ${raw.reason || 'Unknown.'}`;
      const evidence = raw.evidence?.chunks || [];
      const gate = raw.verification?.gate || raw.result?.gate || {};

      setResResult({
        raw,
        answer,
        action: (raw.action || 'abstain').toUpperCase(),
        rationale: raw.reason || 'Rationale derived from selected mode and evidence set.',
        confidence: Number(raw.result?.confidence || gate.confidence || 0),
        evidenceScore: Number(gate.score || gate.evidence_score || 0),
        evidence,
      });
      setEvidenceTab('summary');
    } catch (err) {
      setResError(err.message || String(err));
    } finally {
      setResLoading(false);
    }
  };

  const activeTool = useMemo(() => {
    return decResult?.tools?.find((t) => t.id === activeToolId) || null;
  }, [decResult, activeToolId]);

  return (
    <div className="terminal-root">
      <header className="terminal-header">
        <div>
          <div className="brand-title">FinSignal AI</div>
          <div className="brand-subtitle">Institutional Financial Intelligence Dashboard</div>
        </div>
        <div className="header-tabs">
          <button className={tab === 'decision' ? 'header-tab active' : 'header-tab'} onClick={() => setTab('decision')}>Decision Mode</button>
          <button className={tab === 'research' ? 'header-tab active' : 'header-tab'} onClick={() => setTab('research')}>Research Mode</button>
        </div>
      </header>

      {tab === 'decision' && (
        <section className="panel-block">
          <div className="control-panel">
            <div className="control-group">
              <label>Ticker</label>
              <select value={ticker} onChange={(e) => setTicker(e.target.value)}>
                {TICKERS.map((t) => <option key={t} value={t}>{t}</option>)}
              </select>
            </div>
            <div className="control-group control-wide">
              <label>Evidence Strictness: {strictness}</label>
              <input type="range" min="0" max="100" value={strictness} onChange={(e) => setStrictness(Number(e.target.value))} />
            </div>
            <button className="run-btn" onClick={runDecision} disabled={decLoading}>{decLoading ? 'Running...' : 'Run Analysis'}</button>
          </div>

          {decError && <div className="error-banner">{decError}</div>}

          {decResult && (
            <>
              <div className="kpi-grid kpi-large">
                <article className={`kpi-card ${actionClass(decResult.action)}`}>
                  <div className="kpi-label">Decision Signal</div>
                  <div className="kpi-value">{decResult.action}</div>
                  <div className="kpi-note">Rule output from weighted multi-tool signal.</div>
                </article>
                <article className={`kpi-card ${signalClass(decResult.score)}`}>
                  <div className="kpi-label">Signal Score</div>
                  <div className="kpi-value">{decResult.score >= 0 ? '+' : ''}{decResult.score.toFixed(4)}</div>
                  <div className="kpi-note">Confidence-weighted blend of independent tool signals.</div>
                </article>
                <article className="kpi-card">
                  <div className="kpi-label">Confidence</div>
                  <div className="kpi-value">{(decResult.confidence * 100).toFixed(1)}%</div>
                  <div className="kpi-note">Based on evidence quality, model reliability, and uncertainty penalties.</div>
                </article>
              </div>

              <div className="policy-bar">{decResult.policy}</div>

              <h3 className="section-title">Tool Analysis</h3>
              <div className="tool-grid">
                {decResult.tools.map((tool) => (
                  <button
                    key={tool.id}
                    className={activeToolId === tool.id ? `tool-card active ${signalClass(tool.score)}` : `tool-card ${signalClass(tool.score)}`}
                    onClick={() => setActiveToolId(tool.id)}
                  >
                    <div className="tool-name">{tool.name}</div>
                    <div className="tool-score">{tool.score >= 0 ? '+' : ''}{tool.score.toFixed(3)}</div>
                    <div className="tool-meta">Conf {(tool.confidence * 100).toFixed(0)}% | {tool.metric}</div>
                  </button>
                ))}
              </div>

              {activeTool && (
                <div className="tool-expanded">
                  <div className="tool-expanded-header">
                    <h4>{activeTool.name}</h4>
                    <span className={`tag ${signalClass(activeTool.score)}`}>{activeTool.metric}</span>
                  </div>

                  <div className="detail-grid">
                    <article className="detail-card">
                      <h5>How Score Was Calculated</h5>
                      <p>{activeTool.calc}</p>
                      <code>{activeTool.formula}</code>
                    </article>
                    <article className="detail-card">
                      <h5>Confidence Calculation</h5>
                      <p>Base confidence: {(activeTool.confidence * 100).toFixed(1)}%</p>
                      <p>Evidence agreement: {(activeTool.evidenceAgreement * 100).toFixed(1)}%</p>
                      <p>Source reliability: {activeTool.sourceReliability}</p>
                      <p>Model uncertainty: {activeTool.uncertainty}</p>
                    </article>
                    <article className="detail-card">
                      <h5>Contributing Factors</h5>
                      <pre>{JSON.stringify(activeTool.factors, null, 2)}</pre>
                    </article>
                    <article className="detail-card">
                      <h5>Tool Metadata</h5>
                      <pre>{JSON.stringify(activeTool.toolMeta || {}, null, 2)}</pre>
                    </article>
                  </div>

                  <h5 className="sub-title">Evidence Sources</h5>
                  <div className="evidence-grid">
                    {activeTool.evidence.map((ev, idx) => {
                      const meta = sourceMeta(ev);
                      return (
                        <article key={`${activeTool.id}-${idx}`} className="evidence-card" onClick={() => setSelectedEvidence(ev)}>
                          <div className="evidence-top">
                            <span className="source-pill">{meta.icon}</span>
                            <span className="source-ref">{meta.ref}</span>
                          </div>
                          <p>{(ev.text || '').slice(0, 140)}{(ev.text || '').length > 140 ? '...' : ''}</p>
                          <button className="open-source-btn" type="button">Open Source</button>
                        </article>
                      );
                    })}
                  </div>
                </div>
              )}
            </>
          )}
        </section>
      )}

      {tab === 'research' && (
        <section className="panel-block">
          <div className="control-panel">
            <div className="control-group">
              <label>Ticker</label>
              <select value={ticker} onChange={(e) => setTicker(e.target.value)}>
                {TICKERS.map((t) => <option key={t} value={t}>{t}</option>)}
              </select>
            </div>
            <div className="control-group">
              <label>Mode</label>
              <select value={resMode} onChange={(e) => setResMode(e.target.value)}>
                {RESEARCH_MODES.map((m) => <option key={m} value={m}>{m}</option>)}
              </select>
            </div>
            <div className="control-group control-wide">
              <label>Evidence Strictness: {strictness}</label>
              <input type="range" min="0" max="100" value={strictness} onChange={(e) => setStrictness(Number(e.target.value))} />
            </div>
          </div>

          <div className="query-row">
            <input
              value={resQuery}
              onChange={(e) => setResQuery(e.target.value)}
              placeholder="Ask a research question with evidence traceability"
              onKeyDown={(e) => e.key === 'Enter' && runResearch()}
            />
            <button className="run-btn" onClick={runResearch} disabled={resLoading}>{resLoading ? 'Running...' : 'Run Query'}</button>
          </div>

          {resError && <div className="error-banner">{resError}</div>}

          {resResult && (
            <>
              <article className="answer-card">
                <h3 className="section-title">Analysis Summary</h3>
                <p>{resResult.answer}</p>
              </article>

              <div className="kpi-grid kpi-compact">
                <article className="kpi-card"><div className="kpi-label">Evidence Score</div><div className="kpi-small">{resResult.evidenceScore.toFixed(2)}</div></article>
                <article className="kpi-card"><div className="kpi-label">Confidence</div><div className="kpi-small">{(resResult.confidence * 100).toFixed(1)}%</div></article>
                <article className={`kpi-card ${actionClass(resResult.action)}`}><div className="kpi-label">Action</div><div className="kpi-small">{resResult.action}</div></article>
                <article className="kpi-card"><div className="kpi-label">Rationale</div><div className="kpi-small">{resResult.rationale}</div></article>
              </div>

              <section className="evidence-panel">
                <div className="evidence-tabs">
                  <button className={evidenceTab === 'summary' ? 'ev-tab active' : 'ev-tab'} onClick={() => setEvidenceTab('summary')}>Source Summary</button>
                  <button className={evidenceTab === 'table' ? 'ev-tab active' : 'ev-tab'} onClick={() => setEvidenceTab('table')}>Structured Table View</button>
                  <button className={evidenceTab === 'context' ? 'ev-tab active' : 'ev-tab'} onClick={() => setEvidenceTab('context')}>Expandable Context</button>
                </div>

                {evidenceTab === 'summary' && (
                  <div className="evidence-grid">
                    {resResult.evidence.map((ev, idx) => {
                      const meta = sourceMeta(ev);
                      return (
                        <article key={idx} className="evidence-card" onClick={() => setSelectedEvidence(ev)}>
                          <div className="evidence-top">
                            <span className="source-pill">{meta.icon}</span>
                            <span className="source-ref">{meta.ref}</span>
                          </div>
                          <p>{(ev.text || '').slice(0, 140)}{(ev.text || '').length > 140 ? '...' : ''}</p>
                          <div className="evidence-bottom">
                            <span>Conf {(Number(ev.score || ev.confidence || 0.5) * 100).toFixed(0)}%</span>
                            <button className="open-source-btn" type="button">Open Source</button>
                          </div>
                        </article>
                      );
                    })}
                  </div>
                )}

                {evidenceTab === 'table' && (
                  <div className="table-wrap">
                    <table>
                      <thead>
                        <tr>
                          <th>Type</th>
                          <th>Citation</th>
                          <th>Snippet</th>
                          <th>Confidence</th>
                        </tr>
                      </thead>
                      <tbody>
                        {resResult.evidence.map((ev, idx) => {
                          const meta = sourceMeta(ev);
                          return (
                            <tr key={`row-${idx}`}>
                              <td>{meta.icon}</td>
                              <td>{meta.ref}</td>
                              <td>{(ev.text || '').slice(0, 90)}{(ev.text || '').length > 90 ? '...' : ''}</td>
                              <td>{(Number(ev.score || ev.confidence || 0.5) * 100).toFixed(0)}%</td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}

                {evidenceTab === 'context' && (
                  <div className="context-list">
                    {resResult.evidence.map((ev, idx) => (
                      <details key={`ctx-${idx}`}>
                        <summary>{sourceMeta(ev).ref} | {(ev.source || 'Document')}</summary>
                        <p>{ev.text || 'No text available.'}</p>
                        <button className="open-source-btn" type="button" onClick={() => setSelectedEvidence(ev)}>Open Source</button>
                      </details>
                    ))}
                  </div>
                )}
              </section>
            </>
          )}
        </section>
      )}

      {selectedEvidence && (
        <div className="drawer-overlay" onClick={() => setSelectedEvidence(null)}>
          <aside className="doc-drawer" onClick={(e) => e.stopPropagation()}>
            <div className="drawer-head">
              <h4>Document Preview</h4>
              <button onClick={() => setSelectedEvidence(null)} className="close-btn" type="button">Close</button>
            </div>
            <div className="drawer-meta">{selectedEvidence.source || 'Source document'} | {sourceMeta(selectedEvidence).ref}</div>
            <div className="drawer-body">{selectedEvidence.text || 'No preview text available.'}</div>
          </aside>
        </div>
      )}
    </div>
  );
}

export default FinSightTerminal;
