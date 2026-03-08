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
  const [toolEvCategory, setToolEvCategory] = useState('All');
  const [resEvCategory, setResEvCategory] = useState('All');

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
      const toolEvidenceMap = raw.tool_evidence || {};

      const tools = [
        {
          id: 'risk',
          name: 'Risk Extraction',
          metric: 'Risk Severity',
          score: Number(scoreObj.component_scores?.risk || 0),
          confidence: 0.68,
          description: 'Extracts and scores key risks outlined by management in the 10-K to understand major headwinds.',
          formula: 'score = -1 * severity_avg (top risk buckets)',
          calc: 'NLP extraction from SEC Item 1A risk language with severity weighting.',
          factors: toolsUsed.risk?.factors || toolsUsed.risk?.top_risks || [],
          sourceReliability: 'High (SEC filing text)',
          uncertainty: 'Medium (keyword/context extraction)',
          evidenceAgreement: 0.8,
          evidence: toolEvidenceMap.risk || [],
          toolMeta: toolsUsed.risk,
        },
        {
          id: 'tone',
          name: 'Management Tone',
          metric: 'Tone Delta',
          score: Number(scoreObj.component_scores?.tone || 0),
          confidence: 0.58,
          description: 'Analyzes earnings transcripts to detect shifts in management optimism or pessimism over time.',
          formula: 'score = LLM_Tone(current transcript) - LLM_Tone(prior transcript)',
          calc: 'Compares current vs prior earnings call sentiment trend.',
          factors: toolsUsed.tone?.factors || toolsUsed.tone || {},
          sourceReliability: 'Medium (transcripts)',
          uncertainty: 'Medium-High (sentiment model noise)',
          evidenceAgreement: 0.65,
          evidence: toolEvidenceMap.tone || [],
          toolMeta: toolsUsed.tone,
        },
        {
          id: 'valuation',
          name: 'DCF Valuation',
          metric: 'Valuation Gap',
          score: Number(scoreObj.component_scores?.valuation || 0),
          confidence: 0.61,
          description: 'Calculates the intrinsic value of the company using a Discounted Cash Flow model.',
          formula: 'gap = (intrinsic_value - market_price) / market_price',
          calc: 'Projects FCF, applies discounting, compares to market price.',
          factors: toolsUsed.valuation?.factors || toolsUsed.valuation || {},
          sourceReliability: 'Medium-High (market + yfinance inputs)',
          uncertainty: 'High (WACC/terminal assumptions)',
          evidenceAgreement: 0.7,
          evidence: toolEvidenceMap.valuation || [],
          toolMeta: toolsUsed.valuation,
        },
        {
          id: 'growth',
          name: 'Growth Signal',
          metric: 'Revenue YoY',
          score: Number(scoreObj.component_scores?.growth || 0),
          confidence: 0.62,
          description: 'Evaluates the company’s recent top-line revenue growth trajectory.',
          formula: 'score = normalize(revenue_growth_yoy)',
          calc: 'Normalizes YoY growth from financial statements.',
          factors: toolsUsed.growth?.factors || { yoy: toolsUsed.growth?.yoy },
          sourceReliability: 'High (reported financial data)',
          uncertainty: 'Low-Medium',
          evidenceAgreement: 0.75,
          evidence: toolEvidenceMap.growth || [],
          toolMeta: toolsUsed.growth,
        },
        {
          id: 'news',
          name: 'News Catalyst',
          metric: 'Catalyst Sentiment',
          score: Number(scoreObj.component_scores?.news || 0),
          confidence: 0.42,
          description: 'Scans recent headlines for market-moving events and product announcements.',
          formula: 'score = avg(catalyst sentiment over recent articles)',
          calc: 'Classifies market-moving news events using LLM.',
          factors: toolsUsed.news?.factors || toolsUsed.news || [],
          sourceReliability: 'Medium (external news feed)',
          uncertainty: 'High (headline volatility)',
          evidenceAgreement: 0.55,
          evidence: toolEvidenceMap.news || [],
          toolMeta: toolsUsed.news,
        },
        {
          id: 'scenarios',
          name: 'Scenario Analysis',
          metric: 'Upside/Downside',
          score: Number(scoreObj.component_scores?.valuation || 0),
          confidence: 0.75,
          description: 'Stresses the valuation model with Bull and Bear assumptions to show potential price outcomes.',
          formula: 'Projection = FCF * Growth / WACC',
          calc: 'Simulates Bull/Bear scenarios based on FCF variability.',
          factors: raw.scenarios || {},
          sourceReliability: 'High (Quantitative)',
          uncertainty: 'High (Projections)',
          evidenceAgreement: 1.0,
          evidence: toolEvidenceMap.scenarios || toolEvidenceMap.valuation || [],
          toolMeta: toolsUsed.scenarios,
        },
        {
          id: 'peers',
          name: 'Peer Comparison',
          metric: 'Relative Val',
          score: Number(raw.peers?.premium_pct ? -raw.peers.premium_pct : 0),
          confidence: 0.82,
          description: 'Benchmarks the company’s valuation multiples against industry competitors.',
          formula: 'Premium = (Target - Peer Median) / Peer Median',
          calc: 'Compares target to industry peers (AMD, INTC, etc).',
          factors: raw.peers || {},
          sourceReliability: 'High (Market Data)',
          uncertainty: 'Medium',
          evidenceAgreement: 0.9,
          evidence: toolEvidenceMap.peers || toolEvidenceMap.valuation || [],
          toolMeta: toolsUsed.peers,
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
      setToolEvCategory('All');
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
                    onClick={() => { setActiveToolId(tool.id); setToolEvCategory('All'); }}
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
                      <h5>What This Tool Does</h5>
                      <p>{activeTool.description}</p>
                      <h5 style={{marginTop: '15px'}}>How Score Was Calculated</h5>
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
                    <article className="detail-card full-width">
                      <h5>Contributing Factors</h5>
                      <div className="factor-container">
                        {activeToolId === 'risk' && (
                          <div className="risk-factors">
                            {Array.isArray(activeTool.factors) ? activeTool.factors.map((f, i) => (
                              <div key={i} className="risk-chip">
                                <span className={`severity-hex sev-${Math.round((f.severity || 0) * 10)}`}></span>
                                <b>{f.category || 'Risk'}</b>: {f.reasoning || f.text || 'No description provided.'}
                              </div>
                            )) : <div style={{padding: '10px'}}>No risk factors mapped.</div>}
                          </div>
                        )}
                        {activeToolId === 'tone' && (
                          <div className="tone-report">
                            <div className="tone-meter">
                              <div className="tone-bar" style={{ width: `${((activeTool.factors.delta || 0) + 1) * 50}%` }}></div>
                            </div>
                            <div className="tone-labels">
                              <span>Prior: {(activeTool.factors.prior?.tone_score ?? activeTool.factors.prior_sentiment ?? 0).toFixed(2)}</span>
                              <span>Current: {(activeTool.factors.current?.tone_score ?? activeTool.factors.current_sentiment ?? 0).toFixed(2)}</span>
                            </div>
                            <p className="tone-summary">Management tone shifted <b>{activeTool.factors.direction || 'stable'}</b> by {Math.abs(activeTool.factors.delta || 0).toFixed(3)} units.</p>
                          </div>
                        )}
                        {activeToolId === 'valuation' && (
                          <div className="valuation-table">
                            <table>
                              <tbody>
                                <tr><td>Intrinsic Value</td><td className="signal-positive">${activeTool.factors.intrinsic_value}</td></tr>
                                <tr><td>Market Price</td><td>${activeTool.factors.current_price}</td></tr>
                                <tr><td>Valuation Gap</td><td className={signalClass(activeTool.factors.valuation_gap_pct)}>{(activeTool.factors.valuation_gap_pct * 100).toFixed(1)}%</td></tr>
                                <tr><td>FCF (Proxy)</td><td>${(activeTool.factors.fcf / 1e9).toFixed(1)}B</td></tr>
                                <tr><td>Rev (Reported)</td><td>${(activeTool.factors.revenue / 1e9).toFixed(1)}B</td></tr>
                              </tbody>
                            </table>
                          </div>
                        )}
                        {activeToolId === 'news' && (
                          <div className="news-list">
                            {Array.isArray(activeTool.factors) ? activeTool.factors.map((c, i) => (
                              <div key={i} className="news-item">
                                <span className={`news-score ${signalClass(c.score || 0)}`}>{(c.score || 0).toFixed(1)}</span>
                                <div className="news-content">
                                  <div className="news-title">{c.title || 'News Update'}</div>
                                  <div className="news-meta">{c.reasoning || 'Market movement detected.'}</div>
                                </div>
                              </div>
                            )) : <div style={{padding: '10px'}}>No recent catalysts mapped.</div>}
                          </div>
                        )}
                        {activeToolId === 'scenarios' && (
                          <div className="scenario-mini-table">
                            <div className="row"><span>Bull Case:</span> <b>${activeTool.factors.bull?.intrinsic_value}</b> <span className="signal-positive">+{(activeTool.factors.bull?.upside_pct * 100).toFixed(1)}%</span></div>
                            <div className="row"><span>Base Case:</span> <b>${activeTool.factors.base?.intrinsic_value}</b></div>
                            <div className="row"><span>Bear Case:</span> <b>${activeTool.factors.bear?.intrinsic_value}</b> <span className="signal-negative">{(activeTool.factors.bear?.downside_pct * 100).toFixed(1)}%</span></div>
                          </div>
                        )}
                        {activeToolId === 'peers' && (
                          <div className="peers-mini-report">
                            <div className="row"><span>Assessment:</span> <b className={signalClass(activeTool.score)}>{activeTool.factors.assessment}</b></div>
                            <div className="row"><span>Relative Premium:</span> <b>{(activeTool.factors.premium_pct * 100).toFixed(1)}%</b></div>
                            <div className="row"><span>Benchmarked Peers:</span> <span className="peer-tags">{activeTool.factors.peer_tickers?.map(p => <span key={p} className="peer-tag">{p}</span>)}</span></div>
                          </div>
                        )}
                        {activeToolId === 'growth' && (
                          <div className="growth-display">
                            <div className="big-percent">{(activeTool.factors.yoy * 100).toFixed(1)}%</div>
                            <div className="sub-label">Revenue Growth YoY (Normalized)</div>
                          </div>
                        )}
                      </div>
                    </article>
                    <article className="detail-card">
                      <h5>Tool Metadata</h5>
                      <div className="meta-info">
                        <p><b>Backend Tool:</b> {activeTool.toolMeta?.tool || 'Analytical Engine'}</p>
                        <p><b>Primary Source:</b> {activeTool.toolMeta?.source || 'SEC Filings/Market Data'}</p>
                        <p><b>Model:</b> {activeTool.id === 'tone' || activeTool.id === 'news' ? 'Gemini 2.0 Flash' : 'Quantitative DCF'}</p>
                        <p><b>Confidence:</b> {(activeTool.confidence * 100).toFixed(1)}%</p>
                      </div>
                    </article>
                  </div>

                  <div className="tool-ev-section">
                    <h5 className="sub-title" style={{ marginTop: '20px', marginBottom: '10px' }}>Evidence Sources</h5>
                    <div className="evidence-tabs" style={{ marginBottom: '15px' }}>
                      {['All', 'SEC', 'NEWS', 'CALL', 'DOC'].map(cat => (
                        <button 
                          key={cat} 
                          className={toolEvCategory === cat ? 'ev-tab active' : 'ev-tab'} 
                          onClick={() => setToolEvCategory(cat)}
                        >
                          {cat === 'SEC' ? 'SEC Filings' : cat === 'NEWS' ? 'News' : cat === 'CALL' ? 'Transcripts' : cat === 'DOC' ? 'Market Data' : 'All'}
                        </button>
                      ))}
                    </div>
                    
                    <div className="evidence-grid">
                      {activeTool.evidence.filter(ev => {
                        if (toolEvCategory === 'All') return true;
                        const meta = sourceMeta(ev);
                        if (toolEvCategory === 'SEC' && meta.icon === 'TBL') return true;
                        return meta.icon === toolEvCategory || (toolEvCategory === 'DOC' && meta.icon === 'DOC');
                      }).map((ev, idx) => {
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
                      {activeTool.evidence.filter(ev => {
                        if (toolEvCategory === 'All') return true;
                        if (toolEvCategory === 'SEC' && sourceMeta(ev).icon === 'TBL') return true;
                        return sourceMeta(ev).icon === toolEvCategory || (toolEvCategory === 'DOC' && sourceMeta(ev).icon === 'DOC');
                      }).length === 0 && (
                        <div style={{ padding: '20px', color: '#888' }}>No evidence matching this category.</div>
                      )}
                    </div>
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
              <article className="answer-card" style={{ marginBottom: '25px' }}>
                <h3 className="section-title">Research Analysis & Conclusion</h3>
                <p style={{ fontSize: '1.1rem', lineHeight: '1.6', marginBottom: '20px' }}>{resResult.answer}</p>
                <div className="kpi-grid kpi-compact">
                  <article className="kpi-card">
                    <div className="kpi-label">Confidence Level</div>
                    <div className="kpi-small">{(resResult.confidence * 100).toFixed(1)}%</div>
                  </article>
                  <article className={`kpi-card ${actionClass(resResult.action)}`}>
                    <div className="kpi-label">Recommended Action</div>
                    <div className="kpi-small">{resResult.action}</div>
                  </article>
                  <article className="kpi-card" style={{ flex: 2 }}>
                    <div className="kpi-label">Key Contributing Factor / Rationale</div>
                    <div className="kpi-small" style={{ fontSize: '1rem', fontWeight: 'normal' }}>{resResult.rationale}</div>
                  </article>
                </div>
              </article>

              <section className="evidence-panel">
                <h3 className="section-title">Verifiable Evidence</h3>
                 <div className="evidence-tabs" style={{ marginBottom: '15px' }}>
                  {['All', 'SEC', 'NEWS', 'CALL', 'DOC'].map(cat => (
                    <button 
                      key={cat} 
                      className={resEvCategory === cat ? 'ev-tab active' : 'ev-tab'} 
                      onClick={() => setResEvCategory(cat)}
                    >
                      {cat === 'SEC' ? 'SEC Filings' : cat === 'NEWS' ? 'News' : cat === 'CALL' ? 'Transcripts' : cat === 'DOC' ? 'Market Data' : 'All'}
                    </button>
                  ))}
                </div>

                <div className="evidence-grid">
                  {resResult.evidence.filter(ev => {
                    if (resEvCategory === 'All') return true;
                    const meta = sourceMeta(ev);
                    if (resEvCategory === 'SEC' && meta.icon === 'TBL') return true;
                    return meta.icon === resEvCategory || (resEvCategory === 'DOC' && meta.icon === 'DOC');
                  }).map((ev, idx) => {
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
                  {resResult.evidence.filter(ev => {
                    if (resEvCategory === 'All') return true;
                    if (resEvCategory === 'SEC' && sourceMeta(ev).icon === 'TBL') return true;
                    return sourceMeta(ev).icon === resEvCategory || (resEvCategory === 'DOC' && sourceMeta(ev).icon === 'DOC');
                  }).length === 0 && (
                    <div style={{ padding: '20px', color: '#888' }}>No evidence matching this category.</div>
                  )}
                </div>
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
