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

const REASON_LABELS = {
  'insufficient_source_diversity': 'Insufficient evidence sources — try lowering strictness or changing mode.',
  'missing_market_inputs': 'Market data unavailable for this ticker.',
  'insufficient_blocks': 'Not enough evidence retrieved — try a more specific query.',
  'required_source_missing:news': 'News data not available.',
  'required_source_missing:transcript': 'Earnings transcript not available.',
  'required_source_missing:filing': 'SEC filing data not available.',
  'missing_slot:ticker': 'No ticker identified — include a stock symbol in your query.',
  'missing_slot:fiscal_year': 'Fiscal year not identified.',
};

function ModeStructuredPanel({ mode, result }) {
  if (!result) return null;

  // lookup_numeric - show the numeric value prominently
  if ((mode === 'lookup_numeric' || mode === 'auto') && result.numeric) {
    const { metric, value, unit, notes, citation } = result.numeric;
    return (
      <article className="answer-card" style={{ marginBottom: '25px', padding: '30px', background: 'linear-gradient(135deg, #1a3a3a 0%, #0f2828 100%)' }}>
        <h3 className="section-title">{metric || 'Metric Result'}</h3>
        <div className="numeric-display" style={{ textAlign: 'center', padding: '40px 20px' }}>
          <div style={{ fontSize: '3.5rem', fontWeight: 'bold', color: '#00d4ff', marginBottom: '10px' }}>
            {typeof value === 'number' ? value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : value}
          </div>
          <div style={{ fontSize: '1.3rem', color: '#88ccff', marginBottom: '15px' }}>
            {unit === 'PERCENT' ? '%' : unit === 'USD' ? '(USD)' : unit ? `(${unit})` : ''}
          </div>
          {notes && <p style={{ color: '#ccc', marginTop: '15px' }}>{notes}</p>}
          {citation && <p style={{ color: '#888', fontSize: '0.9rem', marginTop: '10px', fontStyle: 'italic' }}>Source: {citation}</p>}
        </div>
      </article>
    );
  }

  // compute_metric - show formula and computed value
  if (mode === 'compute_metric' && result.computed) {
    const { metric, value, unit, formula, inputs } = result.computed;
    return (
      <article className="answer-card" style={{ marginBottom: '25px', padding: '30px' }}>
        <h3 className="section-title">{metric || 'Computed Metric'}</h3>
        <div className="metric-breakdown" style={{ marginBottom: '25px' }}>
          <h5 style={{ color: '#00d4ff', marginBottom: '15px' }}>Formula</h5>
          <code style={{ display: 'block', background: '#1a1a1a', padding: '15px', borderRadius: '4px', marginBottom: '15px', color: '#88ff88', overflowX: 'auto' }}>
            {formula || 'N/A'}
          </code>
          {inputs && inputs.length > 0 && (
            <div>
              <h5 style={{ color: '#00d4ff', marginBottom: '10px' }}>Inputs</h5>
              <table style={{ width: '100%', marginBottom: '20px' }}>
                <tbody>
                  {inputs.map((inp, i) => (
                    <tr key={i} style={{ borderBottom: '1px solid #333' }}>
                      <td style={{ padding: '10px', color: '#ccc' }}>{inp.name}</td>
                      <td style={{ padding: '10px', color: '#00d4ff', textAlign: 'right' }}>
                        {typeof inp.value === 'number' ? inp.value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : inp.value} {inp.unit}
                      </td>
                      <td style={{ padding: '10px', color: '#888', fontSize: '0.9rem' }}>Src: {inp.citation || 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          <h5 style={{ color: '#00d4ff', marginBottom: '10px' }}>Computed Value</h5>
          <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#88ff88', background: '#0a1a0a', padding: '20px', borderRadius: '4px' }}>
            {typeof value === 'number' ? value.toLocaleString(undefined, { maximumFractionDigits: 4 }) : value} {unit}
          </div>
        </div>
      </article>
    );
  }

  // valuation - DCF table with inputs and outputs
  if (mode === 'valuation' && result.valuation) {
    const { verified_inputs, assumptions, outputs, valuation_gap_pct } = result.valuation;
    return (
      <article className="answer-card" style={{ marginBottom: '25px', padding: '30px' }}>
        <h3 className="section-title">DCF Valuation Analysis</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '25px', marginBottom: '25px' }}>
          <div>
            <h5 style={{ color: '#00d4ff', marginBottom: '15px' }}>Verified Inputs</h5>
            <table style={{ width: '100%', fontSize: '0.95rem' }}>
              <tbody>
                {verified_inputs && verified_inputs.map((inp, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #333' }}>
                    <td style={{ padding: '8px', color: '#ccc' }}>{inp.name}</td>
                    <td style={{ padding: '8px', color: '#88ccff', textAlign: 'right' }}>
                      ${(inp.value / 1e9).toFixed(2)}B
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div>
            <h5 style={{ color: '#00d4ff', marginBottom: '15px' }}>Assumptions</h5>
            <table style={{ width: '100%', fontSize: '0.95rem' }}>
              <tbody>
                {assumptions && assumptions.map((ass, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #333' }}>
                    <td style={{ padding: '8px', color: '#ccc' }}>{ass.name}</td>
                    <td style={{ padding: '8px', color: '#ffcc88', textAlign: 'right' }}>
                      {typeof ass.value === 'number' ? (ass.value * 100).toFixed(2) + '%' : ass.value}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
        <div>
          <h5 style={{ color: '#00d4ff', marginBottom: '15px' }}>Valuation Outputs</h5>
          <table style={{ width: '100%', fontSize: '0.95rem' }}>
            <tbody>
              {outputs && outputs.map((out, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #333' }}>
                  <td style={{ padding: '10px', color: '#ccc' }}>{out.name}</td>
                  <td style={{ padding: '10px', color: '#00ff88', textAlign: 'right', fontWeight: 'bold' }}>
                    ${out.value.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {valuation_gap_pct !== undefined && (
            <div style={{ marginTop: '15px', padding: '15px', background: '#0a1a0a', borderRadius: '4px' }}>
              <span style={{ color: '#ccc' }}>Valuation Gap: </span>
              <span style={{ color: valuation_gap_pct > 0 ? '#00ff88' : '#ff8888', fontWeight: 'bold', fontSize: '1.2rem' }}>
                {(valuation_gap_pct * 100).toFixed(1)}%
              </span>
            </div>
          )}
        </div>
      </article>
    );
  }

  // relative_valuation - multiples comparison
  if (mode === 'relative_valuation' && result.relative_valuation) {
    const { multiple, numerator, denominator, value, peer_median, peer_premium_pct } = result.relative_valuation;
    return (
      <article className="answer-card" style={{ marginBottom: '25px', padding: '30px' }}>
        <h3 className="section-title">Relative Valuation ({multiple})</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '25px', marginBottom: '25px' }}>
          <div style={{ padding: '20px', background: '#1a2a1a', borderRadius: '4px' }}>
            <div style={{ color: '#888', fontSize: '0.9rem', marginBottom: '5px' }}>Numerator</div>
            <div style={{ color: '#88ff88', fontWeight: 'bold', fontSize: '1.5rem', marginBottom: '5px' }}>
              {typeof numerator.value === 'number' ? numerator.value.toLocaleString() : numerator.value}
            </div>
            <div style={{ color: '#aaa', fontSize: '0.85rem' }}>{numerator.name}</div>
          </div>
          <div style={{ padding: '20px', background: '#1a1a2a', borderRadius: '4px' }}>
            <div style={{ color: '#888', fontSize: '0.9rem', marginBottom: '5px' }}>Denominator</div>
            <div style={{ color: '#88ccff', fontWeight: 'bold', fontSize: '1.5rem', marginBottom: '5px' }}>
              {typeof denominator.value === 'number' ? denominator.value.toLocaleString(undefined, { maximumFractionDigits: 2 }) : denominator.value}
            </div>
            <div style={{ color: '#aaa', fontSize: '0.85rem' }}>{denominator.name}</div>
          </div>
        </div>
        <div style={{ padding: '25px', background: '#0a2a0a', borderRadius: '4px', marginBottom: '15px', textAlign: 'center' }}>
          <div style={{ color: '#888', marginBottom: '10px' }}>Current Multiple</div>
          <div style={{ fontSize: '2.5rem', color: '#00ff88', fontWeight: 'bold' }}>
            {typeof value === 'number' ? value.toFixed(2) : value}x
          </div>
        </div>
        {peer_median !== null && peer_median !== undefined && (
          <div style={{ padding: '15px', background: '#2a1a0a', borderRadius: '4px' }}>
            <div style={{ marginBottom: '10px' }}>
              <span style={{ color: '#ccc' }}>Peer Median: </span>
              <span style={{ color: '#ffaa66', fontWeight: 'bold' }}>{peer_median.toFixed(2)}x</span>
            </div>
            {peer_premium_pct !== null && peer_premium_pct !== undefined && (
              <div>
                <span style={{ color: '#ccc' }}>Premium/(Discount): </span>
                <span style={{ color: peer_premium_pct > 0 ? '#ff8888' : '#88ff88', fontWeight: 'bold' }}>
                  {(peer_premium_pct * 100).toFixed(1)}%
                </span>
              </div>
            )}
          </div>
        )}
      </article>
    );
  }

  // scenario_analysis - bull/base/bear scenarios
  if (mode === 'scenario_analysis' && result.scenario_analysis) {
    const { base_ev, comparisons } = result.scenario_analysis;
    const baseData = result.valuation || base_ev;
    const bullData = comparisons?.[0];
    const bearData = comparisons?.[1];

    return (
      <article className="answer-card" style={{ marginBottom: '25px', padding: '30px' }}>
        <h3 className="section-title">Scenario Analysis</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px' }}>
          <div style={{ padding: '20px', background: '#1a1a1a', borderRadius: '4px', borderLeft: '4px solid #ffcc88' }}>
            <div style={{ color: '#ffcc88', fontWeight: 'bold', marginBottom: '15px', fontSize: '1.1rem' }}>Base Case</div>
            {baseData?.intrinsic_value && (
              <>
                <div style={{ fontSize: '0.85rem', color: '#999' }}>Intrinsic Value</div>
                <div style={{ fontSize: '1.5rem', color: '#ffcc88', fontWeight: 'bold', marginBottom: '15px' }}>
                  ${baseData.intrinsic_value.toLocaleString(undefined, { maximumFractionDigits: 2 })}
                </div>
              </>
            )}
            {baseData?.intrinsic_value_per_share && (
              <div style={{ fontSize: '0.9rem', color: '#ccc' }}>Per Share: ${baseData.intrinsic_value_per_share.toFixed(2)}</div>
            )}
          </div>

          <div style={{ padding: '20px', background: '#0a2a0a', borderRadius: '4px', borderLeft: '4px solid #88ff88' }}>
            <div style={{ color: '#88ff88', fontWeight: 'bold', marginBottom: '15px', fontSize: '1.1rem' }}>Bull Case</div>
            {bullData?.ev && (
              <>
                <div style={{ fontSize: '0.85rem', color: '#999' }}>Enterprise Value</div>
                <div style={{ fontSize: '1.5rem', color: '#88ff88', fontWeight: 'bold', marginBottom: '15px' }}>
                  ${bullData.ev.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </div>
              </>
            )}
            {bullData?.ev_delta_pct && (
              <div style={{ fontSize: '0.9rem', color: '#88ff88' }}>
                Upside: <strong>+{(bullData.ev_delta_pct * 100).toFixed(1)}%</strong>
              </div>
            )}
          </div>

          <div style={{ padding: '20px', background: '#2a0a0a', borderRadius: '4px', borderLeft: '4px solid #ff8888' }}>
            <div style={{ color: '#ff8888', fontWeight: 'bold', marginBottom: '15px', fontSize: '1.1rem' }}>Bear Case</div>
            {bearData?.ev && (
              <>
                <div style={{ fontSize: '0.85rem', color: '#999' }}>Enterprise Value</div>
                <div style={{ fontSize: '1.5rem', color: '#ff8888', fontWeight: 'bold', marginBottom: '15px' }}>
                  ${bearData.ev.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                </div>
              </>
            )}
            {bearData?.ev_delta_pct && (
              <div style={{ fontSize: '0.9rem', color: '#ff8888' }}>
                Downside: <strong>{(bearData.ev_delta_pct * 100).toFixed(1)}%</strong>
              </div>
            )}
          </div>
        </div>
      </article>
    );
  }

  // peer_analysis - peer comparison table
  if (mode === 'peer_analysis' && result.peer_analysis) {
    const { target_ticker, peer_median, peer_premium_pct, signal } = result.peer_analysis;
    return (
      <article className="answer-card" style={{ marginBottom: '25px', padding: '30px' }}>
        <h3 className="section-title">Peer Analysis: {target_ticker}</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '25px' }}>
          <div style={{ padding: '20px', background: '#1a2a1a', borderRadius: '4px' }}>
            <div style={{ color: '#888', marginBottom: '10px' }}>Peer Median Multiple</div>
            <div style={{ fontSize: '2rem', color: '#88ff88', fontWeight: 'bold', marginBottom: '15px' }}>
              {peer_median !== null && peer_median !== undefined ? peer_median.toFixed(2) : 'N/A'}x
            </div>
            <div style={{ color: '#ccc', fontSize: '0.9rem' }}>Based on industry comps</div>
          </div>
          <div style={{ padding: '20px', background: peer_premium_pct > 0 ? '#2a1a0a' : '#0a2a0a', borderRadius: '4px' }}>
            <div style={{ color: '#888', marginBottom: '10px' }}>Valuation Premium</div>
            <div style={{ fontSize: '2rem', color: peer_premium_pct > 0 ? '#ff8888' : '#88ff88', fontWeight: 'bold', marginBottom: '15px' }}>
              {peer_premium_pct !== null && peer_premium_pct !== undefined ? (peer_premium_pct * 100).toFixed(1) : 'N/A'}%
            </div>
            <div style={{ color: '#ccc', fontSize: '0.9rem' }}>
              {peer_premium_pct > 0 ? 'Premium to Peers' : peer_premium_pct < 0 ? 'Discount to Peers' : 'At Peer Median'}
            </div>
          </div>
        </div>
        {signal && (
          <div style={{ marginTop: '20px', padding: '15px', background: '#1a1a1a', borderRadius: '4px' }}>
            <div style={{ color: '#00d4ff', fontWeight: 'bold', marginBottom: '10px' }}>Signal Assessment</div>
            <div style={{ color: '#ccc' }}>{JSON.stringify(signal).substring(0, 200)}</div>
          </div>
        )}
      </article>
    );
  }

  // comparative_analysis - side-by-side comparison
  if (mode === 'comparative_analysis' && result.comparison) {
    const { targets, facts, summary } = result.comparison;
    return (
      <article className="answer-card" style={{ marginBottom: '25px', padding: '30px' }}>
        <h3 className="section-title">Comparative Analysis</h3>
        {summary && <p style={{ color: '#ccc', marginBottom: '20px' }}>{summary}</p>}
        {facts && facts.length > 0 && (
          <table style={{ width: '100%', marginTop: '20px' }}>
            <tbody>
              {facts.map((fact, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #333' }}>
                  <td style={{ padding: '10px', color: '#88ccff', width: '30%' }}>{fact.entity}</td>
                  <td style={{ padding: '10px', color: '#ccc' }}>{fact.metric || fact.topic}</td>
                  <td style={{ padding: '10px', color: '#00d4ff', textAlign: 'right' }}>{fact.value_or_summary}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </article>
    );
  }

  // risk_analysis - risk factor cards
  if (mode === 'risk_analysis' && result.risks) {
    return (
      <article className="answer-card" style={{ marginBottom: '25px', padding: '30px' }}>
        <h3 className="section-title">Risk Analysis</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '15px' }}>
          {result.risks.map((risk, i) => (
            <div key={i} style={{ padding: '15px', background: '#1a1a1a', borderRadius: '4px', borderLeft: '4px solid #ff8888' }}>
              <div style={{ color: '#ff8888', fontWeight: 'bold', marginBottom: '10px' }}>{risk.risk}</div>
              <div style={{ color: '#ccc', fontSize: '0.9rem', marginBottom: '10px' }}>{risk.mechanism}</div>
              {risk.citations && risk.citations.length > 0 && (
                <div style={{ color: '#888', fontSize: '0.8rem' }}>Sources: {risk.citations.join(', ')}</div>
              )}
            </div>
          ))}
        </div>
      </article>
    );
  }

  // mba_framework - SWOT analysis
  if (mode === 'mba_framework' && result.framework) {
    const { type, bullets } = result.framework;
    const groupedByBucket = {};
    if (bullets) {
      bullets.forEach(b => {
        if (!groupedByBucket[b.bucket]) groupedByBucket[b.bucket] = [];
        groupedByBucket[b.bucket].push(b);
      });
    }
    const order = type === 'SWOT' ? ['Strengths', 'Weaknesses', 'Opportunities', 'Threats'] : ['Threat', 'Opportunity', 'Strength', 'Weakness'];
    const swotColors = { 'Strengths': '#88ff88', 'Weaknesses': '#ff8888', 'Opportunities': '#88ccff', 'Threats': '#ffaa66', Strength: '#88ff88', Weakness: '#ff8888', Opportunity: '#88ccff', Threat: '#ffaa66' };

    return (
      <article className="answer-card" style={{ marginBottom: '25px', padding: '30px' }}>
        <h3 className="section-title">{type} Analysis</h3>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
          {order.map(bucket => {
            const items = groupedByBucket[bucket] || [];
            if (items.length === 0) return null;
            return (
              <div key={bucket} style={{ padding: '20px', background: '#1a1a1a', borderRadius: '4px', borderTop: `3px solid ${swotColors[bucket] || '#888'}` }}>
                <h5 style={{ color: swotColors[bucket] || '#888', marginBottom: '15px', fontSize: '1.1rem' }}>{bucket}</h5>
                <ul style={{ listStyle: 'none', padding: 0 }}>
                  {items.map((item, i) => (
                    <li key={i} style={{ color: '#ccc', marginBottom: '10px', paddingLeft: '20px', position: 'relative' }}>
                      <span style={{ position: 'absolute', left: 0, color: swotColors[bucket] || '#888' }}>•</span>
                      {item.text}
                      {item.citations && item.citations.length > 0 && (
                        <div style={{ color: '#888', fontSize: '0.8rem', marginTop: '5px' }}>Src: {item.citations.slice(0, 2).join(', ')}</div>
                      )}
                    </li>
                  ))}
                </ul>
              </div>
            );
          })}
        </div>
      </article>
    );
  }

  // multi_period_analysis - trend table
  if (mode === 'multi_period_analysis' && result.multi_period_analysis) {
    const { periods, trend_summary } = result.multi_period_analysis;
    return (
      <article className="answer-card" style={{ marginBottom: '25px', padding: '30px' }}>
        <h3 className="section-title">Multi-Period Trend Analysis</h3>
        {periods && periods.length > 0 && (
          <table style={{ width: '100%', marginBottom: '20px' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid #00d4ff' }}>
                <th style={{ padding: '10px', color: '#00d4ff', textAlign: 'left' }}>Period</th>
                <th style={{ padding: '10px', color: '#00d4ff', textAlign: 'left' }}>Summary</th>
                <th style={{ padding: '10px', color: '#00d4ff', textAlign: 'right' }}>YoY Change</th>
              </tr>
            </thead>
            <tbody>
              {periods.map((period, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #333' }}>
                  <td style={{ padding: '10px', color: '#88ccff', fontWeight: 'bold' }}>FY{period.fiscal_year}</td>
                  <td style={{ padding: '10px', color: '#ccc' }}>{period.summary.substring(0, 80)}{period.summary.length > 80 ? '...' : ''}</td>
                  <td style={{ padding: '10px', color: '#888', textAlign: 'right', fontSize: '0.9rem' }}></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {trend_summary && <p style={{ color: '#aaa', fontStyle: 'italic' }}>Trend: {trend_summary.substring(0, 200)}</p>}
      </article>
    );
  }

  return null;
}

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

      // Improve rationale: use reason codes from verification if abstaining
      let rationale = 'Rationale derived from selected mode and evidence set.';
      if (raw.action === 'abstain' || raw.action === 'ABSTAIN') {
        const reasonCodes = raw.verification?.reason_codes || [];
        if (reasonCodes.length > 0) {
          const firstReason = reasonCodes[0];
          rationale = REASON_LABELS[firstReason] || firstReason;
        } else {
          rationale = raw.reason || 'Analysis could not be completed.';
        }
      } else if (raw.result?.final_answer) {
        // For successful answers, pull from the final answer or reasoning
        rationale = raw.result?.final_answer?.substring(0, 150) || rationale;
      }

      const answer = raw.result?.final_answer || (raw.action === 'abstain' || raw.action === 'ABSTAIN' ? `Analysis Abstained. ${rationale}` : 'Unable to generate answer.');
      const evidence = raw.evidence_hydrated || [];
      const gate = raw.verification?.gate || raw.result?.gate || {};

      setResResult({
        raw,
        answer,
        action: (raw.action || 'abstain').toUpperCase(),
        rationale,
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
              {resResult.raw.action !== 'ABSTAIN' && resResult.raw.action !== 'abstain' && (
                <ModeStructuredPanel mode={resMode} result={resResult.raw.result} />
              )}

              <article className="answer-card" style={{ marginBottom: '25px', padding: '30px' }}>
                <h3 className="section-title">Investment Research Report</h3>
                {(resResult.raw.action === 'ABSTAIN' || resResult.raw.action === 'abstain') && (
                  <div style={{ padding: '20px', background: '#2a1a0a', borderRadius: '4px', marginBottom: '20px', borderLeft: '4px solid #ffaa66' }}>
                    <h4 style={{ color: '#ffaa66', marginBottom: '10px' }}>⚠ Analysis Abstained</h4>
                    <p style={{ color: '#ccc', marginBottom: '5px' }}>Reason: {resResult.rationale}</p>
                    <p style={{ color: '#888', fontSize: '0.9rem' }}>Try adjusting strictness level, mode selection, or query specificity.</p>
                  </div>
                )}
                <div className="report-content" style={{ fontSize: '1.05rem', lineHeight: '1.7', color: '#e0e0e0', whiteSpace: 'pre-wrap' }}>
                  {resResult.answer.split('\n').map((line, i) => {
                    if (line.startsWith('###')) {
                      return <h4 key={i} style={{ color: '#00d4ff', marginTop: '20px', marginBottom: '10px', borderBottom: '1px solid #333', paddingBottom: '5px' }}>{line.replace('###', '').trim()}</h4>;
                    }
                    if (line.startsWith('-')) {
                      return <li key={i} style={{ marginLeft: '20px', marginBottom: '5px' }}>{line.replace('-', '').trim()}</li>;
                    }
                    // Simple bolding replace
                    const bolded = line.split(/(\*\*.*?\*\*)/g).map((part, j) => {
                      if (part.startsWith('**') && part.endsWith('**')) {
                        return <strong key={j} style={{ color: '#fff' }}>{part.slice(2, -2)}</strong>;
                      }
                      return part;
                    });
                    return <p key={i} style={{ marginBottom: '10px' }}>{bolded}</p>;
                  })}
                </div>
                
                <div className="kpi-grid kpi-compact" style={{ marginTop: '30px', borderTop: '1px solid #333', paddingTop: '20px' }}>
                  <article className="kpi-card">
                    <div className="kpi-label">Evidence Confidence</div>
                    <div className="kpi-small">{(resResult.confidence * 100).toFixed(1)}%</div>
                  </article>
                  <article className={`kpi-card ${actionClass(resResult.action)}`}>
                    <div className="kpi-label">System Recommendation</div>
                    <div className="kpi-small">{resResult.action}</div>
                  </article>
                  <article className="kpi-card" style={{ flex: 2 }}>
                    <div className="kpi-label">Execution Rationale</div>
                    <div className="kpi-small" style={{ fontSize: '0.95rem', fontWeight: 'normal', fontStyle: 'italic' }}>{resResult.rationale}</div>
                  </article>
                </div>
              </article>

              <section className="evidence-panel">
                <h3 className="section-title">Research Evidence Traceability</h3>
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
