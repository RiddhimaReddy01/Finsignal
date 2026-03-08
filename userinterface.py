"""
userinterface.py — FinSight Financial Intelligence Terminal
Professional dark UI: sticky header · central query · animated evidence · signal dashboard
"""
from __future__ import annotations

import time
import os
import re
import json
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import asdict, is_dataclass

import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# ENV / IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

from local_llm import build_local_llm_client, DEFAULT_PRIMARY_MODEL, DEFAULT_FALLBACK_MODEL
from market_api import YahooFinanceMarketDataProvider
from news_client_adapter import build_optional_news_client
from orchestrator import FinancialOrchestrator, OrchestratorConfig
from verification import Mode

try:
    from knowledge_base import TICKERS as KB_TICKERS, TARGET_FYS as KB_TARGET_FYS
except Exception:
    KB_TICKERS = []
    KB_TARGET_FYS = []

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

ALL_MODES: list[str] = ["auto"] + list(Mode.__args__)  # type: ignore[attr-defined]
UI_TICKERS: list[str] = sorted({str(t).strip().upper() for t in KB_TICKERS if str(t).strip()})
UI_FISCAL_YEARS: list[int] = sorted({int(y) for y in KB_TARGET_FYS if isinstance(y, int)})


# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM CSS
# ─────────────────────────────────────────────────────────────────────────────

DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* ── TOKENS ───────────────────────────────────────────── */
:root {
  --bg-base:       #0d1117;
  --bg-surface:    #161b22;
  --bg-elevated:   #21262d;
  --bg-overlay:    #30363d;
  --border:        rgba(240,246,252,0.10);
  --border-sub:    rgba(240,246,252,0.055);
  --text-1:        #e6edf3;
  --text-2:        #8b949e;
  --text-3:        #6e7681;
  --blue:          #58a6ff;
  --green:         #3fb950;
  --amber:         #e3b341;
  --orange:        #f0883e;
  --red:           #f85149;
  --purple:        #bc8cff;
  --cyan:          #39d3f5;
  --r-xs: 4px; --r-sm: 6px; --r-md: 10px; --r-lg: 16px; --r-xl: 22px;
  --shadow-sm: 0 1px 4px rgba(0,0,0,.35);
  --shadow-md: 0 4px 14px rgba(0,0,0,.45);
  --shadow-lg: 0 8px 28px rgba(0,0,0,.55);
}

/* ── BASE ─────────────────────────────────────────────── */
html, body, [class*="css"] {
  font-family: 'Manrope', 'Segoe UI', sans-serif !important;
  background: var(--bg-base) !important;
  color: var(--text-1) !important;
  font-size: 14px;
}
.block-container {
  padding: 0 1.5rem 4rem !important;
  max-width: 1400px !important;
}
h1,h2,h3,h4 { color: var(--text-1) !important; letter-spacing: -0.25px; }
p, li        { color: var(--text-2); line-height: 1.72; }
a            { color: var(--blue) !important; }
strong       { color: var(--text-1) !important; }
code, pre    { font-family: 'IBM Plex Mono', ui-monospace, monospace !important; font-size: 0.83em !important; }
hr           { border-color: var(--border) !important; }

/* ── SIDEBAR ──────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: var(--bg-surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { font-size: 0.84rem !important; }

/* ── STICKY HEADER ────────────────────────────────────── */
.fin-header {
  position: sticky; top: 0; z-index: 400;
  background: rgba(13,17,23,0.94);
  backdrop-filter: blur(14px) saturate(140%);
  -webkit-backdrop-filter: blur(14px) saturate(140%);
  border-bottom: 1px solid var(--border);
  margin: 0 -1.5rem 1.75rem -1.5rem;
  padding: 0 1.5rem;
}
.fin-header-inner {
  max-width: 1400px; margin: 0 auto;
  display: flex; align-items: center; height: 52px; gap: 18px;
}
.fin-logo {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 1.05rem; font-weight: 600;
  color: var(--blue) !important; white-space: nowrap; letter-spacing: -0.3px;
}
.fin-logo em { color: var(--text-3); font-style: normal; font-weight: 400; font-size: 0.78em; margin-left: 8px; }
.fin-hdr-spacer { flex: 1; }
.fin-hdr-crumb {
  font-size: 0.78rem; color: var(--text-3);
  font-family: 'IBM Plex Mono', monospace;
  max-width: 380px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.fin-hdr-status { display: flex; align-items: center; gap: 6px; }
.fin-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.fin-dot-green  { background: var(--green);  box-shadow: 0 0 5px var(--green); }
.fin-dot-amber  { background: var(--amber);  box-shadow: 0 0 5px var(--amber); }
.fin-dot-red    { background: var(--red);    box-shadow: 0 0 5px var(--red);   }
.fin-dot-blue   { background: var(--blue);   box-shadow: 0 0 5px var(--blue);  }
.fin-hdr-sep { width: 1px; height: 16px; background: var(--border); margin: 0 4px; }
.fin-status-lbl { font-size: 0.72rem; color: var(--text-3); }

/* ── CARDS ────────────────────────────────────────────── */
.fin-card {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  padding: 16px 20px; margin-bottom: 1rem;
}
.fin-card-elevated {
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  padding: 16px 20px; margin-bottom: 1rem;
  box-shadow: var(--shadow-sm);
}
.fin-section-label {
  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.1em;
  text-transform: uppercase; color: var(--text-3); margin-bottom: 10px;
}

/* ── QUERY ZONE ───────────────────────────────────────── */
.fin-query-zone {
  background: linear-gradient(145deg, rgba(22,27,34,0.98) 0%, rgba(33,38,45,0.95) 100%);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  padding: 16px 22px 14px;
  margin-bottom: 1.25rem;
  box-shadow: var(--shadow-md);
  position: relative; overflow: hidden;
}
.fin-query-zone::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--blue), var(--purple), var(--cyan));
  opacity: 0.7;
}
.fin-query-title {
  font-size: 0.65rem; font-weight: 700; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--blue); margin-bottom: 10px;
}

/* ── BADGES ───────────────────────────────────────────── */
.fin-badge {
  display: inline-block; padding: 2px 8px; border-radius: 999px;
  font-size: 0.7rem; font-weight: 600; letter-spacing: 0.05em;
  text-transform: uppercase; white-space: nowrap;
}
.fin-badge-blue   { background: rgba(88,166,255,.14);  color: var(--blue);   border: 1px solid rgba(88,166,255,.3);   }
.fin-badge-green  { background: rgba(63,185,80,.14);   color: var(--green);  border: 1px solid rgba(63,185,80,.3);    }
.fin-badge-amber  { background: rgba(227,179,65,.14);  color: var(--amber);  border: 1px solid rgba(227,179,65,.3);   }
.fin-badge-orange { background: rgba(240,136,62,.14);  color: var(--orange); border: 1px solid rgba(240,136,62,.3);   }
.fin-badge-red    { background: rgba(248,81,73,.14);   color: var(--red);    border: 1px solid rgba(248,81,73,.3);    }
.fin-badge-purple { background: rgba(188,140,255,.14); color: var(--purple); border: 1px solid rgba(188,140,255,.3);  }
.fin-badge-cyan   { background: rgba(57,211,245,.14);  color: var(--cyan);   border: 1px solid rgba(57,211,245,.3);   }
.fin-badge-muted  { background: rgba(110,118,129,.12); color: var(--text-2); border: 1px solid rgba(110,118,129,.2);  }

/* ── STICKY ANSWER SUMMARY BAR ────────────────────────── */
.fin-answer-bar {
  position: sticky; top: 52px; z-index: 300;
  background: rgba(13,17,23,0.96);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  padding: 9px 16px;
  margin-bottom: 14px;
  display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
  box-shadow: var(--shadow-sm);
  animation: fadeSlideDown 0.3s ease both;
}
.fin-bar-sep { width: 1px; height: 22px; background: var(--border); flex-shrink: 0; }
.fin-bar-item { display: flex; flex-direction: column; gap: 1px; }
.fin-bar-lbl  { font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.07em; color: var(--text-3); }
.fin-bar-val  { font-size: 0.82rem; font-weight: 500; color: var(--text-1); font-family: 'IBM Plex Mono', monospace; }
.fin-bar-preview {
  flex: 1; min-width: 160px;
  font-size: 0.81rem; color: var(--text-2);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  font-style: italic;
}

/* ── ANSWER BODY ──────────────────────────────────────── */
.fin-answer-body {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-left: 3px solid var(--blue);
  border-radius: var(--r-md);
  padding: 22px 26px;
  margin-bottom: 14px;
  animation: fadeSlideUp 0.4s ease both;
}
.fin-answer-text {
  font-size: 1rem; line-height: 1.78; color: var(--text-1);
  max-width: 900px;
}
.fin-answer-text p { color: var(--text-1) !important; margin-bottom: 0.75em; }
.fin-answer-meta {
  display: flex; gap: 7px; flex-wrap: wrap;
  margin-top: 14px; padding-top: 12px; border-top: 1px solid var(--border-sub);
}
.fin-assumptions {
  margin-top: 10px; font-size: 0.8rem; color: var(--text-3);
  font-family: 'IBM Plex Mono', monospace;
}

/* ── SIGNAL DASHBOARD ─────────────────────────────────── */
.fin-signal-wrap {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  border-radius: var(--r-lg);
  padding: 20px 24px;
  margin-bottom: 1rem;
  animation: fadeSlideUp 0.45s ease 0.05s both;
}
.fin-rec-box {
  border: 2px solid; border-radius: var(--r-md);
  padding: 14px 16px; text-align: center;
}
.fin-rec-BUY      { border-color: rgba(63,185,80,.5);   background: rgba(63,185,80,.08);   color: var(--green);  }
.fin-rec-HOLD     { border-color: rgba(227,179,65,.5);  background: rgba(227,179,65,.08);  color: var(--amber);  }
.fin-rec-CAUTIOUS { border-color: rgba(240,136,62,.5);  background: rgba(240,136,62,.08);  color: var(--orange); }
.fin-rec-AVOID    { border-color: rgba(248,81,73,.5);   background: rgba(248,81,73,.08);   color: var(--red);    }
.fin-rec-label { font-size: 0.66rem; text-transform: uppercase; letter-spacing: 0.1em; opacity: 0.65; margin-bottom: 4px; }
.fin-rec-value { font-size: 1.9rem; font-weight: 700; font-family: 'IBM Plex Mono', monospace; }
.fin-score-bar { height: 5px; border-radius: 3px; background: var(--bg-overlay); overflow: hidden; margin: 7px 0 4px; }
.fin-score-fill { height: 100%; border-radius: 3px; transition: width .6s cubic-bezier(.25,.8,.25,1); }
.fin-metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(88px, 1fr)); gap: 8px; margin: 14px 0 4px; }
.fin-metric-box {
  background: var(--bg-elevated); border: 1px solid var(--border-sub);
  border-radius: var(--r-sm); padding: 9px 10px; text-align: center;
}
.fin-metric-lbl { font-size: 0.62rem; color: var(--text-3); text-transform: uppercase; letter-spacing: 0.08em; }
.fin-metric-val { font-size: 1rem; font-weight: 600; font-family: 'IBM Plex Mono', monospace; margin-top: 3px; }
.fin-risk-row {
  padding: 7px 0; border-bottom: 1px solid var(--border-sub);
  font-size: 0.83rem; display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap;
}
.fin-risk-row:last-child { border-bottom: none; }
.fin-catalyst-row {
  padding: 4px 0; display: flex; align-items: flex-start; gap: 8px; font-size: 0.82rem;
}

/* ── EVIDENCE TABS ────────────────────────────────────── */
div[data-testid="stTabs"] > div:first-child {
  border-bottom: 1px solid var(--border) !important;
  gap: 4px;
}
div[data-testid="stTabs"] button[data-testid="stTab"] {
  font-size: 0.82rem !important; font-weight: 500 !important;
  color: var(--text-2) !important;
  padding: 7px 14px !important; border-radius: var(--r-sm) var(--r-sm) 0 0 !important;
  border: none !important; background: transparent !important;
}
div[data-testid="stTabs"] button[data-testid="stTab"][aria-selected="true"] {
  color: var(--blue) !important;
  background: rgba(88,166,255,.08) !important;
  border-bottom: 2px solid var(--blue) !important;
}

/* ── SOURCE CARDS ─────────────────────────────────────── */
.fin-src-card {
  background: var(--bg-elevated);
  border: 1px solid var(--border-sub);
  border-radius: var(--r-md);
  padding: 13px 15px; margin-bottom: 9px;
  transition: border-color .2s, box-shadow .2s, transform .15s;
  animation: srcSlideIn .35s ease both;
}
.fin-src-card:hover {
  border-color: rgba(88,166,255,.4);
  box-shadow: 0 0 14px rgba(88,166,255,.12);
  transform: translateY(-1px);
}
.fin-src-card:nth-child(1)  { animation-delay: .04s; }
.fin-src-card:nth-child(2)  { animation-delay: .08s; }
.fin-src-card:nth-child(3)  { animation-delay: .12s; }
.fin-src-card:nth-child(4)  { animation-delay: .16s; }
.fin-src-card:nth-child(5)  { animation-delay: .20s; }
.fin-src-card:nth-child(6)  { animation-delay: .24s; }
.fin-src-card:nth-child(7)  { animation-delay: .28s; }
.fin-src-card:nth-child(8)  { animation-delay: .32s; }
.fin-src-card:nth-child(9)  { animation-delay: .36s; }
.fin-src-card:nth-child(10) { animation-delay: .40s; }
/* Cited source — animated highlight for top/key sources */
.fin-src-card.fin-src-cited {
  border-color: rgba(88,166,255,.45);
  box-shadow: 0 0 0 1px rgba(88,166,255,.2);
  animation: srcSlideIn .35s ease both, pulseGlow 2.5s ease-in-out 0.4s 2;
}
.fin-src-card.fin-src-cited:hover {
  box-shadow: 0 0 20px rgba(88,166,255,.3), 0 0 0 1px rgba(88,166,255,.35);
}
.fin-src-idx {
  background: var(--bg-overlay); border: 1px solid var(--border);
  border-radius: var(--r-xs); min-width: 22px; height: 22px;
  display: inline-flex; align-items: center; justify-content: center;
  font-size: 0.68rem; font-weight: 600; color: var(--blue);
  font-family: 'IBM Plex Mono', monospace; flex-shrink: 0;
}
.fin-src-header { display: flex; align-items: flex-start; gap: 9px; margin-bottom: 7px; }
.fin-src-title  { font-size: 0.87rem; font-weight: 500; color: var(--text-1); flex: 1; line-height: 1.4; }
.fin-src-excerpt {
  font-size: 0.79rem; color: var(--text-2); line-height: 1.6;
  padding: 8px 10px; background: rgba(0,0,0,.2); border-radius: var(--r-sm);
  border-left: 2px solid rgba(88,166,255,.3); font-family: 'IBM Plex Mono', monospace;
  margin: 7px 0; white-space: pre-wrap; word-break: break-word;
}
.fin-src-footer {
  display: flex; align-items: center; gap: 8px; margin-top: 8px;
  padding-top: 8px; border-top: 1px solid var(--border-sub); font-size: 0.72rem; flex-wrap: wrap;
}
.fin-src-url { color: var(--text-3); flex: 1; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.fin-open-btn {
  color: var(--blue) !important; font-size: 0.72rem !important; font-weight: 500;
  padding: 3px 8px; border: 1px solid rgba(88,166,255,.3); border-radius: var(--r-xs);
  background: rgba(88,166,255,.08); text-decoration: none !important;
  transition: background .2s, border-color .2s;
}
.fin-open-btn:hover { background: rgba(88,166,255,.18) !important; border-color: rgba(88,166,255,.5); }

/* ── XBRL FACT CARD ───────────────────────────────────── */
.fin-xbrl-card {
  background: var(--bg-elevated); border: 1px solid var(--border-sub);
  border-radius: var(--r-sm); padding: 10px 12px; margin-bottom: 7px;
  display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  animation: srcSlideIn .3s ease both;
}
.fin-xbrl-concept { font-size: 0.85rem; font-weight: 500; color: var(--text-1); flex: 1; }
.fin-xbrl-value {
  font-family: 'IBM Plex Mono', monospace; font-size: 0.95rem;
  font-weight: 600; color: var(--cyan);
}
.fin-xbrl-unit { font-size: 0.72rem; color: var(--text-3); margin-left: 4px; }

/* ── TRACE / AUDIT ────────────────────────────────────── */
.fin-trace-panel {
  background: var(--bg-elevated); border: 1px solid rgba(88,166,255,.15);
  border-radius: var(--r-md); overflow: hidden;
}
.fin-trace-row {
  display: flex; align-items: flex-start; gap: 12px;
  padding: 7px 14px; border-bottom: 1px solid var(--border-sub);
  font-size: 0.81rem;
}
.fin-trace-row:last-child { border-bottom: none; }
.fin-trace-key { color: var(--text-3); min-width: 120px; font-family: 'IBM Plex Mono', monospace; font-size: 0.76rem; padding-top: 1px; }
.fin-trace-val { color: var(--text-1); flex: 1; word-break: break-word; }

/* ── REASONING PANEL ──────────────────────────────────── */
.fin-reasoning {
  background: var(--bg-elevated); border: 1px solid var(--border);
  border-radius: var(--r-md); overflow: hidden;
}
.fin-rsn-row {
  display: flex; align-items: flex-start; gap: 12px;
  padding: 8px 14px; border-bottom: 1px solid var(--border-sub); font-size: 0.82rem;
}
.fin-rsn-row:last-child { border-bottom: none; }
.fin-rsn-key { color: var(--text-3); min-width: 150px; font-size: 0.75rem; font-family: 'IBM Plex Mono', monospace; padding-top: 2px; }
.fin-rsn-val { flex: 1; color: var(--text-1); flex-wrap: wrap; }
.fin-reason-tag {
  display: inline-block; padding: 2px 7px; border-radius: var(--r-xs); margin: 2px;
  font-size: 0.71rem; font-family: 'IBM Plex Mono', monospace;
  background: rgba(88,166,255,.1); color: var(--blue); border: 1px solid rgba(88,166,255,.2);
}
.fin-warn-tag {
  display: inline-block; padding: 2px 7px; border-radius: var(--r-xs); margin: 2px;
  font-size: 0.71rem; font-family: 'IBM Plex Mono', monospace;
  background: rgba(227,179,65,.1); color: var(--amber); border: 1px solid rgba(227,179,65,.2);
}

/* ── DOCUMENT DRAWER (SEC excerpt preview) ───────────────── */
.fin-drawer {
  background: var(--bg-surface); border: 1px solid rgba(88,166,255,.3);
  border-radius: var(--r-lg); overflow: hidden; margin-bottom: 1.2rem;
  box-shadow: 0 4px 24px rgba(0,0,0,.4), 0 0 40px rgba(88,166,255,.06);
  animation: drawerDrop .35s cubic-bezier(.22,1,.36,1) both;
}
.fin-drawer-hdr {
  background: linear-gradient(180deg, var(--bg-elevated) 0%, var(--bg-surface) 100%);
  border-bottom: 1px solid var(--border);
  padding: 14px 20px; display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
}
.fin-drawer-title {
  font-size: 0.9rem; font-weight: 600; flex: 1; color: var(--text-1);
  font-family: 'IBM Plex Mono', monospace;
}
.fin-drawer-body { padding: 18px 20px; max-height: 520px; overflow-y: auto; font-size: 0.9rem; line-height: 1.7; }

/* ── EMPTY STATE ──────────────────────────────────────── */
.fin-empty {
  text-align: center; padding: 64px 20px;
  border: 1px dashed var(--border); border-radius: var(--r-xl);
  animation: fadeIn .5s ease both;
}
.fin-empty-icon  { font-size: 2.6rem; margin-bottom: 14px; line-height: 1; }
.fin-empty-title { font-size: 1.05rem; font-weight: 600; color: var(--text-1); margin-bottom: 6px; }
.fin-empty-sub   { font-size: 0.85rem; color: var(--text-3); }

/* ── STREAMLIT WIDGET OVERRIDES ───────────────────────── */
[data-testid="stTextInput"] input {
  background: var(--bg-elevated) !important; color: var(--text-1) !important;
  border: 1px solid var(--border) !important; border-radius: var(--r-md) !important;
  font-size: 0.97rem !important; padding: 10px 14px !important;
  transition: border-color .2s, box-shadow .2s !important;
}
[data-testid="stTextInput"] input:focus {
  border-color: var(--blue) !important;
  box-shadow: 0 0 0 3px rgba(88,166,255,.18) !important;
}
[data-testid="stTextInput"] label { color: var(--text-3) !important; font-size: 0.77rem !important; }

[data-testid="stSelectbox"] > div > div {
  background: var(--bg-elevated) !important; color: var(--text-1) !important;
  border: 1px solid var(--border) !important; border-radius: var(--r-sm) !important;
}
[data-testid="stSelectbox"] label { font-size: 0.77rem !important; color: var(--text-3) !important; }

[data-testid="stSlider"] [data-testid="stMarkdownContainer"] p {
  color: var(--text-3) !important; font-size: 0.77rem !important;
}
[data-testid="stSlider"] label { font-size: 0.77rem !important; color: var(--text-3) !important; }
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
  background: var(--blue) !important;
}

div[data-testid="stButton"] > button {
  background: var(--blue) !important; color: #000 !important;
  border: none !important; border-radius: var(--r-md) !important;
  font-weight: 600 !important; font-size: 0.88rem !important;
  padding: 10px 22px !important; letter-spacing: 0.02em;
  transition: all .2s !important;
}
div[data-testid="stButton"] > button:hover {
  background: #79bbff !important; transform: translateY(-1px) !important;
  box-shadow: 0 4px 14px rgba(88,166,255,.35) !important;
}
div[data-testid="stButton"] > button:active { transform: translateY(0) !important; }

[data-testid="stExpander"] > details {
  background: var(--bg-surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--r-md) !important;
}
[data-testid="stExpander"] > details summary {
  color: var(--text-1) !important; font-size: 0.86rem !important; font-weight: 500 !important;
  padding: 10px 14px !important;
}
[data-testid="stExpander"] > details summary:hover { background: rgba(255,255,255,.03) !important; }

[data-testid="stMetricLabel"] p  { color: var(--text-3) !important; font-size: 0.72rem !important; }
[data-testid="stMetricValue"]    { color: var(--text-1) !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 1.25rem !important; }

div[data-testid="stStatusWidget"] {
  background: var(--bg-surface) !important;
  border: 1px solid rgba(88,166,255,.25) !important;
  border-radius: var(--r-md) !important;
}
div[data-testid="stStatusWidget"] p { color: var(--text-2) !important; font-size: 0.84rem !important; }

/* Code blocks */
[data-testid="stCode"] pre {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border-sub) !important;
  border-radius: var(--r-sm) !important;
  font-size: 0.8rem !important; line-height: 1.6 !important;
  color: var(--text-2) !important;
}

/* Download buttons */
[data-testid="stDownloadButton"] > button {
  background: transparent !important; color: var(--blue) !important;
  border: 1px solid rgba(88,166,255,.3) !important; border-radius: var(--r-sm) !important;
  font-size: 0.8rem !important; padding: 6px 14px !important;
}
[data-testid="stDownloadButton"] > button:hover {
  background: rgba(88,166,255,.1) !important; transform: none !important; box-shadow: none !important;
}

/* Alerts */
[data-testid="stAlert"] { border-radius: var(--r-md) !important; }

/* ── ANIMATIONS ───────────────────────────────────────── */
@keyframes fadeIn       { from { opacity: 0 }           to { opacity: 1 } }
@keyframes fadeSlideUp  { from { opacity: 0; transform: translateY(12px) } to { opacity: 1; transform: translateY(0) } }
@keyframes fadeSlideDown{ from { opacity: 0; transform: translateY(-8px) } to { opacity: 1; transform: translateY(0) } }
@keyframes srcSlideIn   { from { opacity: 0; transform: translateY(7px) }  to { opacity: 1; transform: translateY(0) } }
@keyframes drawerDrop   { from { opacity: 0; transform: scaleY(.97) }       to { opacity: 1; transform: scaleY(1) } }
@keyframes pulseGlow {
  0%, 100% { box-shadow: 0 0 0 rgba(88,166,255,0), 0 0 0 1px rgba(88,166,255,.15); }
  50%       { box-shadow: 0 0 24px rgba(88,166,255,.35), 0 0 0 2px rgba(88,166,255,.4); }
}
@keyframes shimmer {
  0%   { background-position: -200% 0; }
  100% { background-position:  200% 0; }
}
.fin-shimmer {
  background: linear-gradient(90deg, var(--bg-surface) 25%, var(--bg-elevated) 50%, var(--bg-surface) 75%);
  background-size: 200% 100%; animation: shimmer 1.6s infinite;
  border-radius: var(--r-sm); height: 14px;
}

/* Workspace KPI tiles */
.fin-kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 10px;
  margin: 8px 0 14px;
}
.fin-kpi-tile {
  background: var(--bg-elevated);
  border: 1px solid var(--border);
  border-radius: var(--r-md);
  padding: 14px 16px;
}
.fin-kpi-lbl {
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.09em;
  color: var(--text-3);
  margin-bottom: 4px;
}
.fin-kpi-val {
  font-size: 1.35rem;
  font-weight: 700;
  color: var(--text-1);
  font-family: 'IBM Plex Mono', monospace;
  margin-bottom: 4px;
}
.fin-kpi-sub {
  font-size: 0.78rem;
  color: var(--text-2);
  line-height: 1.5;
}

/* Tool cards */
.fin-tool-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 10px;
  margin-bottom: 10px;
}
.fin-tool-card {
  background: var(--bg-elevated);
  border: 1px solid var(--border-sub);
  border-radius: var(--r-md);
  padding: 12px 14px;
}
.fin-tool-name {
  color: var(--text-1);
  font-size: 0.84rem;
  font-weight: 600;
  margin-bottom: 6px;
}
.fin-tool-row {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  align-items: center;
}
.fin-tool-mini {
  font-family: 'IBM Plex Mono', monospace;
  font-size: 0.77rem;
  color: var(--text-2);
}

/* Responsive layout tuning for tablet/mobile */
@media (max-width: 1100px) {
  .block-container { padding: 0 1rem 3.5rem !important; }
  .fin-header { margin: 0 -1rem 1.25rem -1rem; padding: 0 1rem; }
  .fin-hdr-crumb { max-width: 220px; }
  .fin-answer-text { max-width: 100%; }
}

@media (max-width: 820px) {
  .block-container { padding: 0 0.75rem 3rem !important; }
  .fin-header { margin: 0 -0.75rem 1rem -0.75rem; padding: 0 0.75rem; }
  .fin-header-inner { height: auto; min-height: 48px; padding: 8px 0; flex-wrap: wrap; gap: 10px; }
  .fin-logo { font-size: 0.92rem; }
  .fin-hdr-crumb { order: 3; max-width: 100%; font-size: 0.72rem; }
  .fin-query-zone { padding: 12px 14px 12px; border-radius: var(--r-md); }
  .fin-answer-body { padding: 16px 16px; }
  .fin-answer-text { font-size: 0.95rem; line-height: 1.7; }
  .fin-answer-bar {
    top: 48px;
    gap: 8px;
    padding: 8px 10px;
    margin-bottom: 10px;
  }
  .fin-bar-sep { display: none; }
  .fin-bar-item { min-width: 40%; }
  .fin-drawer-body { max-height: 380px; font-size: 0.86rem; }
}
</style>
"""

_EMPTY_HTML = """
<div class="fin-empty">
  <div class="fin-empty-icon">{icon}</div>
  <div class="fin-empty-title">{title}</div>
  <div class="fin-empty-sub">{sub}</div>
</div>
"""


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_orchestrator() -> FinancialOrchestrator:
    base_dir = Path(__file__).resolve().parent
    audit_log = base_dir / "logs" / "audit.jsonl"

    llm_client = build_local_llm_client(
        primary_model=os.environ.get("GEMINI_SMALL_MODEL", DEFAULT_PRIMARY_MODEL),
        fallback_model=os.environ.get("GEMINI_FALLBACK_MODEL", DEFAULT_FALLBACK_MODEL),
        base_url=os.environ.get("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com"),
    )
    cfg = OrchestratorConfig(
        base_dir=base_dir,
        audit_log_path=audit_log,
        small_model_name=os.environ.get("LOCAL_SMALL_MODEL", "small"),
        large_model_name=os.environ.get("LOCAL_LARGE_MODEL", "large"),
        known_tickers={t.upper() for t in KB_TICKERS if isinstance(t, str) and t.strip()} or None,
        market_provider=YahooFinanceMarketDataProvider(),
        news_client=build_optional_news_client(),
    )
    return FinancialOrchestrator(cfg=cfg, llm_client=llm_client)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _safe_to_dict(x: Any) -> Any:
    return asdict(x) if is_dataclass(x) else x


def _extract_final_answer(res: Dict[str, Any]) -> str:
    result = res.get("result", {}) or {}
    ans = result.get("final_answer", "")
    assumptions = res.get("assumptions", []) or []
    suffix = ("\n\n*Assumptions: " + "; ".join(str(a) for a in assumptions) + "*") if assumptions else ""
    if isinstance(ans, str) and ans.strip():
        text = ans.strip()
        sep = "--- Investment Signal:"
        if sep in text:
            text = text[: text.index(sep)].rstrip()
        return (text + suffix) if text else f"No final answer generated.{suffix}"
    reason = res.get("reason")
    if isinstance(reason, str) and reason.strip():
        return f"[{reason}]{suffix}"
    return f"No answer generated.{suffix}"


def _health_check(base_dir: Path) -> Dict[str, Any]:
    index_dir = base_dir / "index"
    required = [
        index_dir / "chunks.parquet",
        index_dir / "bm25.pkl",
        index_dir / "faiss.index",
        index_dir / "tables.parquet",
        index_dir / "table_bm25.pkl",
        index_dir / "table_faiss.index",
        base_dir / "data" / "xbrl_companyfacts",
    ]
    missing = [str(p) for p in required if not p.exists()]
    return {"ok": len(missing) == 0, "missing": missing}


def _flatten_evidence(res: Dict[str, Any]) -> Dict[str, Any]:
    ev = res.get("evidence", {}) or {}
    narrative = ev.get("narrative", {}) if isinstance(ev, dict) else {}
    tables = ev.get("tables", {}) if isinstance(ev, dict) else {}
    xbrl = ev.get("xbrl", {}) if isinstance(ev, dict) else {}
    return {"narrative": narrative, "tables": tables, "xbrl": xbrl}


def _read_audit_log(path: Path, tail: int = 300) -> str:
    if not path.exists():
        return ""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        return "\n".join(lines[-tail:] if tail and len(lines) > tail else lines)
    except Exception:
        return ""


def _available_fiscal_years(base_dir: Path) -> list[int]:
    years: set[int] = set()
    fy_re = re.compile(r"_FY(\d{4})_")
    for rel_dir in ("data/sections", "data/tables", "data/raw_html"):
        d = base_dir / rel_dir
        if not d.exists():
            continue
        for p in d.glob("*"):
            m = fy_re.search(p.name)
            if m:
                try:
                    years.add(int(m.group(1)))
                except Exception:
                    pass
    return sorted(years) if years else UI_FISCAL_YEARS


def _get_best_evidence(r: Dict) -> List[Dict]:
    """Extract evidence blocks from verification.best_evidence (preferred) or flatten."""
    raw = r.get("raw", {}) or {}
    ver = raw.get("verification", {}) or {}
    best = ver.get("best_evidence", []) or []
    if best:
        return [b if isinstance(b, dict) else _safe_to_dict(b) for b in best]
    return []


def _action_badge_cls(action: str) -> str:
    return {"answer": "green", "abstain": "amber", "clarify": "blue", "error": "red"}.get(action, "muted")


def _rec_cls(rec: str) -> str:
    return {"BUY": "BUY", "HOLD": "HOLD", "CAUTIOUS": "CAUTIOUS", "AVOID": "AVOID"}.get(rec.upper(), "HOLD")


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

def _llm_status() -> Tuple[str, bool]:
    """Returns (label, ok) for LLM provider status."""
    if (os.environ.get("GEMINI_API_KEY") or "").strip():
        return "Gemini", True
    if (os.environ.get("OLLAMA_HOST") or os.environ.get("OLLAMA_URL") or "").strip():
        return "Ollama", True  # Assume available if configured
    return "LLM", False


def render_header(hc: Dict, llm_ok: bool, llm_label: str = "LLM", crumb: str = "") -> None:
    idx_dot = "green" if hc["ok"] else "red"
    api_dot = "green" if llm_ok else "red"
    crumb_html = (
        f'<div class="fin-hdr-crumb">▸ {_esc(crumb[:80])}</div>'
        if crumb else ""
    )
    st.markdown(
        f"""
        <div class="fin-header">
          <div class="fin-header-inner">
            <div class="fin-logo">FinSignal AI<em>// Institutional Analysis Dashboard</em></div>
            {crumb_html}
            <div class="fin-hdr-spacer"></div>
            <div class="fin-hdr-status">
              <span class="fin-dot fin-dot-{idx_dot}"></span>
              <span class="fin-status-lbl">Index</span>
              <div class="fin-hdr-sep"></div>
              <span class="fin-dot fin-dot-{api_dot}"></span>
              <span class="fin-status-lbl">{_esc(llm_label)}</span>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ANSWER SECTION
# ─────────────────────────────────────────────────────────────────────────────

def render_answer_section(r: Dict) -> None:
    raw      = r.get("raw", {}) or {}
    action   = r.get("action", "abstain")
    routing  = r.get("routing") or {}
    ver      = raw.get("verification", {}) or {}
    conf     = ver.get("confidence")
    model    = routing.get("model", "—")
    ticker   = r.get("ticker") or "auto"
    fy       = r.get("fiscal_year") or "auto"
    mode_str = r.get("mode", "—")
    lat      = r.get("latency_s", 0)
    run_id   = r.get("run_id", "—")
    conf_str = f"{float(conf):.0%}" if conf is not None else "—"
    ans_prev = _esc((r.get("final_answer") or "")[:100])
    act_cls  = _action_badge_cls(action)
    raw_r    = r.get("raw", {}) or {}
    sig_rep  = raw_r.get("hackathon_signal_report") or {}
    sig_rec  = str(sig_rep.get("recommendation", "")).upper() if sig_rep else ""
    rec_cls  = _rec_cls(sig_rec)
    sig_col   = {"BUY": "green", "HOLD": "amber", "CAUTIOUS": "orange", "AVOID": "red"}.get(rec_cls, "muted")
    sig_badge = (f'<span class="fin-badge fin-badge-{sig_col}" style="margin-left:4px">Signal: {sig_rec}</span>' if sig_rec else "")

    # ── Sticky summary bar ──
    st.markdown(
        f"""
        <div class="fin-answer-bar">
          <div class="fin-bar-item">
            <div class="fin-bar-lbl">Action</div>
            <span class="fin-badge fin-badge-{act_cls}">{action.upper()}</span>
            {sig_badge}
          </div>
          <div class="fin-bar-sep"></div>
          <div class="fin-bar-item">
            <div class="fin-bar-lbl">Ticker / FY</div>
            <div class="fin-bar-val">{_esc(str(ticker))} / {_esc(str(fy))}</div>
          </div>
          <div class="fin-bar-sep"></div>
          <div class="fin-bar-item">
            <div class="fin-bar-lbl">Mode</div>
            <div class="fin-bar-val">{_esc(mode_str)}</div>
          </div>
          <div class="fin-bar-sep"></div>
          <div class="fin-bar-item">
            <div class="fin-bar-lbl">Model · Confidence · Latency</div>
            <div class="fin-bar-val">{_esc(model)} · {conf_str} · {lat}s</div>
          </div>
          <div class="fin-bar-sep"></div>
          <div class="fin-bar-preview">{ans_prev}…</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Full answer body ──
    st.markdown('<div class="fin-answer-body">', unsafe_allow_html=True)
    st.markdown('<div class="fin-answer-text">', unsafe_allow_html=True)
    st.markdown(r.get("final_answer", "*No answer generated.*"))
    st.markdown("</div>", unsafe_allow_html=True)

    # Validation warnings
    for err in (raw.get("validation_errors") or []):
        st.warning(f"Validation: {err}")

    # Meta badges
    st.markdown(
        f'<div class="fin-answer-meta">'
        f'<span class="fin-badge fin-badge-muted">run: {_esc(run_id)}</span>'
        f'<span class="fin-badge fin-badge-muted">mode: {_esc(mode_str)}</span>'
        f'<span class="fin-badge fin-badge-blue">model: {_esc(model)}</span>'
        f'<span class="fin-badge fin-badge-{"green" if act_cls == "green" else "muted"}">{action}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)  # fin-answer-body


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def render_signal_section(raw: Dict) -> None:
    report    = raw.get("hackathon_signal_report") or {}
    score_obj = raw.get("hackathon_signal_score") or {}
    if not report:
        return

    rec      = str(report.get("recommendation", "HOLD")).upper()
    rec_key  = _rec_cls(rec)
    strength = float(report.get("signal_strength", 0.0))
    conf     = float(report.get("confidence", 0.0))
    comp     = score_obj.get("component_scores", {}) or {}

    st.markdown(
        '<div class="fin-section-label" style="margin-top:1rem;margin-bottom:6px">Investment Signals</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Signal Dashboard", expanded=True):
        st.markdown('<div class="fin-signal-wrap" style="border:none;padding:0;margin:0">', unsafe_allow_html=True)

        # ── Top row: rec + strength + confidence ──
        c1, c2, c3, c4 = st.columns([1.2, 2, 2, 2])

        with c1:
            st.markdown(
                f'<div class="fin-rec-box fin-rec-{rec_key}">'
                f'<div class="fin-rec-label">Recommendation</div>'
                f'<div class="fin-rec-value">{rec}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

        with c2:
            bar_pct  = min(100, int(abs(strength) * 100))
            bar_col  = "var(--green)" if strength > 0 else "var(--red)" if strength < 0 else "var(--text-3)"
            st.markdown(
                f'<div style="padding:6px 0">'
                f'<div class="fin-metric-lbl" style="margin-bottom:6px">Signal Strength</div>'
                f'<div class="fin-score-bar"><div class="fin-score-fill" style="width:{bar_pct}%;background:{bar_col}"></div></div>'
                f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:1rem;font-weight:600;color:{bar_col}">{strength:+.3f}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

        with c3:
            conf_pct = int(conf * 100)
            st.markdown(
                f'<div style="padding:6px 0">'
                f'<div class="fin-metric-lbl" style="margin-bottom:6px">Confidence</div>'
                f'<div class="fin-score-bar"><div class="fin-score-fill" style="width:{conf_pct}%;background:var(--blue)"></div></div>'
                f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:1rem;font-weight:600;color:var(--blue)">{conf:.0%}</div>'
                f"</div>",
                unsafe_allow_html=True,
            )

        with c4:
            if comp:
                dominant = max(comp, key=lambda k: abs(float(comp[k])))
                dom_val  = float(comp[dominant])
                dom_col  = "var(--green)" if dom_val > 0 else "var(--red)"
                st.markdown(
                    f'<div style="padding:6px 0">'
                    f'<div class="fin-metric-lbl" style="margin-bottom:6px">Dominant Factor</div>'
                    f'<div style="font-size:1rem;font-weight:600;color:{dom_col}">{dominant.title()}</div>'
                    f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.82rem;color:{dom_col}">{dom_val:+.2f}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # ── Component score grid ──
        if comp:
            score_html = ""
            for k, v in comp.items():
                v_f  = float(v)
                col  = "var(--green)" if v_f > 0.05 else "var(--red)" if v_f < -0.05 else "var(--text-3)"
                score_html += (
                    f'<div class="fin-metric-box">'
                    f'<div class="fin-metric-lbl">{k.upper()}</div>'
                    f'<div class="fin-metric-val" style="color:{col}">{v_f:+.2f}</div>'
                    f"</div>"
                )
            st.markdown(f'<div class="fin-metric-grid">{score_html}</div>', unsafe_allow_html=True)

        st.markdown("<hr style='border-color:var(--border);margin:14px 0'>", unsafe_allow_html=True)

        # ── Details: findings + risks | tone + valuation + catalysts ──
        left, right = st.columns([3, 2])

        with left:
            findings = report.get("key_findings", []) or []
            if findings:
                st.markdown("**Key Findings**")
                for f in findings[:6]:
                    st.markdown(f"- {f}")

            risks = report.get("top_risks", []) or []
            if risks:
                st.markdown("**Top Risks**")
                for risk in risks[:5]:
                    sev     = str(risk.get("severity", "")).lower()
                    sev_cls = "red" if sev in ("high", "critical") else "amber" if sev == "medium" else "muted"
                    cat     = _esc(str(risk.get("category", "")))
                    cnt     = risk.get("count", 0)
                    st.markdown(
                        f'<div class="fin-risk-row">'
                        f'<span class="fin-badge fin-badge-{sev_cls}">{sev.upper() or "?"}</span>'
                        f'<span style="color:var(--text-1)">{cat}</span>'
                        f'<span style="color:var(--text-3);font-size:0.75rem">({cnt} mentions)</span>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    for snip in (risk.get("snippets") or [])[:1]:
                        st.caption(f'"{str(snip)[:120]}…"')

        with right:
            # Tone trend
            tone = report.get("tone_trend", {}) or {}
            if tone and tone.get("direction"):
                direction = tone.get("direction", "flat")
                delta     = float(tone.get("delta", 0.0))
                arrows    = {"improving": "↑", "worsening": "↓", "flat": "→"}
                t_cols    = {"improving": "var(--green)", "worsening": "var(--red)", "flat": "var(--text-3)"}
                col = t_cols.get(direction, "var(--text-3)")
                st.markdown(
                    f'<div class="fin-card" style="margin-bottom:10px">'
                    f'<div class="fin-metric-lbl">Tone Trend</div>'
                    f'<div style="font-size:1.3rem;font-weight:700;color:{col};margin:4px 0">'
                    f'{arrows.get(direction,"→")} {direction.title()}</div>'
                    f'<div style="font-family:\'IBM Plex Mono\',monospace;font-size:0.8rem;color:var(--text-3)">delta: {delta:+.2f}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Valuation summary
            val    = report.get("valuation_summary", {}) or {}
            gap    = val.get("valuation_gap_pct")
            growth = val.get("revenue_growth_yoy")
            if gap is not None or growth is not None:
                gap_s = f"{float(gap):+.1%}" if gap is not None else "—"
                grw_s = f"{float(growth):+.1%}" if growth is not None else "—"
                st.markdown(
                    f'<div class="fin-card" style="margin-bottom:10px">'
                    f'<div class="fin-metric-lbl" style="margin-bottom:8px">Valuation</div>'
                    f'<div style="display:flex;gap:20px">'
                    f'<div><div class="fin-metric-lbl">Val. Gap</div>'
                    f'<div class="fin-bar-val" style="font-family:\'IBM Plex Mono\',monospace;font-size:0.9rem;color:var(--text-1)">{gap_s}</div></div>'
                    f'<div><div class="fin-metric-lbl">Rev. Growth YoY</div>'
                    f'<div class="fin-bar-val" style="font-family:\'IBM Plex Mono\',monospace;font-size:0.9rem;color:var(--text-1)">{grw_s}</div></div>'
                    f"</div></div>",
                    unsafe_allow_html=True,
                )

            # News catalysts
            news = report.get("news_summary", []) or []
            if news:
                st.markdown("**Recent Catalysts**")
                for n in news[:6]:
                    d   = n.get("direction", "neutral")
                    dot = {"positive": "green", "negative": "red"}.get(d, "muted")
                    ttl = _esc(str(n.get("title", "")))
                    st.markdown(
                        f'<div class="fin-catalyst-row">'
                        f'<span class="fin-dot fin-dot-{dot}" style="margin-top:5px;flex-shrink:0"></span>'
                        f'<span style="color:var(--text-2)">{ttl}</span>'
                        f"</div>",
                        unsafe_allow_html=True,
                    )

        # Full analyst report
        mkd = raw.get("hackathon_signal_markdown", "")
        if mkd:
            with st.expander("Full Analyst Report (Markdown)"):
                st.markdown(mkd)

        st.markdown("</div>", unsafe_allow_html=True)


def render_research_summary_kpis(r: Dict) -> None:
    raw = r.get("raw", {}) or {}
    ver = raw.get("verification", {}) or {}
    dec = raw.get("quant_decision", {}) or {}
    score = float(dec.get("score", 0.0)) if isinstance(dec, dict) else 0.0
    conf = float(ver.get("confidence", 0.0) or 0.0)
    action = str(r.get("action", "abstain")).upper()
    reason = ", ".join((ver.get("reason_codes") or [])[:2]) or "evidence-grounded"
    html = (
        '<div class="fin-kpi-grid">'
        f'<div class="fin-kpi-tile"><div class="fin-kpi-lbl">Evidence Score</div><div class="fin-kpi-val">{score:+.3f}</div><div class="fin-kpi-sub">Composite evidence-driven signal.</div></div>'
        f'<div class="fin-kpi-tile"><div class="fin-kpi-lbl">Confidence</div><div class="fin-kpi-val">{conf:.0%}</div><div class="fin-kpi-sub">Verification confidence from gate.</div></div>'
        f'<div class="fin-kpi-tile"><div class="fin-kpi-lbl">Action</div><div class="fin-kpi-val">{_esc(action)}</div><div class="fin-kpi-sub">Current research outcome.</div></div>'
        f'<div class="fin-kpi-tile"><div class="fin-kpi-lbl">Rationale</div><div class="fin-kpi-val" style="font-size:0.95rem">{_esc(reason)}</div><div class="fin-kpi-sub">Primary justification tags.</div></div>'
        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


def _build_tool_panels(raw: Dict) -> List[Dict[str, Any]]:
    dec = raw.get("quant_decision", {}) or {}
    ver = raw.get("verification", {}) or {}
    routing = raw.get("routing", {}) or {}
    result = raw.get("result", {}) or {}
    report = raw.get("hackathon_signal_report", {}) or {}
    weighted_map = {str(w.get("name")): w for w in (dec.get("weighted_signals") or []) if isinstance(w, dict)}

    tools: List[Dict[str, Any]] = []

    def add_tool(
        tool_id: str,
        name: str,
        signal_key: Optional[str],
        calc: str,
        metrics: str,
        formula: str,
        factors: str,
        evidence_refs: List[str],
        default_conf: float = 0.5,
    ) -> None:
        w = weighted_map.get(signal_key or "", {})
        score = float(w.get("score", 0.0) or 0.0)
        conf = float(w.get("confidence", default_conf) or default_conf)
        direction = "bullish" if score > 0.03 else "bearish" if score < -0.03 else "neutral"
        tools.append({
            "id": tool_id,
            "name": name,
            "signal_key": signal_key,
            "score": score,
            "direction": direction,
            "confidence": conf,
            "calc": calc,
            "metrics": metrics,
            "formula": formula,
            "factors": factors,
            "evidence_refs": evidence_refs,
            "weighting": f"effective_weight={float(w.get('effective_weight', 0.0) or 0.0):.3f}, regime_mult={float(w.get('regime_mult', 1.0) or 1.0):.2f}",
        })

    add_tool(
        "T1", "numeric_lookup", None,
        "Deterministic lookup from filing/XBRL evidence.",
        "metric value, unit, citation strength",
        "direct extraction",
        "audited table/XBRL precedence",
        [str((result.get("numeric") or {}).get("citation") or "")],
        default_conf=float(ver.get("confidence", 0.75) or 0.75),
    )
    add_tool(
        "T2", "compute_metric", "growth",
        "Derived metric computed from verified inputs.",
        "input metrics, fiscal alignment, unit normalization",
        str((result.get("computed") or {}).get("formula") or "computed ratio"),
        "input evidence integrity + consistency",
        [str(x.get("citation")) for x in ((result.get("computed") or {}).get("inputs") or []) if isinstance(x, dict) and x.get("citation")],
    )
    add_tool(
        "T3", "dcf_valuation", "valuation",
        "Discounted cash flow valuation gap signal.",
        "FCF, WACC, terminal growth, sensitivity",
        "valuation_gap -> normalized to [-1,1]",
        "assumption stress + valuation spread",
        [str(x.get("citation")) for x in ((result.get("valuation") or {}).get("verified_inputs") or []) if isinstance(x, dict) and x.get("citation")],
    )
    add_tool(
        "T4", "relative_valuation", "peer_valuation",
        "Market multiple vs peer median premium/discount.",
        "target multiple, peer median, premium pct",
        "signal = -peer_premium_pct",
        "peer quality + multiple reliability",
        [str((result.get("relative_valuation") or {}).get("denominator", {}).get("citation") or "")],
    )
    add_tool(
        "T5", "risk_extraction", "risk",
        "Risk language extraction from filing evidence.",
        "severity, category counts, snippet density",
        "signal = -avg_top_risk_severity",
        "risk concentration in Item 1A / filing text",
        [str(r.get("category")) for r in (report.get("top_risks") or [])[:3]],
    )
    add_tool(
        "T6", "tone_comparison", "tone",
        "Transcript tone delta vs prior period.",
        "current tone, prior tone, delta",
        "signal = clipped(tone_delta)",
        "tone regime + transcript availability",
        [str((report.get("tone_trend") or {}).get("direction") or "")],
    )
    add_tool(
        "T7", "news_catalysts", "news",
        "Recent catalyst direction from news classification.",
        "article direction scores, source count",
        "signal = avg(news_direction_score)",
        "source diversity + recency weighting",
        [str(n.get("title")) for n in (report.get("news_summary") or [])[:3]],
    )
    add_tool(
        "T8", "evidence_gate", None,
        "Verification gate on source sufficiency and consistency.",
        "slot coverage, source coverage, thresholds",
        "confidence gating policy",
        "required-source pass/fail checks",
        [str(x) for x in (ver.get("reason_codes") or [])[:4]],
        default_conf=float(ver.get("confidence", 0.7) or 0.7),
    )
    add_tool(
        "T9", "routing", None,
        "Cost/risk aware model routing decision.",
        "retrieval risk, coverage, margin",
        "risk -> model selection",
        "inference cost and uncertainty tradeoff",
        [str(routing.get("model") or ""), str(routing.get("action") or "")],
        default_conf=0.6,
    )
    return tools


def render_decision_section(r: Dict) -> None:
    raw = r.get("raw", {}) or {}
    dec = raw.get("quant_decision") or {}
    if not isinstance(dec, dict) or not dec:
        st.info("No decision trace available yet. Run a query to generate decision diagnostics.")
        return

    action = str(dec.get("action", "WATCH"))
    score = dec.get("score", 0.0)
    conf = dec.get("aggregate_confidence", 0.0)
    reason = str(dec.get("reason_code", ""))
    regime = dec.get("regime", {}) or {}
    weighted = dec.get("weighted_signals", []) or []
    contradictions = dec.get("contradictions", []) or []
    trace = dec.get("decision_tree_trace", []) or []

    cls = {"ACT": "green", "WATCH": "amber", "NO_ACT": "red"}.get(action, "muted")
    st.markdown(
        '<div class="fin-kpi-grid">'
        f'<div class="fin-kpi-tile"><div class="fin-kpi-lbl">Decision Signal</div><div class="fin-kpi-val" style="color:var(--{cls if cls in ("green","red","amber") else "text-1"})">{_esc(action)}</div><div class="fin-kpi-sub">Final action after multi-signal aggregation and contradiction checks.</div></div>'
        f'<div class="fin-kpi-tile"><div class="fin-kpi-lbl">Signal Score</div><div class="fin-kpi-val">{float(score):+.4f}</div><div class="fin-kpi-sub">Confidence-weighted, regime-adjusted composite score.</div></div>'
        f'<div class="fin-kpi-tile"><div class="fin-kpi-lbl">Confidence</div><div class="fin-kpi-val">{float(conf):.1%}</div><div class="fin-kpi-sub">Weighted confidence minus contradiction penalty.</div></div>'
        f'<div class="fin-kpi-tile"><div class="fin-kpi-lbl">Reason</div><div class="fin-kpi-val" style="font-size:0.95rem">{_esc(reason)}</div><div class="fin-kpi-sub">Primary decision rule fired.</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if regime:
        st.markdown("**Regime Context**")
        st.json(regime)

    tools = _build_tool_panels(raw)
    st.markdown("**Tool Analysis**")
    cards = ""
    for t in tools:
        dir_cls = "green" if t["direction"] == "bullish" else "red" if t["direction"] == "bearish" else "muted"
        cards += (
            f'<div class="fin-tool-card">'
            f'<div class="fin-tool-name">{_esc(t["id"])} · {_esc(t["name"])}</div>'
            f'<div class="fin-tool-row">'
            f'<span class="fin-badge fin-badge-{dir_cls}">{_esc(t["direction"])}</span>'
            f'<span class="fin-badge fin-badge-cyan">{float(t["score"]):+.3f}</span>'
            f'<span class="fin-badge fin-badge-muted">{float(t["confidence"]):.0%}</span>'
            f'</div>'
            f'<div class="fin-tool-mini">{_esc(t["calc"])}</div>'
            f'</div>'
        )
    st.markdown(f'<div class="fin-tool-grid">{cards}</div>', unsafe_allow_html=True)
    for t in tools:
        with st.expander(f'{t["id"]} — {t["name"]}', expanded=False):
            st.markdown(f'**How Score Was Calculated**: {t["calc"]}')
            st.markdown(f'**Metrics Used**: {t["metrics"]}')
            st.markdown(f'**Formula / Weighting**: `{t["formula"]}` · {t["weighting"]}')
            st.markdown(f'**Contributing Factors**: {t["factors"]}')
            st.markdown(f'**Confidence Calculation**: base reliability with evidence agreement and model uncertainty controls (`{float(t["confidence"]):.2f}`)')
            refs = [x for x in t["evidence_refs"] if str(x).strip()]
            if refs:
                st.markdown("**Evidence Sources**")
                for i, ref in enumerate(refs[:5], 1):
                    st.markdown(f"- `{i}` {_esc(str(ref)[:140])}")

    st.markdown("**Contradiction Check**")
    if contradictions:
        st.warning(f"{len(contradictions)} contradiction(s) found across high-confidence signals.")
        st.dataframe(contradictions, use_container_width=True)
    else:
        st.success("No material contradictions detected across high-confidence signals.")

    with st.expander("Decision Tree Trace", expanded=True):
        for line in trace:
            st.code(str(line), language=None)

    ver = raw.get("verification", {}) or {}
    best_ev = ver.get("best_evidence", []) or []
    if best_ev:
        st.markdown("**Evidence Sources**")
        for i, ev in enumerate(best_ev[:8], 1):
            label = f"{i}. [{ev.get('kind')}] {ev.get('evidence_id')} · {ev.get('item') or 'n/a'}"
            snippet = str(ev.get("preview") or "")
            with st.expander(label, expanded=False):
                st.caption(snippet)


# ─────────────────────────────────────────────────────────────────────────────
# EVIDENCE TABS
# ─────────────────────────────────────────────────────────────────────────────

def render_evidence_tabs(r: Dict, base_dir: Path) -> None:
    evidence    = r.get("evidence", {}) or {}
    xbrl_ev     = evidence.get("xbrl", {}) or {}
    best_ev     = _get_best_evidence(r)

    # Split best_evidence by kind
    chunks = [e for e in best_ev if e.get("kind") not in ("table",)]
    tables = [e for e in best_ev if e.get("kind") == "table"]

    # Also pull table records from evidence dict if present
    tables_ev = evidence.get("tables", {}) or {}
    tbl_recs  = tables_ev.get("records", []) or []
    if not tables and tbl_recs:
        tables = tbl_recs

    xbrl_hits = xbrl_ev.get("hits", []) or xbrl_ev.get("facts", []) or []
    packed_ctx = r.get("packed_context", "") or ""

    tab_src, tab_tbl, tab_ctx, tab_xbrl, tab_audit = st.tabs([
        "Source Summary", "Table View", "Expandable Context", "XBRL Facts", "Audit Trace",
    ])

    # ── TAB 1: Source Summary ─────────────────────────────
    with tab_src:
        if not chunks:
            st.markdown(_EMPTY_HTML.format(icon="📄", title="No sources retrieved", sub="Run a query to see evidence sources"), unsafe_allow_html=True)
        else:
            for i, blk in enumerate(chunks[:16], 1):
                title     = (blk.get("metadata", {}) or {}).get("doc_title") or blk.get("source") or f"Source {i}"
                ticker_b  = blk.get("ticker", "") or ""
                fy_b      = blk.get("fiscal_year", "") or ""
                item_b    = blk.get("item", "") or ""
                evid_id   = blk.get("evid", "") or blk.get("id", "")
                kind_b    = blk.get("kind", "chunk")
                src_type  = blk.get("source_type", "")
                text_full = blk.get("text", "") or ""
                excerpt   = _esc(text_full[:240])
                url       = (blk.get("metadata", {}) or {}).get("source_url") or (blk.get("metadata", {}) or {}).get("url") or ""

                ticker_badge = f'<span class="fin-badge fin-badge-purple">{_esc(str(ticker_b))}</span>' if ticker_b else ""
                fy_badge     = f'<span class="fin-badge fin-badge-muted">FY{fy_b}</span>' if fy_b else ""
                item_badge   = f'<span class="fin-badge fin-badge-muted">{_esc(str(item_b))}</span>' if item_b else ""
                kind_badge   = f'<span class="fin-badge fin-badge-cyan">{kind_b}</span>' if kind_b else ""
                open_btn     = (f'<a class="fin-open-btn" href="{url}" target="_blank">↗ Open Source</a>' if url else "")
                cited_cls    = " fin-src-cited" if i <= 3 else ""  # Animated highlight for top 3 sources

                st.markdown(
                    f'<div class="fin-src-card{cited_cls}">'
                    f'<div class="fin-src-header">'
                    f'  <div class="fin-src-idx">{i}</div>'
                    f'  <div class="fin-src-title">{_esc(str(title))}</div>'
                    f'</div>'
                    f'<div style="display:flex;gap:5px;flex-wrap:wrap;margin-bottom:6px">{ticker_badge}{fy_badge}{item_badge}{kind_badge}</div>'
                    f'<div class="fin-src-excerpt">{excerpt}{"…" if len(text_full) > 240 else ""}</div>'
                    f'<div class="fin-src-footer">'
                    f'  <span class="fin-src-url">{_esc(url[:70])}{"…" if len(url) > 70 else ""}</span>'
                    f'  {open_btn}'
                    f'</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
                # Preview drawer button
                if text_full:
                    if st.button("Preview document", key=f"prev_{i}_{evid_id}", help="Open document preview"):
                        st.session_state.preview_doc = {
                            "title": str(title), "text": text_full,
                            "url": url, "ticker": str(ticker_b),
                            "fy": str(fy_b), "item": str(item_b),
                        }
                        st.session_state.show_drawer = True
                        st.rerun()

    # ── TAB 2: Table View ─────────────────────────────────
    with tab_tbl:
        if not tables:
            st.markdown(_EMPTY_HTML.format(icon="📊", title="No tables retrieved", sub="Numeric and financial queries surface structured tables"), unsafe_allow_html=True)
        else:
            for i, blk in enumerate(tables[:12], 1):
                title    = (blk.get("metadata", {}) or {}).get("title") or blk.get("title") or f"Table {i}"
                ticker_b = blk.get("ticker", "") or ""
                fy_b     = blk.get("fiscal_year", "") or ""
                url      = (blk.get("metadata", {}) or {}).get("source_url") or ""
                text     = blk.get("text", "") or blk.get("surrogate_text", "") or ""

                with st.expander(f"Table {i} — {title}", expanded=(i <= 2)):
                    col_meta, col_link = st.columns([5, 1])
                    with col_meta:
                        badges = ""
                        if ticker_b: badges += f'<span class="fin-badge fin-badge-purple" style="margin-right:5px">{_esc(str(ticker_b))}</span>'
                        if fy_b:     badges += f'<span class="fin-badge fin-badge-muted" style="margin-right:5px">FY{fy_b}</span>'
                        if badges:   st.markdown(badges, unsafe_allow_html=True)
                    with col_link:
                        if url:
                            st.markdown(f'<a class="fin-open-btn" href="{url}" target="_blank">↗ Source</a>', unsafe_allow_html=True)
                    st.code(text[:8000], language=None)

    # ── TAB 3: Expandable Context ─────────────────────────
    with tab_ctx:
        all_blocks = chunks or best_ev
        if not all_blocks:
            st.markdown(_EMPTY_HTML.format(icon="📝", title="No context chunks", sub=""), unsafe_allow_html=True)
        else:
            for i, blk in enumerate(all_blocks[:20], 1):
                title    = (blk.get("metadata", {}) or {}).get("doc_title") or f"Chunk {i}"
                item_b   = blk.get("item", "") or ""
                evid_id  = blk.get("evid", "") or blk.get("id", "")
                text     = blk.get("text", "") or ""
                url      = (blk.get("metadata", {}) or {}).get("source_url") or ""
                label    = f"{i}. {title}{(' · ' + str(item_b)) if item_b else ''}"

                with st.expander(label, expanded=False):
                    st.code(text[:10000], language=None)
                    if len(text) > 10000:
                        st.caption(f"…truncated to 10 000 of {len(text):,} chars")
                    foot_c1, foot_c2 = st.columns(2)
                    with foot_c1:
                        if url:
                            st.markdown(f'<a class="fin-open-btn" href="{url}" target="_blank">↗ Open Source</a>', unsafe_allow_html=True)
                    with foot_c2:
                        if evid_id:
                            st.caption(f"Evidence ID: {evid_id}")

    # ── TAB 4: XBRL Facts ────────────────────────────────
    with tab_xbrl:
        if not xbrl_hits:
            st.markdown(_EMPTY_HTML.format(icon="🔢", title="No XBRL facts", sub="Numeric lookups pull structured XBRL company facts"), unsafe_allow_html=True)
        else:
            for fact in xbrl_hits[:40]:
                if not isinstance(fact, dict):
                    continue
                concept  = fact.get("concept") or fact.get("label") or "Unknown"
                value    = fact.get("value")
                unit     = fact.get("unit", "") or ""
                fy_x     = fact.get("fy") or fact.get("fiscal_year") or ""
                ticker_x = fact.get("ticker", "") or ""
                end_x    = fact.get("end", "") or ""
                val_str  = f"{value:,.2f}" if isinstance(value, (int, float)) else str(value) if value is not None else "—"

                tb = f'<span class="fin-badge fin-badge-purple" style="margin-right:4px">{_esc(str(ticker_x))}</span>' if ticker_x else ""
                fyb = f'<span class="fin-badge fin-badge-muted" style="margin-right:4px">FY{fy_x}</span>' if fy_x else ""
                eb  = f'<span class="fin-badge fin-badge-muted">{_esc(end_x)}</span>' if end_x else ""

                st.markdown(
                    f'<div class="fin-xbrl-card">'
                    f'<div class="fin-xbrl-concept">{_esc(str(concept))}</div>'
                    f'<div><span class="fin-xbrl-value">{_esc(val_str)}</span>'
                    f'<span class="fin-xbrl-unit">{_esc(unit)}</span></div>'
                    f'<div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:4px">{tb}{fyb}{eb}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

    # ── TAB 5: Audit Trace ───────────────────────────────
    with tab_audit:
        raw_r   = r.get("raw", {}) or {}
        routing = _safe_to_dict(r.get("routing", {})) or {}
        gate    = _safe_to_dict(r.get("gate", {})) or _safe_to_dict(raw_r.get("verification", {})) or {}
        timing  = raw_r.get("timing_ms", {}) or {}

        if routing:
            st.markdown("**Routing Decision**")
            rows = "".join(
                f'<div class="fin-trace-row"><span class="fin-trace-key">{_esc(str(k))}</span>'
                f'<span class="fin-trace-val">{_esc(str(v))}</span></div>'
                for k, v in routing.items()
            )
            st.markdown(f'<div class="fin-trace-panel">{rows}</div>', unsafe_allow_html=True)

        if timing:
            st.markdown("**Timing (ms)**")
            rows = "".join(
                f'<div class="fin-trace-row"><span class="fin-trace-key">{_esc(str(k))}</span>'
                f'<span class="fin-trace-val" style="font-family:\'IBM Plex Mono\',monospace">{_esc(str(v))}</span></div>'
                for k, v in timing.items()
            )
            st.markdown(f'<div class="fin-trace-panel">{rows}</div>', unsafe_allow_html=True)

        for err in (raw_r.get("validation_errors") or []):
            st.error(err)

        if packed_ctx.strip():
            with st.expander("Packed Context (sent to LLM)"):
                st.code(packed_ctx[:12000], language=None)
                st.download_button(
                    "Download packed_context.txt", packed_ctx.encode(),
                    f"packed_context_{r['run_id']}.txt", "text/plain",
                )

        audit_path = base_dir / "logs" / "audit.jsonl"
        audit_tail = _read_audit_log(audit_path, tail=200)
        if audit_tail.strip():
            with st.expander("Audit Log (last 200 entries)"):
                st.code(audit_tail)
                st.download_button(
                    "Download audit.jsonl",
                    audit_path.read_bytes() if audit_path.exists() else b"",
                    "audit.jsonl", "application/json",
                )


# ─────────────────────────────────────────────────────────────────────────────
# WHY THIS ANSWER (reasoning panel)
# ─────────────────────────────────────────────────────────────────────────────

def render_reasoning_panel(r: Dict) -> None:
    raw     = r.get("raw", {}) or {}
    ver     = raw.get("verification", {}) or {}
    routing = r.get("routing") or {}

    status   = ver.get("status", "—")
    conf     = ver.get("confidence")
    mode_str = raw.get("mode", "—")
    model    = routing.get("model", "—")
    r_action = routing.get("action", "—")
    codes    = list(ver.get("reason_codes", []) or []) + list(routing.get("reason_codes", []) or [])
    warnings = ver.get("warnings", []) or []
    coverage = ver.get("source_coverage", {}) or {}

    with st.expander("Why This Answer", expanded=False):
        rows_html = ""

        s_cls = "green" if status == "answer" else "amber"
        rows_html += (
            f'<div class="fin-rsn-row"><span class="fin-rsn-key">verification_status</span>'
            f'<span class="fin-badge fin-badge-{s_cls}">{_esc(status)}</span></div>'
        )

        if conf is not None:
            rows_html += (
                f'<div class="fin-rsn-row"><span class="fin-rsn-key">gate_confidence</span>'
                f'<span class="fin-rsn-val" style="font-family:\'IBM Plex Mono\',monospace">{float(conf):.2%}</span></div>'
            )

        rows_html += (
            f'<div class="fin-rsn-row"><span class="fin-rsn-key">query_mode</span>'
            f'<span class="fin-rsn-val" style="font-family:\'IBM Plex Mono\',monospace">{_esc(mode_str)}</span></div>'
        )

        rows_html += (
            f'<div class="fin-rsn-row"><span class="fin-rsn-key">routing</span>'
            f'<span class="fin-rsn-val">{_esc(r_action)} via <strong style="color:var(--text-1)">{_esc(model)}</strong></span></div>'
        )

        if codes:
            tags = "".join(f'<span class="fin-reason-tag">{_esc(c)}</span>' for c in codes)
            rows_html += (
                f'<div class="fin-rsn-row"><span class="fin-rsn-key">reason_codes</span>'
                f'<span class="fin-rsn-val" style="flex-wrap:wrap;display:flex;gap:2px">{tags}</span></div>'
            )

        if warnings:
            tags = "".join(f'<span class="fin-warn-tag">{_esc(w)}</span>' for w in warnings)
            rows_html += (
                f'<div class="fin-rsn-row"><span class="fin-rsn-key">warnings</span>'
                f'<span class="fin-rsn-val" style="flex-wrap:wrap;display:flex;gap:2px">{tags}</span></div>'
            )

        if coverage:
            cov_html = " ".join(
                f'<span class="fin-badge fin-badge-{"green" if v else "muted"}">'
                f'{_esc(src)} {"✓" if v else "✗"}</span>'
                for src, v in coverage.items()
            )
            rows_html += (
                f'<div class="fin-rsn-row"><span class="fin-rsn-key">source_coverage</span>'
                f'<span class="fin-rsn-val">{cov_html}</span></div>'
            )

        # Best evidence IDs
        best_ev  = _get_best_evidence(r)
        evid_ids = [e.get("evid") or e.get("id") for e in best_ev if e.get("evid") or e.get("id")]
        if evid_ids:
            id_tags = " ".join(f'<span class="fin-reason-tag">{_esc(str(x))}</span>' for x in evid_ids[:12])
            rows_html += (
                f'<div class="fin-rsn-row"><span class="fin-rsn-key">evidence_ids</span>'
                f'<span class="fin-rsn-val" style="flex-wrap:wrap;display:flex;gap:2px">{id_tags}</span></div>'
            )

        st.markdown(f'<div class="fin-reasoning">{rows_html}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DOCUMENT PREVIEW DRAWER
# ─────────────────────────────────────────────────────────────────────────────

def render_document_drawer() -> None:
    doc = st.session_state.get("preview_doc") or {}
    if not doc:
        return

    title  = doc.get("title", "Document Preview")
    text   = doc.get("text", "")
    url    = doc.get("url", "")
    ticker = doc.get("ticker", "")
    fy     = doc.get("fy", "")
    item   = doc.get("item", "")

    tb  = f'<span class="fin-badge fin-badge-purple" style="margin-left:8px">{_esc(ticker)}</span>' if ticker else ""
    fyb = f'<span class="fin-badge fin-badge-muted" style="margin-left:4px">FY{_esc(fy)}</span>' if fy else ""
    itb = f'<span class="fin-badge fin-badge-muted" style="margin-left:4px">{_esc(item)}</span>' if item else ""

    st.markdown(
        f'<div class="fin-drawer">'
        f'<div class="fin-drawer-hdr">'
        f'<div class="fin-drawer-title">Document Preview</div>'
        f'<span style="font-size:0.82rem;color:var(--text-2)">{_esc(str(title))[:60]}</span>'
        f"{tb}{fyb}{itb}"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    col_text, col_btn = st.columns([6, 1])
    with col_text:
        st.code(text[:14000], language=None)
        if len(text) > 14000:
            st.caption(f"…truncated ({len(text):,} chars total)")
        if url:
            st.markdown(f'<a class="fin-open-btn" href="{url}" target="_blank" style="font-size:0.8rem">↗ Open Full Source Document</a>', unsafe_allow_html=True)
    with col_btn:
        if st.button("✕ Close", key="close_drawer"):
            st.session_state.show_drawer = False
            st.session_state.preview_doc = None
            st.rerun()

    st.markdown("<hr style='border-color:var(--border);margin:4px 0 1.2rem'>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="FinSignal AI",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(DARK_CSS, unsafe_allow_html=True)

    base_dir   = Path(__file__).resolve().parent
    hc         = _health_check(base_dir)
    llm_label, llm_ok = _llm_status()

    # ── Session state init ──────────────────────────────
    if "history"     not in st.session_state: st.session_state.history     = []
    if "show_drawer" not in st.session_state: st.session_state.show_drawer = False
    if "preview_doc" not in st.session_state: st.session_state.preview_doc = None

    # ── Sidebar ─────────────────────────────────────────
    with st.sidebar:
        st.markdown("### System Status")
        if hc["ok"]:
            st.success("Index files: READY")
        else:
            st.error("Index files: MISSING")
            with st.expander("Missing files"):
                for m in hc["missing"]:
                    st.caption(m)

        if llm_ok:
            st.success(f"{llm_label}: ACTIVE")
        else:
            st.error("LLM: NOT CONFIGURED")
            st.caption("Set GEMINI_API_KEY or OLLAMA_HOST in .env")

        st.markdown("---")
        st.markdown(f"**Session runs:** {len(st.session_state.history)}")
        if st.button("Clear history"):
            st.session_state.history = []
            st.rerun()

        if st.session_state.history:
            st.markdown("**Recent queries**")
            for i, rec in enumerate(st.session_state.history[:5]):
                st.caption(f"{i+1}. {rec.get('question','')[:50]}")

    # ── Sticky Header ────────────────────────────────────
    crumb = (st.session_state.history[0].get("question", "") if st.session_state.history else "")
    render_header(hc, llm_ok, llm_label, crumb)

    # ── Document Preview Drawer ──────────────────────────
    if st.session_state.get("show_drawer"):
        render_document_drawer()

    # ── Query Zone ───────────────────────────────────────
    dynamic_fy = _available_fiscal_years(base_dir)

    try:
        orch = get_orchestrator()
    except Exception as e:
        st.error(f"Orchestrator init failed: {type(e).__name__}: {e}")
        st.stop()

    def _latest_for_workspace(workspace: str) -> Optional[Dict[str, Any]]:
        for rec in st.session_state.history:
            if rec.get("workspace") == workspace:
                return rec
        return None

    def _run_analysis(*, workspace: str, question: str, mode: str, ticker: Optional[str], fiscal_year: Optional[int], strictness: int) -> None:
        if not question.strip():
            st.warning("Please enter a query before running analysis.")
            return

        forced_mode = None if mode == "auto" else mode
        t0 = time.time()
        trace_placeholder = st.empty()
        with trace_placeholder.container():
            with st.status("Running financial analysis pipeline...", expanded=True) as status:
                st.write("**Planning** - classifying query, identifying entities...")
                try:
                    result = orch.answer(
                        question.strip(),
                        market_inputs=None,
                        auto_fetch_market=True,
                        forced_mode=forced_mode,
                        ui_intent=mode,
                        ui_ticker=ticker,
                        ui_fiscal_year=fiscal_year,
                        evidence_strictness=strictness,
                        decision_time=datetime.now(timezone.utc).isoformat(),
                    )
                except Exception as e:
                    status.update(label="Analysis failed", state="error")
                    st.error(f"Request failed: {type(e).__name__}: {e}")
                    return

                latency_s = round(time.time() - t0, 2)
                r_obj = result.get("routing", {}) or {}
                v_obj = result.get("verification", {}) or {}
                tm_obj = result.get("timing_ms", {}) or {}
                det_mode = result.get("mode", "?")
                det_mdl = r_obj.get("model", "?")
                det_gate = v_obj.get("status", "?")
                st.write(f"**Retrieval & Verification** - mode: `{det_mode}` ? model: `{det_mdl}` ? gate: `{det_gate}`")
                if tm_obj:
                    parts = " ? ".join(f"{k}: {v}ms" for k, v in tm_obj.items())
                    st.write(f"**Timing** - {parts}")
                status.update(
                    label=f"Analysis complete - {latency_s}s  ?  mode: {det_mode}  ?  {det_gate}",
                    state="complete",
                    expanded=False,
                )
        trace_placeholder.empty()

        routing = result.get("routing", {}) or {}
        gate = result.get("verification", result.get("gate", {})) or {}
        action = result.get("action", "abstain")
        final_answer = _extract_final_answer(result)
        evidence = _flatten_evidence(result)
        packed_ctx = result.get("packed_context", "") or ""
        if not isinstance(packed_ctx, str):
            packed_ctx = str(packed_ctx)

        st.session_state.history.insert(0, {
            "ts": int(time.time()),
            "workspace": workspace,
            "question": question.strip(),
            "ticker": ticker,
            "mode": mode,
            "fiscal_year": fiscal_year,
            "strictness": strictness,
            "run_id": result.get("run_id", "?"),
            "action": action,
            "latency_s": latency_s,
            "routing": routing,
            "gate": gate,
            "final_answer": final_answer,
            "evidence": evidence,
            "packed_context": packed_ctx,
            "raw": result,
        })

    tab_decision_mode, tab_research_mode = st.tabs(["Decision Mode", "Research Mode"])

    with tab_decision_mode:
        st.markdown('<div class="fin-query-zone">', unsafe_allow_html=True)
        st.markdown('<div class="fin-query-title">Decision Mode Control Panel</div>', unsafe_allow_html=True)
        d1, d2, d3, d4 = st.columns([1.4, 2.0, 5.2, 1.4])
        with d1:
            ticker_opts = ["(auto)"] + UI_TICKERS if UI_TICKERS else ["(auto)"]
            t_choice = st.selectbox("Ticker", ticker_opts, index=0, key="dec_ticker")
            dec_ticker = None if t_choice == "(auto)" else t_choice
        with d2:
            dec_strict = st.slider("Evidence Strictness", 0, 100, 70, 1, key="dec_strict")
        with d3:
            dec_query = st.text_input(
                "Decision Query",
                value="",
                placeholder="Optional thesis prompt (leave blank for default decision analysis)",
                label_visibility="collapsed",
                key="dec_query",
            )
        with d4:
            run_decision = st.button("Run Analysis", use_container_width=True, key="run_decision")
        st.markdown("</div>", unsafe_allow_html=True)

        if run_decision:
            q = dec_query.strip() if isinstance(dec_query, str) and dec_query.strip() else f"Provide an investment decision overview for {dec_ticker or 'the selected company'}."
            _run_analysis(
                workspace="decision",
                question=q,
                mode="auto",
                ticker=dec_ticker,
                fiscal_year=None,
                strictness=dec_strict,
            )

        dec_run = _latest_for_workspace("decision")
        if dec_run:
            st.markdown("<hr style='border-color:var(--border);margin:0.5rem 0 1rem'>", unsafe_allow_html=True)
            render_decision_section(dec_run)
        else:
            st.markdown(
                _EMPTY_HTML.format(icon="?", title="No decision run yet", sub="Pick a ticker and run Decision Mode to generate a traceable signal."),
                unsafe_allow_html=True,
            )

    with tab_research_mode:
        st.markdown('<div class="fin-query-zone">', unsafe_allow_html=True)
        st.markdown('<div class="fin-query-title">Research Mode Control Panel</div>', unsafe_allow_html=True)
        r1, r2, r3 = st.columns([1.4, 2.0, 2.0])
        with r1:
            ticker_opts = ["(auto)"] + UI_TICKERS if UI_TICKERS else ["(auto)"]
            rt_choice = st.selectbox("Ticker", ticker_opts, index=0, key="res_ticker")
            res_ticker = None if rt_choice == "(auto)" else rt_choice
        with r2:
            res_mode = st.selectbox("Mode (Tools)", ALL_MODES, index=0, key="res_mode")
        with r3:
            res_strict = st.slider("Evidence Strictness", 0, 100, 70, 1, key="res_strict")

        fy_opts = ["(auto)"] + [str(y) for y in dynamic_fy] if dynamic_fy else ["(auto)"]
        fyc = st.selectbox("Fiscal Year", fy_opts, index=0, key="res_fy")
        res_fy = None if fyc == "(auto)" else int(fyc)

        rq1, rq2 = st.columns([8.8, 1.2])
        with rq1:
            res_query = st.text_input(
                "Query",
                value="",
                placeholder="Ask a financial research question with evidence traceability...",
                label_visibility="collapsed",
                key="res_query",
            )
        with rq2:
            run_research = st.button("Run Query", use_container_width=True, key="run_research")
        st.markdown("</div>", unsafe_allow_html=True)

        if run_research:
            _run_analysis(
                workspace="research",
                question=res_query,
                mode=res_mode,
                ticker=res_ticker,
                fiscal_year=res_fy,
                strictness=res_strict,
            )

        res_run = _latest_for_workspace("research")
        if res_run:
            raw = res_run.get("raw", {}) or {}
            st.markdown("<hr style='border-color:var(--border);margin:0.5rem 0 1rem'>", unsafe_allow_html=True)
            render_research_summary_kpis(res_run)
            render_answer_section(res_run)
            if raw.get("hackathon_signal_report"):
                render_signal_section(raw)
            st.markdown('<div class="fin-section-label" style="margin-top:1.5rem;margin-bottom:4px">Evidence Panel</div>', unsafe_allow_html=True)
            render_evidence_tabs(res_run, base_dir)
            st.markdown('<div class="fin-section-label" style="margin-top:1.2rem;margin-bottom:6px">Why This Answer</div>', unsafe_allow_html=True)
            render_reasoning_panel(res_run)
        else:
            st.markdown(
                _EMPTY_HTML.format(icon="?", title="No research run yet", sub="Select ticker/mode/strictness and run a query to get evidence-backed analysis."),
                unsafe_allow_html=True,
            )


if __name__ == "__main__":
    main()
