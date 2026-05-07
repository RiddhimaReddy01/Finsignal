# FinSignal AI System Architecture

```mermaid
flowchart LR
  U[User]
  FE[React UI<br/>frontend/src/FinSightTerminal.jsx]
  API[FastAPI Server<br/>server.py]
  ORCH[FinancialOrchestrator<br/>orchestrator.py]
  VER[Verification & Guardrails<br/>verification.py]
  ROUTE[Model Routing<br/>routing.py + local_llm.py]
  REPORT[Report Generation<br/>report_generation.py]

  subgraph Retrieval_and_Data
    RET[Retrieval Tool<br/>retrieval_tool.py]
    IDX[(Index Files<br/>index/*.parquet + FAISS/BM25)]
    XBRL[(XBRL Companyfacts<br/>data/xbrl_companyfacts)]
    MKT[Market Data Provider<br/>market_api.py]
    NEWS[News Clients<br/>news_client_adapter.py + news_ingestion.py]
    TRN[Transcript APIs<br/>transcript_api.py + transcript_ingestion.py]
  end

  subgraph Analytics_Engines
    VAL[DCF Valuation<br/>valuation_engine.py]
    RVAL[Relative Valuation<br/>relative_valuation_engine.py]
    SCN[Scenario Analysis<br/>scenario_analysis.py]
    PEER[Peer Analysis<br/>peer_analysis.py]
    NLP[NLP Signals<br/>nlp_signals.py]
    SCORE[Signal Scoring<br/>signal_scoring.py]
  end

  subgraph Caching
    APIC[(API Disk TTL Cache<br/>data/cache/api/*)]
    MKT_CACHE[(Market Cache<br/>TTL + disk)]
    NEWS_CACHE[(News Context Cache)]
    TRN_CACHE[(Transcript Context Cache)]
  end

  U --> FE
  FE -->|/api/analyze<br/>/api/decision| API

  API -->|cache-first read| APIC
  APIC -->|hit -> return| API

  API --> ORCH
  ORCH --> RET
  RET --> IDX
  RET --> XBRL

  ORCH --> MKT
  MKT --> MKT_CACHE
  ORCH --> NEWS
  NEWS --> NEWS_CACHE
  ORCH --> TRN
  TRN --> TRN_CACHE

  ORCH --> VER
  ORCH --> ROUTE

  ORCH --> VAL
  ORCH --> RVAL
  ORCH --> SCN
  ORCH --> PEER
  ORCH --> NLP
  VAL --> SCORE
  RVAL --> SCORE
  SCN --> SCORE
  PEER --> SCORE
  NLP --> SCORE

  ORCH --> API
  API --> REPORT
  REPORT -->|HTML/PDF| FE
  API -->|cache write| APIC
```

## Runtime Flow (High Level)
1. User submits query from React UI.
2. FastAPI checks in-memory + disk cache first.
3. Cache miss calls orchestrator for retrieval, verification, analytics, and routing.
4. Results are post-processed into structured answer + evidence + optional report (HTML/PDF).
5. Response is cached and returned to UI.

