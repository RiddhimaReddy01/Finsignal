

import os
import re
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from bs4 import BeautifulSoup


# =========================
# CONFIG
# =========================

# IMPORTANT on Windows: use raw string or forward slashes
BASE = Path(os.environ.get("FIN_TOOL_BASE", Path(__file__).resolve().parent))

TICKERS = ["AAPL", "META", "NVDA", "GOOGL", "TSLA"]
TARGET_FYS = [2024, 2025]

RATE_SLEEP_S = 0.25
MAX_TRIES = 5

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"


# =========================
# DIRS
# =========================

RAW_DIR = BASE / "data" / "raw_html"
META_DIR = BASE / "data" / "meta"
SECT_DIR = BASE / "data" / "sections"
TABLE_DIR = BASE / "data" / "tables"
FACTS_DIR = BASE / "data" / "xbrl_companyfacts"

for d in [RAW_DIR, META_DIR, SECT_DIR, TABLE_DIR, FACTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =========================
# HEADERS + GET HELPERS
# =========================

SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "FinancialAnalysisTool/1.0 (research@localhost)",
)

HEADERS_DATA = {
    "User-Agent": SEC_USER_AGENT,
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

HEADERS_WWW = dict(HEADERS_DATA)
HEADERS_WWW["Host"] = "www.sec.gov"


def http_get(
    url: str,
    headers: Dict[str, str],
    sleep_s: float = RATE_SLEEP_S,
    max_tries: int = MAX_TRIES,
    timeout_s: int = 60,
    session: Optional[requests.Session] = None,
) -> requests.Response:
    sess = session or requests
    for attempt in range(max_tries):
        r = sess.get(url, headers=headers, timeout=timeout_s)
        if r.status_code == 200:
            time.sleep(sleep_s)
            return r
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep((attempt + 1) * 1.5)
            continue
        raise RuntimeError(f"GET failed {r.status_code}: {url}\n{r.text[:300]}")
    raise RuntimeError(f"GET failed after retries: {url}")


# =========================
# SEC LOOKUPS
# =========================

def load_ticker_to_cik(session: Optional[requests.Session] = None) -> Dict[str, str]:
    r = http_get(TICKER_MAP_URL, headers=HEADERS_WWW, session=session)
    tickers_json = r.json()

    out: Dict[str, str] = {}
    for _, rec in tickers_json.items():
        t = rec.get("ticker")
        cik_str = rec.get("cik_str")
        if not t or cik_str is None:
            continue
        out[t.upper()] = str(cik_str).zfill(10)
    return out


def get_submissions(cik10: str, session: Optional[requests.Session] = None) -> dict:
    url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    return http_get(url, headers=HEADERS_DATA, session=session).json()


def select_10k_for_fy(sub_json: dict, target_fy: int) -> Optional[Tuple[str, str, str, str, str]]:
    """
    Returns (form, filing_date, report_date, accession, primary_doc)
    Preference:
      1) 10-K over 10-K/A
      2) latest filing_date
    """
    rf = sub_json.get("filings", {}).get("recent", {})
    forms = rf.get("form", [])
    report_dates = rf.get("reportDate", [])
    filing_dates = rf.get("filingDate", [])
    accession_nos = rf.get("accessionNumber", [])
    primary_docs = rf.get("primaryDocument", [])

    candidates = []
    for i in range(len(forms)):
        form = forms[i]
        rd = report_dates[i] if i < len(report_dates) else None
        fd = filing_dates[i] if i < len(filing_dates) else None
        acc = accession_nos[i] if i < len(accession_nos) else None
        pdoc = primary_docs[i] if i < len(primary_docs) else None
        if not rd or not fd or not acc or not pdoc:
            continue
        if form not in ("10-K", "10-K/A"):
            continue

        # SEC reportDate is "YYYY-MM-DD"; using year as fiscal-year proxy (your current approach)
        try:
            fy = int(rd.split("-")[0])
        except Exception:
            continue

        if fy == target_fy:
            candidates.append((form, fd, rd, acc, pdoc))

    if not candidates:
        return None

    # Prefer 10-K over 10-K/A; within the same form type, prefer the latest filing date.
    candidates.sort(key=lambda x: (x[0] != "10-K", x[1]), reverse=True)
    # - x[0] != "10-K" -> False (0) for 10-K, True (1) for 10-K/A
    # - reverse=True puts 10-K (False=0) last when compared with 10-K/A (True=1),
    #   but since False < True, descending order puts 10-K/A first — we flip:
    # Cleaner: sort ascending by (is_amendment, date) and take the last element.
    candidates.sort(key=lambda x: (x[0] != "10-K", x[1]))  # ascending: 10-K first, oldest first
    best_form = candidates[0][0]
    same_form = [c for c in candidates if c[0] == best_form]
    same_form.sort(key=lambda x: x[1], reverse=True)  # latest filing date first
    return same_form[0]


def archives_primary_doc_url(cik10: str, accession: str, primary_doc: str) -> str:
    cik_nolead = str(int(cik10))  # remove leading zeros
    acc_nodash = accession.replace("-", "")
    return f"https://www.sec.gov/Archives/edgar/data/{cik_nolead}/{acc_nodash}/{primary_doc}"


def download_primary_html(
    ticker: str,
    fy: int,
    cik10: str,
    rec: Tuple[str, str, str, str, str],
    session: Optional[requests.Session] = None
) -> dict:
    form, filing_date, report_date, accession, primary_doc = rec
    url = archives_primary_doc_url(cik10, accession, primary_doc)
    r = http_get(url, headers=HEADERS_WWW, session=session)

    out = RAW_DIR / f"{ticker}_FY{fy}_{report_date}_{accession.replace('-', '')}_{primary_doc}"
    out.write_bytes(r.content)

    return {
        "ticker": ticker,
        "fy": fy,
        "form": form,
        "filing_date": filing_date,
        "report_date": report_date,
        "accession": accession,
        "primary_doc": primary_doc,
        "url": url,
        "path": str(out),
    }


# =========================
# HTML -> (TEXT + TABLE OBJECTS)
# =========================

def clean_text(text: str) -> str:
    text = re.sub(r"\u00a0", " ", text)       # nbsp
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def best_effort_table_title(tbl) -> str:
    # caption if present
    cap = tbl.find("caption")
    if cap:
        t = cap.get_text(" ", strip=True)
        if t:
            return t

    # try preceding heading-ish text
    prev = tbl.find_previous(["b", "strong", "font", "p", "div"])
    if prev:
        t = prev.get_text(" ", strip=True)
        if t and len(t) <= 160:
            return t

    return ""


def html_to_text_and_tables(html_bytes: bytes, table_prefix: str) -> Tuple[str, List[dict]]:
    """
    Extract tables as objects, replace them in DOM with stable placeholders,
    then convert remaining to text for narrative retrieval.
    """
    soup = BeautifulSoup(html_bytes, "html.parser")

    # Remove scripts/styles
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    tables: List[dict] = []

    all_tables = soup.find_all("table")
    for i, tbl in enumerate(all_tables):
        title = best_effort_table_title(tbl)
        table_id = f"{table_prefix}_T{i:03d}"

        # raw HTML retained for reproducibility
        table_html = str(tbl)

        # minimal surrogate fields for retrieval later
        tables.append({
            "table_id": table_id,
            "title": title,
            "html": table_html,
        })

        placeholder_text = f"\n[TABLE:{table_id} {title}]\n"
        tbl.replace_with(soup.new_string(placeholder_text))

    text = soup.get_text("\n")
    return clean_text(text), tables


# =========================
# SECTIONIZATION (Item 1A/7/8)
# =========================

ITEM_HDR = {
    "Item 1A": re.compile(r"\bitem\s+1a\b[\s\.\:\-–—]*risk\s+factors\b", re.IGNORECASE),
    "Item 7": re.compile(r"\bitem\s+7\b[\s\.\:\-–—]*management['’]s\s+discussion\b", re.IGNORECASE),
    "Item 8": re.compile(
        r"\bitem\s+8\b[\s\.\:\-–—]*financial\s+statements(\s+and\s+supp(le)?mentary\s+data)?\b",
        re.IGNORECASE,
    ),
}

ITEM8_FALLBACK = [
    re.compile(r"\breport\s+of\s+independent\s+registered\s+public\s+accounting\s+firm\b", re.IGNORECASE),
    re.compile(r"\bconsolidated\s+financial\s+statements\b", re.IGNORECASE),
    re.compile(r"\bfinancial\s+statements\s+and\s+supp(le)?mentary\s+data\b", re.IGNORECASE),
]


def guess_body_start(text: str) -> int:
    low = text.lower()
    base = min(len(text), 5000)
    candidates = []
    for token in ["table of contents", "part i", "item 1.", "item 1 ", "item 1a"]:
        idx = low.find(token, base)
        if idx != -1:
            candidates.append(idx)
    return min(candidates) if candidates else 0


def pick_best_header(text: str, pat: re.Pattern) -> Optional[int]:
    start_from = guess_body_start(text)
    matches = [m.start() for m in pat.finditer(text, pos=start_from)]
    if not matches:
        matches = [m.start() for m in pat.finditer(text)]
    return matches[-1] if matches else None  # last match avoids TOC


def pick_item8_start(text: str) -> Optional[int]:
    pos = pick_best_header(text, ITEM_HDR["Item 8"])
    if pos is not None:
        return pos

    start_from = guess_body_start(text)
    hits = []
    for pat in ITEM8_FALLBACK:
        for m in pat.finditer(text, pos=start_from):
            hits.append(m.start())
    return min(hits) if hits else None


def extract_items(text: str) -> Dict[str, str]:
    starts = {
        "Item 1A": pick_best_header(text, ITEM_HDR["Item 1A"]),
        "Item 7": pick_best_header(text, ITEM_HDR["Item 7"]),
        "Item 8": pick_item8_start(text),
    }
    found = [(k, v) for k, v in starts.items() if v is not None]
    found.sort(key=lambda x: x[1])

    sections: Dict[str, str] = {}
    for i, (item, pos) in enumerate(found):
        end = found[i + 1][1] if i + 1 < len(found) else len(text)
        sections[item] = text[pos:end].strip()
    # Ensure keys exist
    return {k: sections.get(k, "") for k in ["Item 1A", "Item 7", "Item 8"]}


# =========================
# XBRL COMPANYFACTS
# =========================

def get_companyfacts(cik10: str, session: Optional[requests.Session] = None) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik10}.json"
    return http_get(url, headers=HEADERS_DATA, session=session).json()


# =========================
# MAIN
# =========================

def main():
    print("Base:", BASE)

    with requests.Session() as session:
        # 1) ticker->cik
        ticker_to_cik = load_ticker_to_cik(session=session)
        for t in TICKERS:
            print(f"{t} -> {ticker_to_cik.get(t)}")

        # 2) submissions
        subs = {}
        for t in TICKERS:
            cik10 = ticker_to_cik[t]
            subs[t] = get_submissions(cik10, session=session)
            (META_DIR / f"{t}_submissions.json").write_text(json.dumps(subs[t], indent=2))
            print("Fetched submissions:", t, cik10)

        # 3) select filings for target fiscal years
        selected: Dict[str, Dict[int, Optional[Tuple[str, str, str, str, str]]]] = {}
        for t in TICKERS:
            selected[t] = {}
            for fy in TARGET_FYS:
                rec = select_10k_for_fy(subs[t], fy)
                selected[t][fy] = rec
                print(t, fy, "->", rec)

        # 4) download primary HTML
        downloads = []
        for t in TICKERS:
            cik10 = ticker_to_cik[t]
            for fy in TARGET_FYS:
                rec = selected[t][fy]
                if not rec:
                    continue
                meta = download_primary_html(t, fy, cik10, rec, session=session)
                downloads.append(meta)
                print("Saved:", meta["path"])

        (META_DIR / "downloads.json").write_text(json.dumps(downloads, indent=2))
        print("Downloads:", len(downloads))

        # 5) parse each HTML -> (narrative text w/ table placeholders) + tables
        wrote = 0
        for meta in downloads:
            p = Path(meta["path"])
            html = p.read_bytes()

            table_prefix = f"{meta['ticker']}_FY{meta['fy']}_{meta['accession'].replace('-', '')}"
            narrative_text, tables = html_to_text_and_tables(html, table_prefix=table_prefix)

            items = extract_items(narrative_text)

            # 6) save sections JSON
            out_sections = {
                "ticker": meta["ticker"],
                "fiscal_year": meta["fy"],
                "report_date": meta["report_date"],
                "filing_date": meta["filing_date"],
                "accession": meta["accession"],
                "primary_doc": meta["primary_doc"],
                "source_url": meta["url"],
                "items": items,
            }
            sect_path = SECT_DIR / f"{meta['ticker']}_FY{meta['fy']}_sections.json"
            sect_path.write_text(json.dumps(out_sections, indent=2))

            # 7) save tables JSON
            out_tables = {
                "ticker": meta["ticker"],
                "fiscal_year": meta["fy"],
                "report_date": meta["report_date"],
                "filing_date": meta["filing_date"],
                "accession": meta["accession"],
                "primary_doc": meta["primary_doc"],
                "source_url": meta["url"],
                "tables": tables,
            }
            tables_path = TABLE_DIR / f"{meta['ticker']}_FY{meta['fy']}_tables.json"
            tables_path.write_text(json.dumps(out_tables, indent=2))

            wrote += 1
            print(
                f"Wrote {sect_path.name} | lens:",
                {k: len(v) for k, v in items.items()},
                f"| tables: {len(tables)}",
            )

        print("Wrote section/table files:", wrote)

        # 8) fetch XBRL companyfacts (optional but recommended)
        for t in TICKERS:
            cik10 = ticker_to_cik[t]
            facts = get_companyfacts(cik10, session=session)
            (FACTS_DIR / f"{t}_companyfacts.json").write_text(json.dumps(facts, indent=2))
            print("Saved companyfacts:", t)

    print("Done.")

