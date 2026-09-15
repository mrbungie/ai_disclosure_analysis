"""
scripts/bronze/call_transcripts.py — bronze.call_transcripts: one row per
earnings-call transcript of bronze.filing_manifest (the three raw sources,
huggingface kurry/sp500_earnings_transcripts, equibles, stockanalysis.com)
with what its own text says about it and whether silver keeps it.

The source metadata (ticker, `date`, `year`/`quarter`) is not reliable on its
own: stockanalysis.com serves the acquirer's calls under an acquired ticker,
some transcripts are conferences or investor days, some dates are a quarter
off, and the same call reaches the corpus twice under two fiscal-year labels.
Each transcript is checked against its intro (operator/host opening, the first
INTRO_CHARS characters):

  company       `host_company`: the company the intro welcomes to ("welcome
                to X's first quarter ..."); `other_company` when it is not one of
                the firm's names (the firm-universe name, the EDGAR names of the
                ticker's CIK, current and former, bronze.sec_company_names, the
                ticker, initials, or a close spelling). `company_named`: one of
                those names appears in the text head (descriptive).
  event type    `event_type` results | other_event: a results call names the
                quarter, the results or the earnings; investor/analyst days,
                broker conferences, fireside chats, shareholder meetings and
                M&A calls are other events. An investor/analyst day on the day
                of an 8-K Item 2.02 results release (±1) is the quarter's
                results event when the ticker has no results call that week
                (Target's financial community meeting). A business update or
                investor event without results language is a results call
                only near a release (±RELEASE_DAYS).
  call date     `call_date`: the date the transcript states ("Today is
                Thursday, August 25th, 2022", "is being recorded on ..."),
                otherwise the source metadata (`call_date_source` stated |
                metadata | release_8k, set by the sequence repair below).
  fiscal period `fiscal_period` ('2024Q1'): the first quarter the intro names
                and the fiscal year next to it ("first quarter fiscal 2024",
                "Q1 2024"), otherwise the source label (`fiscal_period_source`
                stated | metadata | sequence).
  duplicates    transcripts of one ticker with similar text (>= SIMILAR), dates
                within DUPLICATE_DAYS, or the same fiscal period are one call;
                the kept one is the first by source priority (huggingface >
                equibles > stockanalysis.com), then length, and takes the
                group's best date.
  sequence      per ticker, kept calls in date order must be >= MIN_GAP_DAYS
                apart and advance their fiscal period by k >= 1 quarters within
                [91k - SEQUENCE_EARLY, 91k + SEQUENCE_LATE] days (k > 1: missing
                calls). A breaking transcript first takes an alternative period
                or date (`repair_sequence`); otherwise it is excluded, or the
                pair is kept as a fiscal-year change (`call_sequence`).

`exclude_reason` (null = kept): other_event, other_company, duplicate,
sequence. `duplicate_of` names the kept transcript. Nothing is deleted:
silver.filing_manifest drops the excluded rows and takes `call_date` /
`fiscal_period` as the call's filing_date / fiscal_period. The manifest lists
the sequence exceptions (gaps and fiscal-year changes).
"""

from __future__ import annotations

import gzip
import json
import re
import unicodedata
import zlib
from concurrent.futures import ProcessPoolExecutor
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import polars as pl
from _paths import L, log

BUILDER = "scripts/bronze/call_transcripts.py"
CALL_FORM_TYPE = "Earnings call transcript"
SOURCE_PRIORITY = ["huggingface:", "equibles:", "stockanalysis.com:"]
INTRO_CHARS = 1200
HEAD_CHARS = 4000
RELEASE_DAYS = 3
DUPLICATE_DAYS = 7
SIMILAR = 0.5
MIN_GAP_DAYS = 30
SEQUENCE_EARLY = 60  # a quarter's call can come 60 days early (after a late fourth-quarter call)
SEQUENCE_LATE = 70  # ... or 70 days late (the fourth-quarter call waits for the annual audit)
STATED_DATE_MAX_GAP = 120

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
STATED_DATE = re.compile(
    rf"(?:today is|today,|(?:being|was) recorded (?:today|on)|as of today)"
    rf"\s*,?\s*(?:(?:mon|tues|wednes|thurs|fri|satur|sun)day,?\s*(?:the\s+)?)?({MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(20\d\d)",
    re.I)
ORDINAL = {"first": 1, "1st": 1, "second": 2, "2nd": 2, "third": 3, "3rd": 3, "fourth": 4, "4th": 4}
_ORD = r"(first|second|third|fourth|1st|2nd|3rd|4th)"
QUARTER_MENTION = re.compile(rf"\b{_ORD}[\s-]+(?:fiscal\s+)?quarter\b|\bQ([1-4])\b", re.I)
YEAR_BEFORE = re.compile(r"(?:\b(?:fiscal\s+(?:year\s+)?)?(20\d\d)|\bFY\s*['‘’]?(\d\d|20\d\d)|['‘’](\d\d))(?:\s+fiscal)?[\s,]*$", re.I)
YEAR_AFTER = re.compile(r"^(?:(?!quarter\b|outlook|guidance|\bQ[1-4]\b)[^.;:!?])*?"
                        r"(?:\b(20\d\d)\b|\bFY\s*['‘’]?(\d\d)\b|['‘’](\d\d)\b)", re.I)
RESULTS = re.compile("|".join([
    r"\bearnings\b", r"\bresults\b", r"\bfinancial (?:teleconference|review)\b", r"\btrading update\b",
    rf"\b{_ORD}[\s-]+(?:fiscal\s+)?quarter\b", r"\bQ[1-4]\b", r"\b(?:full|fiscal)[\s-]year\b", r"\bhalf[\s-]year\b",
]), re.I)
_BROKERS = (r"wells fargo|goldman sachs|morgan stanley|j\.?\s?p\.?\s?morgan|bernstein|ubs|barclays|citi|bofa|bank of america|"
            r"merrill|jefferies|cowen|keybanc|rbc|deutsche bank|evercore|raymond james|credit suisse|baird|oppenheimer|"
            r"stifel|piper sandler|william blair|wolfe|needham|guggenheim|mizuho|bmo|truist|cantor|leerink|macquarie|"
            r"hsbc|bnp|kbw|wedbush|susquehanna|canaccord|stephens|scotiabank")
OTHER_EVENT = re.compile("|".join([
    r"\b(?:investor|analyst)s?(?: and (?:investor|analyst)s?)? (?:day|meeting)\b", r"\bcapital markets? day\b",
    r"\bfinancial community meeting\b", r"\bfireside\b", r"\bstrategy (?:day|session)\b",
    r"\bannual (?:general )?(?:meeting|(?:share|stock)holders?'? meeting)\b", r"\b(?:share|stock)holders?'? meeting\b",
    rf"\b(?:{_BROKERS})\b[\w&.'’ -]{{0,60}}?\b(?:conference|summit|symposium|forum)\b(?!\s+call)",
    r"\b(?:healthcare|health care|technology|tech|tmt|media|consumer|industrials?|energy|financials?|retail|global|"
    r"leveraged finance|growth)\s+(?:[a-z&]+\s+){0,2}(?:conference|summit|symposium|forum)\b(?!\s+call)",
    r"\bto discuss (?:the |our |its |this )?(?:proposed |pending |announced |definitive |planned )?"
    r"(?:acquisition|merger|combination|transaction|agreement|spin[- ]?off|separation)\b",
    r"\b(?:acquisition|merger|combination) (?:announcement |conference )?call\b", r"\bjoint call\b",
]), re.I)
AMBIGUOUS_EVENT = re.compile(r"\bbusiness update\b|\binvestor (?:event|update|conference call)\b|\bstrategic update\b", re.I)
BODY_RESULTS = re.compile(r"\b(?:this|the|last|prior) quarter\b|\bquarterly\b|\bguidance\b|\bprepared remarks\b", re.I)
HOST = re.compile(
    r"(?:welcome (?:everyone |you |all )?to|welcome to|joining (?:us )?(?:for|today for|on)|facilitator today for|"
    r"to review|for joining)\s+(?:the |today['’]s |this )?"
    r"(?:(?:Q[1-4]|first|second|third|fourth|fiscal|FY|20\d\d)[\w'’ ]{0,20}?\s+)??"
    r"(?P<name>[A-Z][\w&.,'’\- ]{1,60}?)(?:['’]s?)?\s+(?:\(?[A-Z]{1,5}\)?\s+)?"
    r"(?=(?:20\d\d|FY|fiscal|first|second|third|fourth|1st|2nd|3rd|4th|Q[1-4]|full|half|quarter|earnings|financial|"
    r"conference|investor|results|annual|business|analyst|capital|call|webcast|teleconference|trading)\b)",
    re.I)
NAME_STOPWORDS = r"\b(?:inc|incorporated|corp|corporation|co|company|companies|ltd|limited|plc|holdings?|group|the|llc|lp|nv|sa|ag|de|new|trust|intl|class [ab]|cl [ab]|n v|l p|s a)\b"
HOST_GENERIC = {"our", "the", "this", "today", "everyone", "you", "all", "us", "we", "conference", "earnings", "quarter",
                "quarterly", "fiscal", "call", "results", "investor", "analyst", "annual", "business", "a", "an", "company",
                "full", "year", "first", "second", "third", "fourth", "for", "to", "discuss", "and", "welcome", "joining", "me",
                "on", "ladies", "gentlemen", "good", "morning", "afternoon", "day", "thank", "thanks", "half", "months",
                "winter", "spring", "summer", "fall", "transition", "strategic", "announcement", "portion", "session",
                "q&a", "of", "corp", "corporation", "inc", "incorporated", "companies", "group", "holdings",
                "january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
                "november", "december", "nine", "six", "three", "twelve", "presentation", "update", "webcast"}


def norm(text: str) -> str:
    """Lowercase ASCII (accents dropped), possessives and punctuation removed,
    company suffixes kept."""
    text = text.replace("’", "'").replace("‘", "'")
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    text = re.sub(r"'s\b", " ", text).replace("'", "")
    return " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9&]+", " ", text)).strip() + " "


def name_aliases(name: str) -> set[str]:
    """Match forms of a company name: without legal suffixes, its first two
    words, its first word when distinctive, and the ampersand-less form."""
    n = re.sub(r"\s+", " ", re.sub(NAME_STOPWORDS, " ", norm(name))).strip()
    words = n.split()
    full = [w for w in norm(name).split() if w not in ("inc", "corp", "co", "ltd", "plc", "llc", "the")]
    out = {n} if len(n) >= 2 else set()
    if len(words) >= 2:
        out.add(" ".join(words[:2]))
    if words and (len(words[0]) >= 5 or len(words) == 1):
        out.add(words[0])
    for w in (words, full):
        if len(w) >= 2:
            out.add("".join(x[0] for x in w))  # initials: Willis Towers Watson -> wtw, Public Service Enterprise Group -> pseg
    out |= {a.replace(" & ", " and ") for a in out if "&" in a}
    return {a for a in out if a}


def read_transcript(path: str) -> tuple:
    with gzip.open(path) as f:
        data = json.load(f)
    content = data.get("content") or ""
    words = re.findall(r"[a-z0-9]+", content[1500:15000].lower())
    shingles = {zlib.crc32(" ".join(words[i:i + 6]).encode()) for i in range(0, max(len(words) - 6, 0), 2)}
    sketch = sorted(shingles)[:256]  # bottom-k sketch of 6-word shingles
    return content[:HEAD_CHARS], len(content), sketch


def stated_period(intro: str) -> str | None:
    """'2024Q1' from the first quarter the intro names and the fiscal year next
    to it ("2024 first quarter", "first quarter of fiscal 2024", "Q1 FY'24"),
    within the same sentence; None when that mention has no year."""
    m = QUARTER_MENTION.search(intro)
    if not m:
        return None
    word = m.group(1) or ""
    quarter = int(m.group(2)) if m.group(2) else ORDINAL[word.lower()]
    year = YEAR_BEFORE.search(intro[max(0, m.start() - 20):m.start()]) or YEAR_AFTER.search(intro[m.end():m.end() + 60])
    if not year:
        return None
    value = int(next(g for g in year.groups() if g))
    return f"{value + 2000 if value < 100 else value}Q{quarter}"


def stated_date(head: str):
    m = STATED_DATE.search(head)
    if not m:
        return None
    try:
        return pd.Timestamp(f"{m.group(1)} {m.group(2)} {m.group(3)}")
    except ValueError:
        return None


def host_company(intro: str) -> str | None:
    """The company the intro welcomes to: a capitalised name before the call's
    description ("welcome to the Hilton Third Quarter ...")."""
    for m in HOST.finditer(intro[:800]):
        raw = m.group("name").strip(" ,.")
        words = [w for w in norm(raw).split() if w not in HOST_GENERIC and not re.fullmatch(r"q[1-4]|\d+[a-z]*|fy\d*|[a-z]", w)]
        if (raw[:1].isupper() and words and len(max(words, key=len)) >= 3 and norm(raw).split()[0] not in HOST_GENERIC
                and norm(raw).replace(" ", "") not in HOST_GENERIC):
            return raw
    return None


def is_firm(name: str, aliases: set[str]) -> bool:
    """Whether a host name is one of the firm's names: an alias inside it, or
    the same letters as an alias up to a prefix or a typo (ExxonMobil, Lowe's
    Companies, Revitty)."""
    n = norm(name)
    squashed = re.sub(NAME_STOPWORDS, " ", n).replace(" ", "")
    alias_words = {w for a in aliases for w in a.split()}
    if squashed in {w for w in alias_words if len(w) >= 3} or squashed in {a.split()[0] for a in aliases}:  # Sands, Eli, M&T
        return True
    first = n.split()[0] if n.split() else ""
    if len(first) >= 4 and any(w.startswith(first) or first.startswith(w) for w in alias_words if len(w) >= 4):  # Lowe's
        return True
    for a in aliases:
        sa = a.replace(" ", "")
        if f" {a} " in n or squashed == sa or len(sa) >= 5 and squashed.startswith(sa) or len(squashed) >= 4 and sa.startswith(squashed):
            return True
        if len(sa) >= 5 and SequenceMatcher(None, squashed[:len(sa)], sa).ratio() >= 0.85:
            return True
        if len(squashed) <= 8 and SequenceMatcher(None, squashed, sa).ratio() >= 0.75:  # CCSX, Lenard
            return True
    return False


def event_type(intro: str, head: str, release_gap: float) -> tuple[str, str | None]:
    near_release = release_gap <= RELEASE_DAYS
    other, results = OTHER_EVENT.search(intro), RESULTS.search(intro)
    if other and (not results or other.start() < results.start()) and not RESULTS.search(other.group(0)):
        return ("release_day_event" if release_gap <= 1 else "other_event"), other.group(0)
    ambiguous = AMBIGUOUS_EVENT.search(intro)
    if ambiguous and not re.search(r"\bearnings\b|\bresults\b|\bquarter\b|\bQ[1-4]\b", intro[:ambiguous.end() + 80], re.I):
        return ("results", ambiguous.group(0)) if near_release else ("other_event", ambiguous.group(0))
    if results:
        return "results", results.group(0)
    body = BODY_RESULTS.search(head)
    if body or near_release:
        return "results", body.group(0) if body else "8-K Item 2.02 release"
    return "other_event", None


def quarter_index(period: str) -> int:
    return int(period[:4]) * 4 + int(period[-1]) - 1


def period_label(index: int) -> str:
    return f"{index // 4}Q{index % 4 + 1}"


def release_gaps(calls: pd.DataFrame, releases: pd.DataFrame, date_col: str) -> np.ndarray:
    """Days from each call to the nearest 8-K Item 2.02 filing of its CIK."""
    by_cik = {c: np.sort(g["filing_date"].to_numpy(dtype="datetime64[D]")) for c, g in releases.groupby("cik")}
    out = np.full(len(calls), np.inf)
    for i, (cik, date) in enumerate(zip(calls["cik"], calls[date_col])):
        dates = by_cik.get(cik)
        if dates is None or pd.isna(date):
            continue
        d = np.datetime64(pd.Timestamp(date).date(), "D")
        j = np.searchsorted(dates, d)
        out[i] = min(abs((dates[k] - d).astype(int)) for k in (j - 1, j) if 0 <= k < len(dates)) if len(dates) else np.inf
    return out


def company_names() -> dict[str, set[str]]:
    universe = L.read("bronze.firm_universe").select("ticker", "cik", "company_name").to_pandas()
    edgar = L.read("bronze.sec_company_names").to_pandas().groupby("cik")["name"].apply(list).to_dict()
    aliases = {}
    for ticker, cik, name in zip(universe["ticker"], universe["cik"], universe["company_name"]):
        found = (name_aliases(name) if isinstance(name, str) else set()) | {ticker.lower().replace(".", " ").replace("-", " ")}
        for other in edgar.get(cik, []):
            found |= name_aliases(other)
        aliases[ticker] = found
    return aliases


def jaccard(a, b) -> float:
    if not len(a) or not len(b):
        return 0.0
    sa, sb = set(a), set(b)
    union = sorted(sa | sb)[:256]
    return sum(1 for x in union if x in sa and x in sb) / len(union)


def same_call_groups(calls: pd.DataFrame) -> dict[int, int]:
    """Union-find over the transcripts of one ticker that are the same call:
    similar text, dates within DUPLICATE_DAYS (unless both intros name
    different quarters), or the same fiscal period (named by both intros, or
    within 45 days). Returns row -> group root."""
    parent = {i: i for i in calls.index}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for _, g in calls.groupby("ticker"):
        g = g.sort_values("call_date")
        rows = list(g.itertuples())
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                ra, rb = rows[a], rows[b]
                gap = (rb.call_date - ra.call_date).days
                if gap > 200:
                    break
                both_stated = pd.notna(ra.stated_fiscal_period) and pd.notna(rb.stated_fiscal_period)
                same = (jaccard(ra.sketch, rb.sketch) >= SIMILAR
                        or gap <= DUPLICATE_DAYS and not (both_stated and ra.fiscal_period != rb.fiscal_period)
                        or ra.fiscal_period == rb.fiscal_period and (both_stated or gap <= 45))
                if same:
                    parent[find(rb.Index)] = find(ra.Index)
    return {i: find(i) for i in calls.index}


def main() -> None:
    manifest = (L.scan("bronze.filing_manifest").filter(pl.col("form_type") == CALL_FORM_TYPE)
                .select("document_id", "ticker", "source", "local_path", pl.col("filing_date").alias("metadata_date"),
                        pl.col("fiscal_period").alias("metadata_fiscal_period"))
                .collect().to_pandas())
    log(f"reading {len(manifest):,} transcripts")
    with ProcessPoolExecutor(8) as pool:
        texts = list(pool.map(read_transcript, manifest["local_path"], chunksize=50))
    calls = manifest.assign(head=[t[0] for t in texts], n_chars=[t[1] for t in texts], sketch=[t[2] for t in texts])
    calls["source_rank"] = [next(i for i, p in enumerate(SOURCE_PRIORITY) if s.startswith(p)) for s in calls["source"]]
    calls["metadata_date"] = pd.to_datetime(calls["metadata_date"])
    calls["cik"] = calls["ticker"].map(dict(L.read("bronze.firm_universe").select("ticker", "cik").iter_rows()))
    releases = (L.scan("bronze.sec_filing_index").filter(pl.col("form").is_in(["8-K", "8-K/A"]) & pl.col("items").str.contains("2.02"))
                .select("cik", "filing_date").collect().to_pandas())
    releases_by_cik = {c: np.sort(g["filing_date"].to_numpy(dtype="datetime64[D]")) for c, g in releases.groupby("cik")}

    # company
    aliases = company_names()
    heads = calls["head"].map(norm)
    calls["company_named"] = [any(f" {a} " in h for a in aliases.get(t, ())) for t, h in zip(calls["ticker"], heads)]
    calls["host_company"] = calls["head"].str[:INTRO_CHARS].map(host_company)
    host_is_firm = [pd.isna(h) or is_firm(h, aliases.get(t, set())) for t, h in zip(calls["ticker"], calls["host_company"])]
    calls["other_company"] = ~pd.Series(host_is_firm, index=calls.index)

    # date and fiscal period
    calls["stated_date"] = calls["head"].map(stated_date)
    stated_ok = calls["stated_date"].notna() & ((calls["stated_date"] - calls["metadata_date"]).dt.days.abs() <= STATED_DATE_MAX_GAP)
    calls["call_date"] = calls["stated_date"].where(stated_ok, calls["metadata_date"]).astype("datetime64[ns]")
    calls["call_date_source"] = np.where(stated_ok, "stated", "metadata")
    calls["stated_fiscal_period"] = calls["head"].str[:INTRO_CHARS].map(stated_period)
    calls["fiscal_period"] = calls["stated_fiscal_period"].fillna(calls["metadata_fiscal_period"])
    calls["fiscal_period_source"] = np.where(calls["stated_fiscal_period"].notna(), "stated", "metadata")
    calls["release_gap_days"] = release_gaps(calls, releases, "call_date")

    # event type
    events = [event_type(h[:INTRO_CHARS], h, gap) for h, gap in zip(calls["head"], calls["release_gap_days"])]
    calls["event_type"] = [e[0] for e in events]
    calls["event_evidence"] = [e[1] for e in events]

    # an investor/analyst day held on the results release day is the quarter's
    # results event unless the ticker also has a results call that week
    for i in calls.index[calls["event_type"] == "release_day_event"]:
        r = calls.loc[i]
        week = calls[(calls["ticker"] == r["ticker"]) & (calls["event_type"] == "results")
                     & ((calls["call_date"] - r["call_date"]).dt.days.abs() <= DUPLICATE_DAYS)]
        calls.loc[i, "event_type"] = "other_event" if len(week) else "results"
    calls["exclude_reason"] = np.select([calls["event_type"] == "other_event", calls["other_company"]],
                                        ["other_event", "other_company"], None)
    calls["duplicate_of"] = None

    # duplicates: the kept transcript takes the group's fiscal period (named by
    # an intro) and best date (stated, then the source that labels the call
    # with that period, then the one closest to an 8-K Item 2.02 release)
    kept = calls[calls["exclude_reason"].isna()]
    groups = same_call_groups(kept)
    order = kept.sort_values(["source_rank", "n_chars"], ascending=[True, False])
    representative = {}
    for i in order.index:
        representative.setdefault(groups[i], i)
    for root, rep in representative.items():
        members = [i for i in kept.index if groups[i] == root]
        stated = calls.loc[members, "stated_fiscal_period"].dropna()
        if pd.isna(calls.loc[rep, "stated_fiscal_period"]) and len(stated):
            calls.loc[rep, ["fiscal_period", "fiscal_period_source"]] = [stated.mode().iloc[0], "stated"]
        period = calls.loc[rep, "fiscal_period"]
        best = min(members, key=lambda i: (calls.loc[i, "call_date_source"] != "stated",
                                           calls.loc[i, "metadata_fiscal_period"] != period,
                                           calls.loc[i, "release_gap_days"], i != rep))
        calls.loc[rep, ["call_date", "call_date_source", "release_gap_days"]] = calls.loc[best, ["call_date", "call_date_source",
                                                                                                "release_gap_days"]].tolist()
        for i in members:
            if i != rep:
                calls.loc[i, ["exclude_reason", "duplicate_of"]] = ["duplicate", calls.loc[rep, "document_id"]]

    calls["call_sequence"] = None
    for _, g in calls[calls["exclude_reason"].isna()].groupby("ticker"):
        repair_sequence(calls, list(g.index), releases_by_cik.get(g["cik"].iloc[0]))
    kept = calls[calls["exclude_reason"].isna()].sort_values(["ticker", "call_date"])
    previous = kept.groupby("ticker")[["document_id", "fiscal_period", "call_date"]].shift()
    exceptions = [{"ticker": r.ticker, "call_sequence": r.call_sequence, "from": p.fiscal_period, "to": r.fiscal_period,
                   "days": int((r.call_date - p.call_date).days)}
                  for (_, r), (_, p) in zip(kept.iterrows(), previous.iterrows()) if r.call_sequence in ("gap", "label_break")]
    out = calls.drop(columns=["head", "sketch", "local_path", "cik", "stated_date_candidate"], errors="ignore")
    out["release_gap_days"] = out["release_gap_days"].replace(np.inf, np.nan)
    summary = {"transcripts": len(out), "kept": int(out["exclude_reason"].isna().sum()),
               "excluded": {k: int(v) for k, v in out["exclude_reason"].value_counts().items()},
               "fiscal_period_source": {k: int(v) for k, v in out.loc[out["exclude_reason"].isna(), "fiscal_period_source"].value_counts().items()},
               "call_date_source": {k: int(v) for k, v in out.loc[out["exclude_reason"].isna(), "call_date_source"].value_counts().items()},
               "sequence_exceptions": exceptions}
    L.write_table("bronze.call_transcripts", pl.from_pandas(out), keys=["document_id"], sort_by=["ticker", "call_date", "document_id"],
                  inputs=[L.path("bronze.filing_manifest"), L.path("bronze.firm_universe"), L.path("bronze.sec_company_names"),
                          L.path("bronze.sec_filing_index")],
                  builder=BUILDER, extra=summary)
    log(json.dumps({k: v for k, v in summary.items() if k != "sequence_exceptions"}) + f" | {len(exceptions)} sequence exceptions")


def repair_sequence(calls: pd.DataFrame, rows: list, releases) -> None:
    """Enforce the sequence invariant on one ticker's kept transcripts.

    Walking the calls in date order, at the first consecutive pair that breaks
    it the later transcript tries its alternative (date, fiscal period): the
    other of stated/metadata date, the 8-K Item 2.02 release nearest to where
    the sequence expects the call (previous call + 91 days per quarter
    advanced, within 30 days, and within STATED_DATE_MAX_GAP of the metadata
    date), the other of stated/metadata period, and a metadata period moved by
    a year. The first alternative consistent with the previous call is kept
    (for the ticker's first pair, the earlier transcript may also change to
    fit the later one). Failing that, the run of metadata-labelled calls
    ending at the earlier transcript moves by a fiscal year if that fits. Otherwise the transcript whose removal restores the
    sequence is excluded (`sequence`), the lower-priority one on a tie; when
    neither removal helps, the dates are one quarter apart and the sequence
    continues consistently after the pair, the fiscal labels restart there (a
    fiscal-year change) and the pair is kept as a `label_break`.
    Sets `call_sequence` on the kept transcripts: first | next | gap (k > 1
    quarters, missing calls) | label_break."""
    rows, breaks = set(rows), set()
    while True:
        seq = sorted(rows, key=lambda i: (calls.loc[i, "call_date"], calls.loc[i, "source_rank"]))
        bad = next(((a, b) for a, b in zip(seq, seq[1:])
                    if (a, b) not in breaks and not sequence_step(calls.loc[a], calls.loc[b])[0]), None)
        if bad is None:
            break
        a, b = bad
        pos = seq.index(a)
        if any(_try_alternatives(calls, rows, b, releases, "previous", release) or
               pos == 0 and _try_alternatives(calls, rows, a, releases, "next", release) for release in (False, True)) \
                or _try_shift_run(calls, seq, pos):
            continue
        prev = calls.loc[seq[pos - 1]] if pos > 0 else None
        nxt = calls.loc[seq[pos + 2]] if pos + 2 < len(seq) else None
        drop_a = prev is None or sequence_step(prev, calls.loc[b])[0]
        drop_b = nxt is None or sequence_step(calls.loc[a], nxt)[0]
        days = (calls.loc[b, "call_date"] - calls.loc[a, "call_date"]).days
        if not drop_a and not drop_b and 91 - SEQUENCE_EARLY <= days <= 91 + SEQUENCE_LATE and days >= MIN_GAP_DAYS:
            breaks.add((a, b))
            continue
        if drop_a != drop_b:
            loser = a if drop_a else b
        else:
            loser = max((a, b), key=lambda i: (calls.loc[i, "source_rank"], -calls.loc[i, "n_chars"]))
        calls.loc[loser, ["exclude_reason", "duplicate_of"]] = ["sequence", calls.loc[b if loser == a else a, "document_id"]]
        rows.discard(loser)
    seq = sorted(rows, key=lambda i: calls.loc[i, "call_date"])
    if seq:
        calls.loc[seq[0], "call_sequence"] = "first"
    for a, b in zip(seq, seq[1:]):
        calls.loc[b, "call_sequence"] = ("label_break" if (a, b) in breaks
                                         else "gap" if sequence_step(calls.loc[a], calls.loc[b])[1] > 1 else "next")


def _try_shift_run(calls: pd.DataFrame, seq: list, pos: int) -> bool:
    """Move the run of metadata-labelled calls ending at seq[pos] by one fiscal
    year when that makes it consistent with the next call and with the call
    before the run (a source whose fiscal-year convention is off by a year
    over several quarters)."""
    b = calls.loc[seq[pos + 1]]
    start = pos
    while start >= 0 and calls.loc[seq[start], "fiscal_period_source"] == "metadata":
        start -= 1
    run = seq[start + 1:pos + 1]
    if not run:
        return False
    before = calls.loc[seq[start]] if start >= 0 else None
    for shift in (-4, 4):
        moved = [period_label(quarter_index(calls.loc[i, "fiscal_period"]) + shift) for i in run]
        last, first = calls.loc[run[-1]].copy(), calls.loc[run[0]].copy()
        last["fiscal_period"], first["fiscal_period"] = moved[-1], moved[0]
        if sequence_step(last, b)[0] and (before is None or sequence_step(before, first)[0]):
            for i, period in zip(run, moved):
                calls.loc[i, ["fiscal_period", "fiscal_period_source"]] = [period, "sequence"]
            return True
    return False


def _try_alternatives(calls: pd.DataFrame, rows: set, x, releases, against: str, release_dates: bool) -> bool:
    """Give transcript x the first alternative (fiscal period, date) consistent
    with its previous (or next) kept call. Without `release_dates` only the
    period, the stated or the metadata date change; with it, only a move to
    an 8-K Item 2.02 release date."""
    r = calls.loc[x]
    others = sorted((i for i in rows if i != x), key=lambda i: calls.loc[i, "call_date"])
    periods = [(r["fiscal_period"], r["fiscal_period_source"]), (r["metadata_fiscal_period"], "metadata")]
    if pd.notna(r["stated_fiscal_period"]):
        periods.append((r["stated_fiscal_period"], "stated"))
    if r["fiscal_period_source"] != "stated":
        periods += [(period_label(quarter_index(r["metadata_fiscal_period"]) + d), "sequence") for d in (-4, 4)]
    for period, period_source in periods:
        if release_dates:
            prev = [i for i in others if calls.loc[i, "call_date"] < r["call_date"]]
            if releases is None or not prev:
                continue
            p = calls.loc[prev[-1]]
            expected = p["call_date"] + pd.Timedelta(days=91 * (quarter_index(period) - quarter_index(p["fiscal_period"])))
            near = releases[np.abs((releases - np.datetime64(expected.date(), "D")).astype(int)) <= 30]
            candidates = [(pd.Timestamp(d), "release_8k") for d in near
                          if abs((pd.Timestamp(d) - r["metadata_date"]).days) <= STATED_DATE_MAX_GAP]
        else:
            candidates = [(r["call_date"], r["call_date_source"]), (r["metadata_date"], "metadata")]
            if pd.notna(r["stated_date"]):
                candidates.append((r["stated_date"], "stated"))
        for date, date_source in candidates:
            if (pd.Timestamp(date), period) == (r["call_date"], r["fiscal_period"]):
                continue
            trial = r.copy()
            trial["call_date"], trial["fiscal_period"] = pd.Timestamp(date), period
            before = [calls.loc[i] for i in others if calls.loc[i, "call_date"] <= trial["call_date"]]
            after = [calls.loc[i] for i in others if calls.loc[i, "call_date"] > trial["call_date"]]
            neighbour_ok = (not before or sequence_step(before[-1], trial)[0]) if against == "previous" else (
                not after or sequence_step(trial, after[0])[0])
            if neighbour_ok:
                calls.loc[x, ["call_date", "call_date_source", "fiscal_period", "fiscal_period_source"]] = [
                    pd.Timestamp(date), date_source, period, period_source]
                return True
    return False


def sequence_step(a, b) -> tuple[bool, int]:
    """(invariant holds, quarters advanced) between consecutive kept calls."""
    days = (b["call_date"] - a["call_date"]).days
    k = quarter_index(b["fiscal_period"]) - quarter_index(a["fiscal_period"])
    return days >= MIN_GAP_DAYS and k >= 1 and 91 * k - SEQUENCE_EARLY <= days <= 91 * k + SEQUENCE_LATE, k


if __name__ == "__main__":
    main()
