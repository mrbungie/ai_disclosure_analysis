"""
scripts/bronze/call_transcripts.py — bronze.call_transcripts: one row per
earnings-call transcript of bronze.filing_manifest (the three raw sources,
huggingface kurry/sp500_earnings_transcripts, equibles, stockanalysis.com and
the Equibles backfill)
with what its own text says about it and whether silver keeps it.

The source metadata (ticker, `date`, `year`/`quarter`) is not reliable on its
own: stockanalysis.com serves the acquirer's calls under an acquired ticker,
some transcripts are conferences or investor days, some dates are a quarter
off, and the same call reaches the corpus twice under two fiscal-year labels.
Each transcript is checked against its intro (operator/host opening, the first
INTRO_CHARS characters):

  company       `host_company`: the company the intro welcomes to ("welcome
                to X's first quarter ..."). A results call whose host names a
                firm-universe firm more specifically than the names of the
                ticker it was filed under belongs to that firm (`ticker`;
                `source_ticker` keeps the source's): Paramount Global's calls
                under PSKY are PARA's, WestRock's under SW are WRK's; a former vendor
                symbol the universe does not carry at all is mapped by
                configs/us/ticker_aliases.csv (FOXA -> FOX, PEAK -> DOC). `other_company` when it is not one of
                the firm's names (the firm-universe name, the EDGAR names of the
                ticker's CIK, current and former, bronze.sec_company_names, the
                ticker, initials, or a close spelling). `company_named`: one of
                those names appears in the text head (descriptive).
  event type    `event_type` results | other_event: a results call names the
                quarter, the results or the earnings; investor/analyst days,
                broker conferences, fireside chats, shareholder meetings and
                M&A calls are other events. An investor/analyst day on the day
                of an 8-K Item 2.02 results release (±1) is the quarter's
                results event (Target's financial community meeting); a
                broker-hosted conference or fireside chat on that day is not. A business update or
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
                equibles > stockanalysis.com > Equibles backfill), then
                length, and takes the group's best date. Enriched transcripts
                come first, and an investor/analyst day loses to a results call
                of the same week.
  sequence      per ticker, kept calls in date order must be >= MIN_GAP_DAYS
                apart and advance their fiscal period by k >= 1 quarters within
                [91k - SEQUENCE_EARLY, 91k + SEQUENCE_LATE] days (k > 1: missing
                calls). A breaking transcript first takes an alternative period
                or date; a pair that breaks it is labelled, never rewritten, and the
                pair is kept as a fiscal-year change (`call_sequence`).

`exclude_reason` (null = kept): other_event, other_company, duplicate,
pending_enrichment (a call no other transcript covers, without scored
paragraphs or LLM outputs yet: extraction -> embeddings -> prefilter -> LLM
before it can enter silver), sequence. `duplicate_of` names the kept transcript. Nothing is deleted:
silver.filing_manifest drops the excluded rows and takes `call_date` /
`fiscal_period` as the call's filing_date / fiscal_period. The manifest lists
the sequence exceptions (gaps and fiscal-year changes).
"""

from __future__ import annotations

import gzip
import heapq
import json
import re
import unicodedata
import zlib
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
import polars as pl
from _paths import L, log

BUILDER = "scripts/bronze/call_transcripts.py"
LISTING_CIKS = L.REPO_ROOT / "configs" / "us" / "listing_ciks.csv"
TICKER_ALIASES = L.REPO_ROOT / "configs" / "us" / "ticker_aliases.csv"
SQUASHED: dict[str, str] = {}  # squashed name phrase -> phrase, filled by main()
CALL_FORM_TYPE = "Earnings call transcript"
SOURCE_PRIORITY = ["huggingface:", "equibles:", "stockanalysis.com:", "equibles_backfill:"]
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
COMPANY_EVENT = re.compile("|".join([
    r"\b(?:investor|analyst)s?(?: and (?:investor|analyst)s?)? (?:day|meeting)\b", r"\bcapital markets? day\b",
    r"\bfinancial community meeting\b", r"\bstrategy (?:day|session)\b",
]), re.I)
EXTERNAL_EVENT = re.compile("|".join([
    r"\bfireside\b",
    r"\bannual (?:general )?(?:meeting|(?:share|stock)holders?'? meeting)\b", r"\b(?:share|stock)holders?'? meeting\b",
    rf"\b(?:{_BROKERS})\b[\w&.'’ -]{{0,60}}?\b(?:conference|summit|symposium|forum)\b(?!\s+call)",
    r"\b(?:healthcare|health care|technology|tech|tmt|media|consumer|industrials?|energy|financials?|retail|global|"
    r"leveraged finance|growth)\s+(?:[a-z&]+\s+){0,2}(?:conference|summit|symposium|forum)\b(?!\s+call)",
    r"\bto discuss (?:the |our |its |this )?(?:proposed |pending |announced |definitive |planned )?"
    r"(?:acquisition|merger|combination|transaction|agreement|spin[- ]?off|separation)\b",
    r"\b(?:acquisition|merger|combination) (?:announcement |conference )?call\b", r"\bjoint call\b",
]), re.I)
OTHER_EVENT = re.compile(f"{COMPANY_EVENT.pattern}|{EXTERNAL_EVENT.pattern}", re.I)
AMBIGUOUS_EVENT = re.compile(r"\bbusiness update\b|\binvestor (?:event|update|conference call)\b|\bstrategic update\b", re.I)
BODY_RESULTS = re.compile(r"\b(?:this|the|last|prior) quarter\b|\bquarterly\b|\bguidance\b|\bprepared remarks\b", re.I)
HOST = re.compile(
    r"(?:welcome (?:everyone |you |all )?to|welcome to|joining (?:us )?(?:today )?(?:for|on)|"
    r"facilitator today for|host(?:ing)? (?:today[''’]s )?(?:call for )?)\s+(?:the |today['’]s |this )?"
    r"(?:(?:Q[1-4]|first|second|third|fourth|fiscal|FY|20\d\d)[\w'’ ]{0,20}?\s+)??"
    r"(?P<name>[A-Z][\w&.,'’\- ]{1,60}?)(?:['’]s?)?\s+(?:\(?[A-Z]{1,5}\)?\s+)?"
    r"(?=(?:20\d\d|FY|fiscal|first|second|third|fourth|1st|2nd|3rd|4th|Q[1-4]|full|half|quarter|earnings|financial|"
    r"conference|investor|results|annual|business|analyst|capital|call|webcast|teleconference|trading)\b)",
    re.I)
NAME_STOPWORDS = r"\b(?:inc|incorporated|corp|corporation|co|company|companies|ltd|limited|plc|holdings?|group|the|llc|lp|nv|sa|ag|de|new|trust|intl|class [ab]|cl [ab]|n v|l p|s a)\b"
# a host name never contains these: the match ran into the speaker's sentence
NOT_A_COMPANY = {"i", "im", "me", "my", "we", "our", "us", "you", "your", "he", "she", "they", "with", "also", "begin",
                 "slide", "review", "turn", "happy", "reminder", "presenting", "during", "went", "jump", "hopefully",
                 "able", "was", "were", "is", "are", "am", "will", "would", "that", "this", "these", "here", "there",
                 "lead", "principal", "senior", "vice", "president", "director", "head", "officer", "chief",
                 "let", "virtual", "good", "hello", "okay", "great", "thanks", "thank", "everybody"}
PERIOD_TOKEN = re.compile(r"[1-4]q|q[1-4]|\d{2}|\d{4}|fy\d*|h[12]|[1-4](?:st|nd|rd|th)")
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
    """The company the intro welcomes to: the capitalised name before the
    call's description ("welcome to the Hilton Third Quarter ..."), with any
    leading filler dropped ("thank you for joining us today for the Duke
    Realty first quarter ...")."""
    for m in HOST.finditer(intro[:800]):
        words = m.group("name").strip(" ,.").split()
        while words and (norm(words[0]).strip() in HOST_GENERIC or PERIOD_TOKEN.fullmatch(norm(words[0]).strip())
                         or not words[0][:1].isupper()):
            words = words[1:]
        raw = " ".join(words)
        kept = [w for w in norm(raw).split() if w not in HOST_GENERIC and not PERIOD_TOKEN.fullmatch(w)]
        if (kept and len(max(kept, key=len)) >= 3 and norm(raw).replace(" ", "") not in HOST_GENERIC
                and not NOT_A_COMPANY.intersection(norm(raw).split())):
            return raw
    return None


def is_firm(name: str, aliases: set[str]) -> bool:
    """Whether a host name is one of the firm's names: an alias inside it, or
    the same letters as an alias up to a prefix or a typo (ExxonMobil, Lowe's
    Companies, Revitty)."""
    n = " " + " ".join(w for w in norm(name).split() if not PERIOD_TOKEN.fullmatch(w)) + " "  # NXP's 2Q '21
    squashed = re.sub(NAME_STOPWORDS, " ", n).replace(" ", "")
    alias_words = {w for a in aliases for w in a.split()}
    if squashed in {w for w in alias_words if len(w) >= 3} or squashed in {a.split()[0] for a in aliases}:  # Sands, Eli, M&T
        return True
    first = n.split()[0] if n.split() else ""
    # against the one-word forms the firm owns, not any word of its name:
    # "American Express" opens with a word of American Airlines
    if len(first) >= 4 and any((w.startswith(first) or first.startswith(w)) and abs(len(w) - len(first)) <= 2
                               for w in aliases if len(w) >= 4 and " " not in w):  # Lowe's
        return True
    for a in aliases:
        sa = a.replace(" ", "")
        if f" {a} " in n or squashed == sa or len(sa) >= 5 and squashed.startswith(sa) or len(squashed) >= 4 and sa.startswith(squashed):
            return True
        if len(sa) >= 5 and SequenceMatcher(None, squashed[:len(sa)], sa).ratio() >= 0.85:
            return True
        if len(squashed) <= 8 and SequenceMatcher(None, squashed, sa).ratio() >= 0.75:  # CCSX, Lenard
            return True
    if any(len(w) >= 5 and len(squashed) <= 8 and SequenceMatcher(None, squashed, w).ratio() >= 0.85
           for w in alias_words):  # SANS
        return True
    host_words = {w for w in n.split() if len(w) >= 5}
    return len(host_words & alias_words) >= 2  # Esso Green Realty Corp = SL Green Realty


def event_type(intro: str, head: str, release_gap: float) -> tuple[str, str | None]:
    near_release = release_gap <= RELEASE_DAYS
    other, results = OTHER_EVENT.search(intro), RESULTS.search(intro)
    if other and (not results or other.start() < results.start()) and not RESULTS.search(other.group(0)):
        # only the firm's own event can be the day's results event; a broker
        # conference or a fireside chat on that day is not
        company_event = COMPANY_EVENT.match(other.group(0)) is not None
        return ("release_day_event" if release_gap <= 1 and company_event else "other_event"), other.group(0)
    ambiguous = AMBIGUOUS_EVENT.search(intro)
    if ambiguous and not re.search(r"\bearnings\b|\bresults\b|\bquarter\b|\bQ[1-4]\b", intro[:ambiguous.end() + 80], re.I):
        return ("results" if near_release else "ambiguous_event"), ambiguous.group(0)
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


def firm_names() -> dict[str, list[str]]:
    """ticker -> every name of the firm: the firm-universe name and the EDGAR
    names (current and former) of its CIK and of its listing CIK
    (configs/us/listing_ciks.csv)."""
    universe = L.read("bronze.firm_universe").select("ticker", "cik", "company_name").to_pandas()
    edgar = L.read("bronze.sec_company_names").to_pandas().groupby("cik")["name"].apply(list).to_dict()
    listing = pd.read_csv(LISTING_CIKS, dtype=str).set_index("ticker")["cik"].to_dict()
    return {ticker: ([name] if isinstance(name, str) else []) + edgar.get(cik, []) + edgar.get(listing.get(ticker), [])
            for ticker, cik, name in zip(universe["ticker"], universe["cik"], universe["company_name"])}


def ticker_aliases() -> dict[str, str]:
    """Former vendor symbol -> the universe ticker of the same firm
    (configs/us/ticker_aliases.csv). A source files a firm's call under the
    symbol the firm traded as at the time — Fox Corporation's calls under the
    Class A symbol FOXA, Healthpeak's under PEAK — and that symbol is not in
    the universe, so the transcript would otherwise belong to no firm."""
    rows = pd.read_csv(TICKER_ALIASES)
    return dict(zip(rows["alias"], rows["ticker"]))


def company_names() -> dict[str, set[str]]:
    """Each firm's match forms, minus the fragments it does not own: "american"
    is a word of American Airlines and of American Express, so neither firm is
    recognised by it alone and both keep their two-word forms. A single word is
    kept when it is the firm's ticker, one of its whole names (Equifax,
    Southern), or no other panel firm answers to it."""
    aliases, whole = {}, {}
    for ticker, names in firm_names().items():
        forms = {ticker.lower().replace(".", " ").replace("-", " ")}
        aliases[ticker] = forms.union(*(name_aliases(n) for n in names))
        whole[ticker] = forms | {re.sub(r"\s+", " ", re.sub(NAME_STOPWORDS, " ", norm(n))).strip() for n in names}
    shared = Counter(a for forms in aliases.values() for a in forms if " " not in a)
    return {t: {a for a in forms if " " in a or shared[a] == 1 or a in whole[t]}
            for t, forms in aliases.items()}


def name_index() -> dict[str, set[str]]:
    """Name phrase -> tickers: each firm's cleaned names and their first two
    words, and a first word of at least five letters that no other firm's
    name contains (Oracle, not Realty); no tickers or initials."""
    cleaned = {t: [re.sub(r"\s+", " ", re.sub(NAME_STOPWORDS, " ", norm(n))).split() for n in names]
               for t, names in firm_names().items()}
    firms_with_word: dict[str, set[str]] = {}
    for ticker, names in cleaned.items():
        for words in names:
            for w in words:
                firms_with_word.setdefault(w, set()).add(ticker)
    index: dict[str, set[str]] = {}
    for ticker, names in cleaned.items():
        for words in names:
            first = words[0] if words and len(words[0]) >= 5 and len(firms_with_word[words[0]]) == 1 else ""
            for phrase in {" ".join(words), " ".join(words[:2]), first}:
                if len(phrase) >= 5:
                    index.setdefault(phrase, set()).add(ticker)
    return index


def named_firm(host: str, index: dict[str, set[str]]) -> tuple[str | None, int]:
    """(ticker, phrase length) of the firm with a name phrase equal to the
    host's name (its words outside legal suffixes and period tokens; squashed
    spelling too: ExxonMobil); None when no firm or several share it, or the
    host has words the phrase lacks (Westrock Coffee is not WestRock)."""
    words = [w for w in re.sub(NAME_STOPWORDS, " ", norm(host)).split()
             if not PERIOD_TOKEN.fullmatch(w) and w not in HOST_GENERIC]
    phrase = " ".join(words)
    for key in (phrase, SQUASHED.get(phrase.replace(" ", ""), "")):
        if key in index:
            return (next(iter(index[key])) if len(index[key]) == 1 else None), len(key)
    return None, 0


def enriched_documents() -> set[str]:
    """Call documents that can be measured: a prefilter prediction for every
    scorable paragraph and an LLM output for every prefiltered one."""
    paragraphs = (L.scan("bronze.paragraphs").filter(pl.col("is_scorable") & (pl.col("form") == "Earnings call"))
                  .select("accession_number", "text_hash").unique())
    scored = paragraphs.join(L.scan("bronze.prefilter_predictions").select("text_hash", "is_ai_prefiltered").unique("text_hash"),
                             on="text_hash", how="left")
    classified = L.scan("bronze.ai_frames").select("text_hash").unique().with_columns(pl.lit(True).alias("_classified"))
    docs = (scored.join(classified, on="text_hash", how="left")
            .group_by("accession_number")
            .agg(pl.col("is_ai_prefiltered").is_null().sum().alias("unscored"),
                 (pl.col("is_ai_prefiltered").fill_null(False) & pl.col("_classified").is_null()).sum().alias("unclassified"))
            .filter((pl.col("unscored") == 0) & (pl.col("unclassified") == 0))
            .collect())
    return set(docs["accession_number"])


def jaccard(sa: frozenset, sb: frozenset) -> float:
    """Bottom-k estimate of the Jaccard similarity of two shingle sketches:
    the share of the 256 smallest shingles of the union that both sketches
    hold. `heapq.nsmallest` keeps that prefix without sorting the union."""
    if not sa or not sb:
        return 0.0
    union = sa | sb
    prefix = union if len(union) <= 256 else set(heapq.nsmallest(256, union))
    return len(prefix & sa & sb) / len(prefix)


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

    for n, (_, g) in enumerate(calls.groupby("ticker"), start=1):
        if n % 100 == 0:
            log(f"  duplicate groups: {n} tickers")
        g = g.sort_values("call_date")
        rows = list(g.itertuples())
        sketches = {r.Index: frozenset(r.sketch) for r in rows}
        for a in range(len(rows)):
            for b in range(a + 1, len(rows)):
                ra, rb = rows[a], rows[b]
                gap = (rb.call_date - ra.call_date).days
                if gap > 200:
                    break
                both_stated = pd.notna(ra.stated_fiscal_period) and pd.notna(rb.stated_fiscal_period)
                # the date and fiscal-period tests are a dictionary lookup; the
                # text sketch only decides the pairs they leave open
                # positive evidence only: the same text, both intros naming the
                # same quarter, or the same day. Proximity alone groups two
                # different calls, and through the union it would chain a third
                same = (gap <= 1
                        or both_stated and ra.stated_fiscal_period == rb.stated_fiscal_period
                        or jaccard(sketches[ra.Index], sketches[rb.Index]) >= SIMILAR)
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
    releases = (L.scan("bronze.sec_filing_index").filter(pl.col("form").is_in(["8-K", "8-K/A"]) & pl.col("items").str.contains("2.02"))
                .select("cik", "filing_date").collect().to_pandas())

    # company: a transcript whose host names another universe firm more
    # specifically than its own ticker's names belongs to that firm
    aliases, index, symbol_aliases = company_names(), name_index(), ticker_aliases()
    SQUASHED.update({k.replace(" ", ""): k for k in index})
    log("  host company")
    calls["host_company"] = calls["head"].str[:INTRO_CHARS].map(host_company)
    calls["source_ticker"] = calls["ticker"]
    calls["ticker"] = calls["ticker"].map(lambda t: symbol_aliases.get(t, t))
    by_ticker: dict[str, dict[str, set[str]]] = {}
    for phrase, tickers in index.items():
        for t in tickers:
            by_ticker.setdefault(t, {})[phrase] = {t}
    for i, host in calls["host_company"].dropna().items():
        intro = calls.loc[i, "head"][:INTRO_CHARS]
        firm, length = named_firm(host, index)
        own = named_firm(host, by_ticker.get(calls.loc[i, "ticker"], {}))[1]
        if firm and firm != calls.loc[i, "ticker"] and length > own and RESULTS.search(intro) and not OTHER_EVENT.search(intro):
            calls.loc[i, "ticker"] = firm
    calls["cik"] = calls["ticker"].map(dict(L.read("bronze.firm_universe").select("ticker", "cik").iter_rows()))
    heads = calls["head"].map(norm)
    calls["company_named"] = [any(f" {a} " in h for a in aliases.get(t, ())) for t, h in zip(calls["ticker"], heads)]
    host_is_firm = [pd.isna(h) or is_firm(h, aliases.get(t, set())) for t, h in zip(calls["ticker"], calls["host_company"])]
    calls["other_company"] = ~pd.Series(host_is_firm, index=calls.index)

    # date and fiscal period
    calls["stated_date"] = calls["head"].map(stated_date)
    calls["stated_fiscal_period"] = calls["head"].str[:INTRO_CHARS].map(stated_period)
    # a stated date far from the metadata date is kept when the stated fiscal
    # period confirms its year (Equibles files the FY2024 Q1 call under 2023)
    stated_year = calls["stated_fiscal_period"].str[:4].astype(float)
    stated_ok = calls["stated_date"].notna() & (
        ((calls["stated_date"] - calls["metadata_date"]).dt.days.abs() <= STATED_DATE_MAX_GAP)
        | ((calls["stated_date"].dt.year - stated_year).abs() <= 1)
          & ((calls["metadata_date"].dt.year - stated_year).abs() > 1))
    calls["call_date"] = calls["stated_date"].where(stated_ok, calls["metadata_date"]).astype("datetime64[ns]")
    calls["call_date_source"] = np.where(stated_ok, "stated", "metadata")
    calls["fiscal_period"] = calls["stated_fiscal_period"].fillna(calls["metadata_fiscal_period"])
    calls["fiscal_period_source"] = np.where(calls["stated_fiscal_period"].notna(), "stated", "metadata")
    log("  release gaps")
    calls["release_gap_days"] = release_gaps(calls, releases, "call_date")

    # event type
    log("  event type")
    events = [event_type(h[:INTRO_CHARS], h, gap) for h, gap in zip(calls["head"], calls["release_gap_days"])]
    calls["event_type"] = [e[0] for e in events]
    calls["event_evidence"] = [e[1] for e in events]

    # an investor/analyst day held on the results release day is the quarter's
    # results event (Target's financial community meeting); a separate results
    # call that week wins the duplicate group over it
    calls["release_day_event"] = calls["event_type"] == "release_day_event"
    calls.loc[calls["release_day_event"], "event_type"] = "results"
    # a business update or investor event away from a release is the quarter's
    # results call when the ticker has no other results call within a quarter
    # (Schwab's quarterly business updates)
    # a business update is the quarter's results call when it sits on the firm's
    # 8-K Item 2.02 results release; the absence of another call is not evidence
    calls.loc[calls["event_type"] == "ambiguous_event", "event_type"] = np.where(
        calls.loc[calls["event_type"] == "ambiguous_event", "release_gap_days"] <= RELEASE_DAYS,
        "results", "other_event")
    log("  enrichment status")
    calls["pending"] = ~calls["document_id"].isin(enriched_documents())
    calls["exclude_reason"] = np.select([calls["event_type"] == "other_event", calls["other_company"]],
                                        ["other_event", "other_company"], None)
    calls["duplicate_of"] = None

    # duplicates: the kept transcript takes the group's fiscal period (named by
    # an intro) and best date (stated, then the source that labels the call
    # with that period, then the one closest to an 8-K Item 2.02 release)
    kept = calls[calls["exclude_reason"].isna()]
    log("  duplicate groups")
    groups = same_call_groups(kept)
    order = kept.sort_values(["pending", "release_day_event", "source_rank", "n_chars"], ascending=[True, True, True, False])
    representative = {}
    for i in order.index:
        representative.setdefault(groups[i], i)
    log("  representatives")
    members_by_root: dict[int, list[int]] = {}
    for i in kept.index:
        members_by_root.setdefault(groups[i], []).append(i)
    for root, rep in representative.items():
        members = members_by_root[root]
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

    # a call without scored paragraphs and LLM outputs cannot be measured yet
    calls.loc[calls["exclude_reason"].isna() & calls["pending"], "exclude_reason"] = "pending_enrichment"
    log("  sequence labels")
    calls["call_sequence"] = None
    for _, g in calls[calls["exclude_reason"].isna()].groupby("ticker"):
        label_sequence(calls, list(g.index))

    kept = calls[calls["exclude_reason"].isna()].sort_values(["ticker", "call_date"])
    previous = kept.groupby("ticker")[["document_id", "fiscal_period", "call_date"]].shift()
    exceptions = [{"ticker": r.ticker, "call_sequence": r.call_sequence, "from": p.fiscal_period, "to": r.fiscal_period,
                   "days": int((r.call_date - p.call_date).days)}
                  for (_, r), (_, p) in zip(kept.iterrows(), previous.iterrows()) if r.call_sequence in ("gap", "label_break")]
    out = calls.drop(columns=["head", "sketch", "local_path", "cik"])
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


def label_sequence(calls: pd.DataFrame, rows: list) -> None:
    """Label one ticker's kept transcripts in date order without altering them.

    `call_sequence` records what the data says: `first`, `next` (the fiscal
    period advances one quarter over about a quarter of calendar time), `gap`
    (it advances k > 1 quarters, so calls are missing from the sources) and
    `conflict` (the period does not advance, or the dates and the periods
    disagree). A conflict is reported, never repaired: a transcript's date and
    fiscal period come from what it states and from its source's metadata, and
    a calendar expectation is not evidence about either.
    """
    seq = sorted(rows, key=lambda i: (calls.at[i, "call_date"], calls.at[i, "source_rank"]))
    if not seq:
        return
    calls.at[seq[0], "call_sequence"] = "first"
    for a, b in zip(seq, seq[1:]):
        ok, k = sequence_step(calls.loc[a], calls.loc[b])
        calls.at[b, "call_sequence"] = "next" if ok and k == 1 else "gap" if ok else "conflict"


def sequence_step(a, b) -> tuple[bool, int]:
    """(invariant holds, quarters advanced) between consecutive kept calls."""
    days = (b["call_date"] - a["call_date"]).days
    k = quarter_index(b["fiscal_period"]) - quarter_index(a["fiscal_period"])
    return days >= MIN_GAP_DAYS and k >= 1 and 91 * k - SEQUENCE_EARLY <= days <= 91 * k + SEQUENCE_LATE, k


if __name__ == "__main__":
    main()
