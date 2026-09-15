"""
Build S&P 500 extended patent assignee aliases seed dataset.
Reads configs/us/universe.csv and expands each firm with:
1. Normalized SEC company name
2. Normalized legal stems (stripped of INC, CORP, CO, LLC, LTD, PLC, etc.)
3. Known patent-holding entities & key IP subsidiaries for tech, pharma, industrial, etc.
4. Google Patents standard harmonized abbreviations (TECHNOLOGIES -> TECH, etc.)
5. Regex pattern for flexible matching in BigQuery.
"""

from pathlib import Path
import re
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
UNIVERSE_CSV = REPO_ROOT / "configs" / "us" / "universe.csv"
OUT_CSV_REF = REPO_ROOT / "data" / "raw" / "reference" / "sp500_patent_aliases_seed.csv"
OUT_PARQUET_REF = REPO_ROOT / "data" / "raw" / "reference" / "sp500_patent_aliases_seed.parquet"
OUT_CSV_RAW = REPO_ROOT / "data" / "raw" / "sp500_patent_aliases_seed.csv"
OUT_LONG_PARQUET = REPO_ROOT / "data" / "raw" / "sp500_patent_assignees_long.parquet"

# Curated dictionary of major patent subsidiaries / patent-holding entities in Google Patents
KNOWN_PATENT_SUBSIDIARIES: dict[str, list[str]] = {
    "GOOGL": [
        "GOOGLE LLC", "GOOGLE INC", "ALPHABET INC", "WAYMO LLC",
        "DEEPMIND TECHNOLOGIES LTD", "DEEPMIND TECH LTD", "VERILY LIFE SCIENCES LLC"
    ],
    "AAPL": ["APPLE INC", "APPLE COMPUTER INC"],
    "MSFT": [
        "MICROSOFT CORP", "MICROSOFT TECHNOLOGY LICENSING LLC",
        "MICROSOFT TECH LICENSING LLC", "MICROSOFT LICENSING GP",
        "NUANCE COMMUNICATIONS INC", "LINKEDIN CORP",
        "ACTIVISION PUBLISHING INC", "BLIZZARD ENTERTAINMENT INC"
    ],
    "AMZN": [
        "AMAZON TECH INC", "AMAZON TECHNOLOGIES INC", "AMAZON COM INC",
        "AMAZON WEB SERVICES INC", "A9 COM INC", "ANNAPURNA LABS LTD", "ZOOX INC"
    ],
    "META": [
        "META PLATFORMS INC", "FACEBOOK INC", "INSTAGRAM LLC",
        "WHATSAPP LLC", "OCULUS VR LLC"
    ],
    "NVDA": ["NVIDIA CORP", "MELLANOX TECHNOLOGIES LTD", "MELLANOX TECH LTD"],
    "INTC": [
        "INTEL CORP", "INTEL CORPORATION", "MOBILEYE VISION TECHNOLOGIES LTD",
        "MOBILEYE VISION TECH LTD", "ALTERA CORP"
    ],
    "QCOM": ["QUALCOMM INC", "QUALCOMM INCORPORATED"],
    "AVGO": [
        "BROADCOM INC", "BROADCOM CORP", "AVAGO TECHNOLOGIES GENERAL IP PTE LTD",
        "AVAGO TECH GENERAL IP PTE LTD", "VMWARE LLC", "VMWARE INC", "CA INC", "SYMANTEC CORP"
    ],
    "CSCO": ["CISCO TECHNOLOGY INC", "CISCO TECH INC", "CISCO SYSTEMS INC"],
    "ORCL": [
        "ORACLE INTERNATIONAL CORP", "ORACLE INT CORP", "ORACLE AMERICA INC",
        # Not "CERNER INNOVATION INC": Oracle acquired Cerner in 2022, but CERN is
        # kept in the panel as a distinct delisted firm up to its exit date (same
        # convention as every other delisted firm here) and already claims Cerner's
        # patents via its own stem. Listing it here too double-counted the same
        # patents for both tickers (2026-09-11 patent alias audit).
    ],
    "IBM": ["INTERNATIONAL BUSINESS MACHINES CORP", "INT BUSINESS MACHINES CORP", "IBM CORP", "IBM"],
    "TSLA": ["TESLA INC", "TESLA MOTORS INC"],
    "GM": [
        "GENERAL MOTORS LLC", "GENERAL MOTORS CO",
        "GM GLOBAL TECHNOLOGY OPERATIONS LLC", "GM GLOBAL TECH OPERATIONS LLC", "CRUISE LLC"
    ],
    "F": ["FORD MOTOR CO", "FORD GLOBAL TECHNOLOGIES LLC", "FORD GLOBAL TECH LLC"],
    "BA": ["THE BOEING COMPANY", "BOEING CO"],
    "RTX": [
        "RTX CORP", "RAYTHEON CO", "UNITED TECHNOLOGIES CORP", "UNITED TECH CORP",
        "PRATT & WHITNEY", "COLLINS AEROSPACE", "RAYTHEON TECHNOLOGIES CORP"
    ],
    "LMT": ["LOCKHEED MARTIN CORP"],
    "GD": ["GENERAL DYNAMICS CORP", "GULFSTREAM AEROSPACE CORP"],
    "GE": [
        "GENERAL ELECTRIC CO", "GE HEALTHCARE", "GE AEROSPACE",
        "GE PRECISION HEALTHCARE LLC", "GEN ELECTRIC",
        "GENERAL ELECTRIC TECHNOLOGY GMBH", "GENERAL ELECTRIC RENOVABLES ESPANA SL"
    ],
    "HON": ["HONEYWELL INTERNATIONAL INC", "HONEYWELL INT INC"],
    "CAT": ["CATERPILLAR INC"],
    "DE": ["DEERE & CO", "DEERE AND COMPANY", "JOHN DEERE"],
    "AMAT": ["APPLIED MATERIALS INC"],
    "LRCX": ["LAM RESEARCH CORP", "LAM RES CORP"],
    "KLAC": ["KLA CORP", "KLA TENCOR CORP"],
    "SNPS": ["SYNOPSYS INC"],
    "CDNS": ["CADENCE DESIGN SYSTEMS INC"],
    "CRM": [
        "SALESFORCE INC", "SALESFORCE COM INC", "MULESOFT LLC",
        "TABLEAU SOFTWARE LLC"
    ],
    "ADBE": ["ADOBE INC", "ADOBE SYSTEMS INC"],
    "NOW": ["SERVICENOW INC"],
    "WDAY": ["WORKDAY INC"],
    "PANW": ["PALO ALTO NETWORKS INC"],
    "FTNT": ["FORTINET INC"],
    "CRWD": ["CROWDSTRIKE INC"],
    "DDOG": ["DATADOG INC"],
    "SNOW": ["SNOWFLAKE INC"],
    "PLTR": ["PALANTIR TECHNOLOGIES INC", "PALANTIR TECH INC"],
    "PYPL": ["PAYPAL INC"],
    "SQ": ["BLOCK INC", "SQUARE INC"],
    "V": ["VISA INC", "VISA INTERNATIONAL SERVICE ASSOCIATION", "VISA INT SERVICE ASSOCIATION"],
    "MA": ["MASTERCARD INTERNATIONAL INC", "MASTERCARD INT INC", "MASTERCARD INC"],
    "AXP": ["AMERICAN EXPRESS CO", "AMERICAN EXPRESS TRAVEL RELATED SERVICES CO INC"],
    "JPM": ["JPMORGAN CHASE BANK NA", "JP MORGAN CHASE & CO"],
    "BAC": ["BANK OF AMERICA CORP"],
    "C": ["CITIGROUP INC", "CITIBANK NA"],
    "WFC": ["WELLS FARGO & CO"],
    "GS": ["GOLDMAN SACHS GROUP INC", "GOLDMAN SACHS & CO LLC"],
    "MS": ["MORGAN STANLEY"],
    "BLK": ["BLACKROCK INC"],
    "COF": ["CAPITAL ONE FINANCIAL CORP", "CAPITAL ONE SERVICES LLC"],
    "DFS": ["DISCOVER FINANCIAL SERVICES"],
    "T": ["AT&T INTELLECTUAL PROPERTY I LP", "AT&T CORP", "AT & T IP I LP", "AT & T MOBILITY II LLC"],
    "VZ": ["VERIZON PATENT AND LICENSING INC", "VERIZON COMMUNICATIONS INC"],
    "CMCSA": ["COMCAST CABLE COMMUNICATIONS LLC"],
    "DIS": ["DISNEY ENTERPRISES INC", "DISNEY ENTPR INC"],
    "NFLX": ["NETFLIX INC"],
    "WMT": ["WALMART APOLLO LLC", "WAL MART STORES INC"],
    "TGT": ["TARGET BRANDS INC"],
    "HD": ["HOME DEPOT PRODUCT AUTHORITY LLC"],
    "LOW": ["LF LLC", "LOWE'S HOME CENTERS LLC"],
    "NKE": ["NIKE INC"],
    "SBUX": ["STARBUCKS CORP"],
    "MCD": ["MCDONALD'S CORP"],
    "PEP": ["PEPSICO INC"],
    "KO": ["THE COCA COLA COMPANY", "COCA COLA CO"],
    "PG": ["THE PROCTER & GAMBLE COMPANY", "PROCTER & GAMBLE CO"],
    "CL": ["COLGATE PALMOLIVE CO"],
    "MDLZ": ["MONDELEZ INTERNATIONAL INC", "KRAFT FOODS GLOBAL BRANDS LLC"],
    "XOM": [
        "EXXONMOBIL RESEARCH AND ENGINEERING CO", "EXXONMOBIL RESEARCH & ENG CO",
        "EXXONMOBIL UPSTREAM RESEARCH CO"
    ],
    "CVX": ["CHEVRON USA INC", "CHEVRON CORP"],
    "COP": ["CONOCOPHILLIPS CO"],
    "SLB": ["SCHLUMBERGER TECHNOLOGY CORP", "SCHLUMBERGER TECH CORP"],
    "EOG": ["EOG RESOURCES INC"],
    "OXY": ["OCCIDENTAL PETROLEUM CORP"],
    "JNJ": [
        "JOHNSON & JOHNSON", "JANSSEN PHARMACEUTICALS INC",
        "JANSSEN BIOTECH INC", "ETHICON INC", "DEPUY SYNTHES PRODUCTS INC",
        "BIOSENSE WEBSTER INC", "ETHICON ENDO SURGERY INC"
    ],
    "PFE": ["PFIZER INC", "ARRAY BIOPHARMA INC", "SEAGEN INC"],
    "MRK": ["MERCK SHARP & DOHME CORP", "MERCK SHARP & DOHME LLC"],
    "ABBV": ["ABBVIE INC", "PHARMACYCLICS LLC", "ALLERGAN INC"],
    "BMY": ["BRISTOL MYERS SQUIBB CO", "CELGENE CORP"],
    "LLY": ["ELI LILLY AND CO", "ELI LILLY & CO", "LILLY CO ELI"],
    "GILD": ["GILEAD SCIENCES INC", "KITE PHARMA INC"],
    "AMGN": ["AMGEN INC"],
    "BIIB": ["BIOGEN INC"],
    "REGN": ["REGENERON PHARMACEUTICALS INC"],
    "VRTX": ["VERTEX PHARMACEUTICALS INC"],
    "ISRG": ["INTUITIVE SURGICAL OPERATIONS INC", "INTUITIVE SURGICAL INC"],
    "MDT": ["MEDTRONIC INC"],
    "ABT": ["ABBOTT LABORATORIES"],
    "BSX": ["BOSTON SCIENTIFIC SCIMED INC", "BOSTON SCIENTIFIC CORP"],
    "SYK": ["STRYKER CORP"],
    "BDX": ["BECTON DICKINSON AND CO", "BECTON DICKINSON & CO"],
    "EW": ["EDWARDS LIFESCIENCES CORP"],
    "DXCM": ["DEXCOM INC"],
    "IDXX": ["IDEXX LABORATORIES INC"],
    "IQV": ["IQVIA INC"],
    "MTD": ["METTLER TOLEDO"],
    "A": ["AGILENT TECHNOLOGIES INC", "AGILENT TECH INC"],
    "TMO": ["THERMO FISHER SCIENTIFIC INC", "LIFE TECHNOLOGIES CORP", "LIFE TECH CORP"],
    "DHR": ["DANAHER CORP", "BECKMAN COULTER INC"],
    "TXN": ["TEXAS INSTRUMENTS INC"],
    "ADI": ["ANALOG DEVICES INC"],
    "MU": ["MICRON TECHNOLOGY INC", "MICRON TECH INC"],
    "NXPI": ["NXP BV", "NXP USA INC", "NXP SEMICONDUCTORS"],
    "MCHP": ["MICROCHIP TECHNOLOGY INC", "MICROCHIP TECH INC"],
    "MRVL": ["MARVELL ASIA PTE LTD", "MARVELL WORLDWIDE LTD", "MARVELL TECHNOLOGY INC", "MARVELL TECH INC"],
    "MPWR": ["MONOLITHIC POWER SYSTEMS INC"],
    "ON": ["ON SEMICONDUCTOR CORP", "SEMICONDUCTOR COMPONENTS INDUSTRIES LLC"],
    "SWKS": ["SKYWORKS SOLUTIONS INC"],
    "QRVO": ["QORVO US INC"],
    "KEYS": ["KEYSIGHT TECHNOLOGIES INC", "KEYSIGHT TECH INC"],
    "TER": ["TERADYNE INC"],
    # Not "SYMANTEC CORP": Broadcom (AVGO) acquired Symantec's Enterprise Security
    # business in 2019 (the larger, more patent-heavy segment); NortonLifeLock/Gen
    # Digital kept the smaller consumer antivirus business. Symantec's patent
    # portfolio was built as an enterprise security vendor, so it is attributed to
    # AVGO only rather than split on no direct evidence (2026-09-11 patent alias
    # audit; previously listed on both tickers, double-counting the same patents).
    "GEN": ["GEN DIGITAL INC", "NORTONLIFELOCK INC"],
    "AKAM": ["AKAMAI TECHNOLOGIES INC", "AKAMAI TECH INC"],
    "FFIV": ["F5 INC", "F5 NETWORKS INC"],
    "JNPR": ["JUNIPER NETWORKS INC"],
    "NTAP": ["NETAPP INC"],
    "WDC": ["WESTERN DIGITAL TECHNOLOGIES INC", "WESTERN DIGITAL TECH INC"],
    "STX": ["SEAGATE TECHNOLOGY LLC", "SEAGATE TECH LLC"],
    "HPQ": ["HP INC", "HEWLETT PACKARD DEVELOPMENT CO LP"],
    "HPE": ["HEWLETT PACKARD ENTERPRISE DEVELOPMENT LP", "HEWLETT PACKARD ENTPR DEV LP"],
    "CTSH": ["COGNIZANT TECHNOLOGY SOLUTIONS US CORP", "COGNIZANT TECH SOLUTIONS US CORP"],
    "EPAM": ["EPAM SYSTEMS INC"],
    "IT": ["GARTNER INC"],
    "ANSS": ["ANSYS INC"],
    "PTC": ["PTC INC"],
    "TYL": ["TYLER TECHNOLOGIES INC", "TYLER TECH INC"],
    "VRSN": ["VERISIGN INC"],
    "ROP": ["ROPER TECHNOLOGIES INC", "ROPER TECH INC"],
    "TTWO": ["TAKE TWO INTERACTIVE SOFTWARE INC"],
    "EA": ["ELECTRONIC ARTS INC"],
    # Former corporate names and abbreviated legal-entity spellings confirmed against
    # Google Patents' assignee_harmonized field directly (2026-09-11 patent alias audit):
    # the generic stem+suffix generator below never produces these because they are not
    # derived from the CURRENT SEC company name (TE Connectivity was "Tyco Electronics"
    # until 2011) or use a Google Patents abbreviation the generator does not model
    # (ENTERPRISES -> ENTPR, RESEARCH -> RES, word-order changes like "Lilly Co Eli").
    "TEL": [
        "TYCO ELECTRONICS CORP", "TYCO ELECTRONICS SHANGHAI CO LTD",
        "TYCO ELECTRONICS AMP GMBH", "TYCO ELECTRONICS JAPAN G K", "TYCO ELECTRONICS AMP KK",
        "TYCO ELECTRONICS LTD UK", "TE CONNECTIVITY SERVICES GMBH",
        "TYCO ELECTRONICS RAYCHEM BVBA", "TYCO ELECTRONICS LOGISTICS AG",
        "TYCO ELECTRONICS RAYCHEM NV", "TYCO ELECTRONICS RAYCHEM GMBH",
        "TYCO ELECTRONICS FRANCE SAS", "TYCO ELECTRONICS JAPAN KK",
        "TYCO ELECTRONICS AMP KOREA LTD", "TYCO ELECTRONICS NEDERLAND BV",
        "TYCO ELECTRONICS SERVICES GMBH", "TYCO ELECTRONICS AMP KOREA CO LTD",
        "TE CONNECTIVITY NEDERLAND BV", "TE CONNECTIVITY INDIA PRIVATE LTD",
        "TE CONNECTIVITY SUZHOU IND PARK CO LTD", "TYCO ELECTRONICS AUSTRIA GMBH",
        "TYCO ELECTRONICS CANADA ULC", "TYCO ELECTRONICS SUBSEA COMM",
        "TYCO ELECTRONICS SUZHOU LTD", "TYCO ELECTRONICS RAYCHEM KK",
        "TYCO ELECTRONICS BELGIUM EC BV", "TYCO ELECTRONICS TECHNOLOGY SIP CO LTD",
        "TE CONNECTIVITY GERMANY GMBH", "TE CONNECTIVITY SOLUTIONS GMBH",
    ],
    "MMM": ["3M INNOVATIVE PROPERTIES CO", "3M INNOVATIVE PROPERTIES COMPANY"],
    "DOW": ["DOW GLOBAL TECHNOLOGIES LLC", "DOW GLOBAL TECHNOLOGIES INC"],
    "OTIS": ["OTIS ELEVATOR CO", "OTIS ELEVATOR COMPANY"],
    "TT": ["TRANE INT INC"],
    "WAB": ["WESTINGHOUSE AIR BRAKE TECH CORP"],
    "MRNA": ["MODERNATX INC"],
}

# Bare company stems that are also common English words/phrases: matching them as a
# free-standing regex alternative (rather than only with a corporate suffix attached)
# produces false positives once the join uses REGEXP_CONTAINS instead of an exact match
# -- e.g. ticker SO's stem "SOUTHERN" matches inside "SOUTHERN MEDICAL UNIVERSITY" or
# "CHINA SOUTHERN POWER GRID", entities unrelated to Southern Company. Confirmed against
# the actual top-5000 assignee_harmonized names by patent count (2026-09-11 patent alias
# audit) -- no other ticker's bare stem produced this collision in that sample, but a
# newly added ticker whose bare stem is a common word should be checked the same way
# before being trusted as a free-standing regex alternative.
RISKY_BARE_STEMS = {"SO"}

LEGAL_SUFFIXES_RE = re.compile(
    r'\b(INC|INCORPORATED|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED|LLC|PLC|SA|AG|NV|GMBH|LP|L\.P\.|HOLDINGS|HOLDING|GROUP|GLOBAL|USA|US|DELAWARE|DE|MO|NEW|THE)\b',
    re.IGNORECASE
)

def clean_company_name(name: str) -> str:
    name = re.sub(r'/[A-Z0-9]+/?', ' ', name)
    name = re.sub(r'[^\w\s&]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip().upper()
    return name

def get_company_stem(cleaned_name: str) -> str:
    stem = LEGAL_SUFFIXES_RE.sub(' ', cleaned_name)
    stem = re.sub(r'\s+', ' ', stem).strip()
    return stem

def expand_abbreviations(alias: str) -> list[str]:
    results = [alias]
    if "TECHNOLOGIES" in alias:
        results.append(alias.replace("TECHNOLOGIES", "TECH"))
    if "TECHNOLOGY" in alias:
        results.append(alias.replace("TECHNOLOGY", "TECH"))
    if "INTERNATIONAL" in alias:
        results.append(alias.replace("INTERNATIONAL", "INT"))
    if "CORPORATION" in alias:
        results.append(alias.replace("CORPORATION", "CORP"))
    if "COMPANY" in alias:
        results.append(alias.replace("COMPANY", "CO"))
    if "INCORPORATED" in alias:
        results.append(alias.replace("INCORPORATED", "INC"))
    return list(set(results))

def build_aliases():
    df = pd.read_csv(UNIVERSE_CSV, dtype={"cik": str})
    print(f"Loaded {len(df)} firms from {UNIVERSE_CSV}")

    records_wide = []
    records_long = []

    for _, row in df.iterrows():
        ticker = str(row["ticker"]).strip()
        cik = str(row["cik"]).zfill(10)
        raw_name = str(row["company_name"]).strip()
        active_status = str(row.get("active_status", "listed"))

        cleaned_name = clean_company_name(raw_name)
        stem = get_company_stem(cleaned_name)

        alias_set = set()
        alias_set.add(cleaned_name)
        if stem and len(stem) >= 3:
            alias_set.add(stem)
            alias_set.add(f"{stem} INC")
            alias_set.add(f"{stem} CORP")
            alias_set.add(f"{stem} CO")
            alias_set.add(f"{stem} LLC")
            alias_set.add(f"{stem} TECHNOLOGIES INC")
            alias_set.add(f"{stem} TECH INC")

        known = KNOWN_PATENT_SUBSIDIARIES.get(ticker, [])
        for k in known:
            k_clean = clean_company_name(k)
            alias_set.add(k_clean)

        # Expand Google Patents standard abbreviations across all aliases
        final_aliases = set()
        for a in alias_set:
            for exp in expand_abbreviations(a):
                final_aliases.add(exp)

        regex_parts = []
        if stem and len(stem) >= 4 and stem not in ["AND", "NEW", "THE", "ALL"] and ticker not in RISKY_BARE_STEMS:
            regex_parts.append(re.escape(stem))
        for a in final_aliases:
            if ticker in RISKY_BARE_STEMS and a == stem:
                continue  # bare stem is a common word for this ticker; suffixed forms only
            regex_parts.append(re.escape(a))

        assignee_regex = r'\b(' + '|'.join(sorted(set(regex_parts), key=len, reverse=True)) + r')\b'

        for a in sorted(final_aliases):
            records_long.append({
                "ticker": ticker,
                "cik": cik,
                "company_name": raw_name,
                "assignee_alias": a,
                "is_known_subsidiary": a in [clean_company_name(x) for x in known],
                "active_status": active_status
            })

        records_wide.append({
            "ticker": ticker,
            "cik": cik,
            "company_name": raw_name,
            "cleaned_name": cleaned_name,
            "company_stem": stem,
            "num_aliases": len(final_aliases),
            "aliases_list": ";".join(sorted(final_aliases)),
            "assignee_regex": assignee_regex,
            "active_status": active_status
        })

    df_wide = pd.DataFrame(records_wide)
    df_long = pd.DataFrame(records_long)

    OUT_CSV_REF.parent.mkdir(parents=True, exist_ok=True)
    df_wide.to_csv(OUT_CSV_REF, index=False)
    df_wide.to_parquet(OUT_PARQUET_REF, index=False)
    df_wide.to_csv(OUT_CSV_RAW, index=False)
    df_long.to_parquet(OUT_LONG_PARQUET, index=False)
    df_long.to_csv(REPO_ROOT / "data" / "raw" / "sp500_patent_assignees_long.csv", index=False)

    print(f"Generated {len(df_wide)} wide records saved to {OUT_CSV_REF} and {OUT_CSV_RAW}")
    print(f"Generated {len(df_long)} long assignee records saved to {OUT_LONG_PARQUET}")

if __name__ == "__main__":
    build_aliases()
