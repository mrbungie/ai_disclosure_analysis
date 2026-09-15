"""
Build BigQuery SQL and metadata for OECD 2025 AI Patent Methodology.
Contains:
- 5 Core AI CPC groups
- 95 AI-Related CPC groups
- 274 OECD AI keywords compiled into BigQuery REGEXP patterns.
"""

from pathlib import Path
import re
import fitz

REPO_ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = REPO_ROOT / ".tmp_oecd" / "oecd_ai_patents_2025.pdf"

CORE_CPC = ["G06N3", "G06N5", "G06N7", "G06N20", "G06F18"]

doc = fitz.open(PDF_PATH)

# Extract 95 CPC groups from Page 30
p30 = doc[29]
cpcs = re.findall(r'\b[A-Z]\d{2}[A-Z]\d+\b', p30.get_text())
RELATED_CPC = sorted(list(set(c for c in cpcs if c not in CORE_CPC)))
assert len(RELATED_CPC) == 95, f"Expected 95 related CPCs, found {len(RELATED_CPC)}"

# Extract 274 keywords from Pages 28-29
keywords_raw = []
for p_num in [27, 28]:
    tabs = doc[p_num].find_tables()
    for t in tabs.tables:
        for row in t.extract():
            for cell in row:
                if cell:
                    cleaned = " ".join(cell.split()).strip()
                    if cleaned:
                        keywords_raw.append(cleaned)

assert len(keywords_raw) == 274, f"Expected 274 keywords, found {len(keywords_raw)}"

# Process keywords for BigQuery REGEXP
regex_patterns = []
for k in keywords_raw:
    k_clean = k.lower().strip()
    if "(…)" in k_clean:
        # e.g. "attention (…) transformer" -> attention.{1,30}transformer
        parts = [re.escape(p.strip()) for p in k_clean.split("(…)") if p.strip()]
        pat = r'\b' + r'.{1,30}'.join(parts) + r'\b'
    else:
        # Escape any special characters, replace spaces with \s+
        words = k_clean.split()
        pat = r'\b' + r'\s+'.join(re.escape(w) for w in words) + r'\b'
    regex_patterns.append(pat)

print(f"Total compiled keyword regex patterns: {len(regex_patterns)}")

# Split into chunks of ~30 patterns to keep regex efficient and well within BigQuery limits
chunk_size = 30
chunks = [regex_patterns[i:i + chunk_size] for i in range(0, len(regex_patterns), chunk_size)]
print(f"Divided into {len(chunks)} regex groups for BigQuery REGEXP_CONTAINS.")

# Print SQL snippet for testing
print("Sample chunk 0 pattern length:", len("|".join(chunks[0])))
