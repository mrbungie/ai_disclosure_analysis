"""
scripts/us/05_fetch_xbrl_frames_alt.py — SEC XBRL "frames" (cross-sectional,
calendar-quarter) pull for the core panel metrics, as an ALTERNATIVE source
to the per-filing inline-XBRL extraction in
data/raw/xbrl_facts/us_by_filing/ (scripts/us/04_extract_inline_xbrl_facts.py).

Frames (data.sec.gov/api/xbrl/frames) return, for one XBRL tag and one
calendar period, every filer's value for that exact period across the whole
SEC universe — the opposite shape from the per-filing extractor, which reads
each firm's own fiscal-quarter facts filing by filing. Measured in-session
(2026-09-08) against the 517-ticker/22-quarter (2021Q1-2026Q2) 10-Q panel:
frames alone covers WORSE than the inline-XBRL extraction for most duration
metrics (revenue 34.5% vs 91.8%, capex 19.1% vs 81.2%) because frames only
returns a value when a filer's fiscal period exactly matches the calendar
quarter boundary — it silently drops every non-December fiscal-year-end
filer for a given quarter. Used as a fallback/union source instead, it adds
only 0.1-1.0pp of coverage per metric. Kept as a documented alternative
(data/raw/xbrl_frames_alt/) rather than a replacement — see
docs/sources/accounting_data.md for the full comparison.

Usage:
    uv run python scripts/us/05_fetch_xbrl_frames_alt.py
"""

import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))  # scripts/common/
import pipeline_logger

# (metric, [XBRL tags in fallback priority order], period_type)
METRICS = [
    ("revenue", ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                 "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet",
                 "SalesRevenueGoodsNet"], "duration"),
    ("cogs", ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfServices"], "duration"),
    ("rd_expense", ["ResearchAndDevelopmentExpense"], "duration"),
    ("sga_expense", ["SellingGeneralAndAdministrativeExpense", "GeneralAndAdministrativeExpense"], "duration"),
    ("operating_income", ["OperatingIncomeLoss"], "duration"),
    ("net_income", ["NetIncomeLoss", "ProfitLoss"], "duration"),
    ("eps_diluted", ["EarningsPerShareDiluted"], "duration"),
    ("capex", ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"], "duration"),
    ("assets", ["Assets"], "instant"),
    ("current_assets", ["AssetsCurrent"], "instant"),
    ("current_liabilities", ["LiabilitiesCurrent"], "instant"),
    ("equity", ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"], "instant"),
    ("debt", ["LongTermDebtNoncurrent", "LongTermDebt"], "instant"),
]


def quarters_in(filing_date_from: str, filing_date_to: str):
    y0, y1 = int(filing_date_from[:4]), int(filing_date_to[:4])
    q1 = (int(filing_date_to[5:7]) - 1) // 3 + 1
    out = []
    for y in range(y0, y1 + 1):
        for q in range(1, 5):
            if y == y1 and q > q1:
                break
            out.append((y, q))
    return out


def fetch_frame(session, tag, year, q, instant, sleep_s):
    unit = "USD-per-shares" if tag == "EarningsPerShareDiluted" else "USD"
    period = f"CY{year}Q{q}I" if instant else f"CY{year}Q{q}"
    url = f"https://data.sec.gov/api/xbrl/frames/us-gaap/{tag}/{unit}/{period}.json"
    r = session.get(url, timeout=30)
    time.sleep(sleep_s)
    if r.status_code != 200:
        return None
    return r.json()


def main():
    with open("configs/us/config.yaml") as f:
        config = yaml.safe_load(f)

    universe = pd.read_csv("configs/us/universe.csv", dtype={"cik": str})
    universe["cik_int"] = universe["cik"].astype(int)
    cik_to_ticker = dict(zip(universe["cik_int"], universe["ticker"]))

    quarters = quarters_in(config["corpus"]["filings_10q"]["filing_date"]["from"],
                            config["corpus"]["filings_10q"]["filing_date"]["to"])

    out_dir = Path(config["storage"].get("raw_xbrl_frames_alt", "data/raw/xbrl_frames_alt"))
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = Path(config["storage"]["interim_manifests"])

    session = requests.Session()
    session.headers.update({"User-Agent": config["sec"]["user_agent"]})

    records = []
    for metric, tags, ptype in METRICS:
        instant = ptype == "instant"
        t0 = time.monotonic()
        metric_records = 0
        for (y, q) in quarters:
            for tag in tags:
                data = fetch_frame(session, tag, y, q, instant, sleep_s=0.11)
                if data and data.get("data"):
                    for row in data["data"]:
                        ticker = cik_to_ticker.get(row.get("cik"))
                        if ticker is None:
                            continue
                        records.append({
                            "ticker": ticker, "cik": row.get("cik"), "metric": metric, "tag_used": tag,
                            "year": y, "quarter": q, "value": row.get("val"),
                            "period_end": row.get("end"), "period_start": row.get("start"),
                            "accession_number": row.get("accn"), "form": row.get("form"),
                        })
                        metric_records += 1
                    break  # first tag with data wins for this quarter
        pipeline_logger.log_event(
            pipeline_step="us_xbrl_frames_alt", level="SUCCESS",
            message=f"Fetched {metric_records} frame rows for {metric}", details={"metric": metric},
            duration_seconds=time.monotonic() - t0, log_dir=manifest_dir,
        )

    df = pd.DataFrame.from_records(records)
    out_path = out_dir / "us_10q_frames.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Done. {len(df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
