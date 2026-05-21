import json
from datetime import datetime
from pathlib import Path

def get_log_path():
    """
    Dynamically locate the log path from configs/config.json.
    Falls back to data/interim/manifests if config cannot be loaded.
    """
    try:
        config_path = Path("configs/config.json")
        if not config_path.exists():
            # Try parent directory in case script is run from scripts/
            config_path = Path("../configs/config.json")
        
        with open(config_path, "r") as f:
            config = json.load(f)
        log_dir = Path(config["paths"]["interim_manifests"])
    except Exception:
        log_dir = Path("data/interim/manifests")
        
    return log_dir / "pipeline_log.jsonl"

def log_event(pipeline_step, level, message, ticker=None, cik=None, accession_number=None, duration_seconds=None, details=None):
    """
    Centralized logging function. Writes structured logs in JSONL format,
    which can be natively and efficiently queried via DuckDB.
    """
    log_path = get_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "pipeline_step": pipeline_step,
        "level": level.upper(),
        "message": message,
        "ticker": ticker,
        "cik": str(cik) if cik is not None else None,
        "accession_number": accession_number,
        "duration_seconds": float(duration_seconds) if duration_seconds is not None else None,
        "details": details if isinstance(details, dict) else {}
    }
    
    # Generate a clean console print
    console_msg = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{pipeline_step.upper()}] [{level.upper()}] {message}"
    if ticker:
        console_msg += f" (Ticker: {ticker})"
    print(console_msg)
    
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"Failed to write to pipeline log file: {e}")
