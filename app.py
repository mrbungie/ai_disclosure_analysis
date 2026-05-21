import os
import json
import sys
import re
import subprocess
import requests
import pandas as pd
import streamlit as st
from pathlib import Path
from datetime import datetime
import hashlib

# Page layout and aesthetics
st.set_page_config(
    page_title="SEC AI Disclosure Pipeline",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF4B4B, #FF8F8F);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f9f9f9;
        border: 1px solid #eee;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)

# Configuration helper functions
CONFIG_PATH = Path("configs/config.json")

@st.cache_data
def get_sec_tickers(user_agent="German Oviedo german.oviedo.b@gmail.com"):
    cache_file = Path("data/sec_company_tickers.json")
    if cache_file.exists():
        try:
            with open(cache_file, "r") as f:
                return json.load(f)
        except Exception:
            pass
            
    # If not cached, fetch it
    headers = {"User-Agent": user_agent}
    url = "https://www.sec.gov/files/company_tickers.json"
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w") as f:
            json.dump(data, f)
        return data
    except Exception as e:
        # Fallback to static subset if offline or blocked
        return {
            "0": {"cik_str": 789019, "ticker": "MSFT", "title": "MICROSOFT CORP"},
            "1": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
            "2": {"cik_str": 320193, "ticker": "AAPL", "title": "APPLE INC"},
            "3": {"cik_str": 1652044, "ticker": "GOOGL", "title": "Alphabet Inc."},
            "4": {"cik_str": 1318605, "ticker": "TSLA", "title": "TESLA, INC."},
            "5": {"cik_str": 1018724, "ticker": "AMZN", "title": "AMAZON COM INC"},
            "6": {"cik_str": 1326801, "ticker": "META", "title": "Meta Platforms, Inc."},
            "7": {"cik_str": 1108524, "ticker": "CRM", "title": "Salesforce, Inc."},
            "8": {"cik_str": 796343, "ticker": "ADBE", "title": "ADOBE INC."},
            "9": {"cik_str": 93410, "ticker": "CSCO", "title": "CISCO SYSTEMS, INC."},
            "10": {"cik_str": 1065280, "ticker": "NFLX", "title": "NETFLIX INC"},
            "11": {"cik_str": 939211, "ticker": "ORCL", "title": "ORACLE CORP"},
            "12": {"cik_str": 2487, "ticker": "AMD", "title": "ADVANCED MICRO DEVICES INC"},
            "13": {"cik_str": 804328, "ticker": "QCOM", "title": "QUALCOMM INC/DE"},
            "14": {"cik_str": 311094, "ticker": "INTU", "title": "INTUIT INC"},
            "15": {"cik_str": 50863, "ticker": "INTC", "title": "INTEL CORP"},
            "16": {"cik_str": 51143, "ticker": "IBM", "title": "INTERNATIONAL BUSINESS MACHINES CORP"},
            "17": {"cik_str": 1640147, "ticker": "SNOW", "title": "Snowflake Inc."},
            "18": {"cik_str": 1321655, "ticker": "PLTR", "title": "Palantir Technologies Inc."},
            "19": {"cik_str": 1577552, "ticker": "PANW", "title": "Palo Alto Networks Inc."}
        }


def load_config():
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {
        "sec": {"user_agent": "German Oviedo german.oviedo.b@gmail.com"},
        "pipeline": {"start_year": 2021, "end_year": 2025, "form_types": ["10-K"], "tickers": []},
        "paths": {
            "raw_submissions": "data/raw/sec_submissions",
            "raw_html": "data/raw/filings_html",
            "interim_manifests": "data/interim/manifests",
            "interim_batches": "data/interim/batches",
            "interim_sections": "data/interim/sections",
            "candidate_chunks": "data/interim/candidate_chunks"
        },
        "prefiltering": {
            "ai_keywords": ["artificial intelligence", "generative ai", "gen ai", "machine learning", "large language model", "llm", "deep learning", "natural language processing", "predictive analytics", "algorithmic", "automation", "computer vision", "neural network"],
            "false_positives": ["adobe illustrator", "appreciation", "said", "paid"]
        }
    }

def save_config(config):
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)

def run_script(script_name, args=None):
    # Run the script using the current virtual environment's python interpreter
    python_bin = sys.executable
    cmd = [python_bin, f"scripts/{script_name}"]
    if args:
        cmd.extend(args)
    
    with st.spinner(f"Running scripts/{script_name} {' '.join(args) if args else ''}..."):
        result = subprocess.run(cmd, capture_output=True, text=True)
        
    if result.returncode == 0:
        st.success(f"Successfully completed scripts/{script_name}")
        with st.expander("Show execution log"):
            st.code(result.stdout)
        return True, result.stdout
    else:
        st.error(f"Error running scripts/{script_name} (Exit code: {result.returncode})")
        with st.expander("Show error log"):
            st.code(result.stderr + "\n" + result.stdout)
        return False, result.stderr

# Main app title
st.markdown("<div class='main-header'>🤖 SEC Corporate AI Disclosure Pipeline</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Interactive research workbench to discover, download, and extract AI narrative sections from SEC filings.</div>", unsafe_allow_html=True)

# Load configuration
config = load_config()

# Initialize session state for active tickers if not already set
if "active_tickers" not in st.session_state:
    st.session_state.active_tickers = [t.upper() for t in config["pipeline"]["tickers"]]

# Sync configuration tickers list to the active state list
config["pipeline"]["tickers"] = st.session_state.active_tickers

# Sidebar: Read-only Status & Metrics
st.sidebar.header("📊 Pipeline Status")

# Compute metrics
cache_dir = Path(config["paths"]["raw_submissions"])
cache_count = len(list(cache_dir.glob("*.json"))) if cache_dir.exists() else 0

active_universe_size = len(st.session_state.active_tickers)
date_scope = f"{config['pipeline']['start_year']} - {config['pipeline']['end_year']}"
form_types_str = ", ".join(config["pipeline"]["form_types"])

st.sidebar.metric("Active Universe Size", f"{active_universe_size} Firms")
st.sidebar.metric("Filing Scope Years", date_scope)
st.sidebar.metric("Target Form Types", form_types_str)
st.sidebar.metric("Submissions Cache Count", f"{cache_count} CIKs")

# Load Manifests and Chunks for visual layout
universe_path = Path(config["paths"]["interim_manifests"]) / "firm_universe.parquet"
manifest_path = Path(config["paths"]["interim_manifests"]) / "filing_manifest.parquet"
chunks_path = Path(config["paths"]["candidate_chunks"]) / "ai_candidate_chunks.parquet"

# Main Application Tabs
tab_config, tab_universe, tab1, tab2, tab_logs, tab3 = st.tabs([
    "⚙️ Configuration",
    "🏢 Universe Management",
    "📂 Discovery & Manifests",
    "🧩 Candidate Chunks",
    "📋 Pipeline Logs",
    "📘 Project Architecture"
])

with tab_config:
    st.subheader("⚙️ Pipeline & SEC Configuration")
    st.markdown("Configure your SEC User-Agent credentials, search scope, and target form types. SEC requests must declare a valid User-Agent to comply with rate limits.")

    col_sec, col_scope = st.columns(2)
    
    with col_sec:
        st.markdown("### 🔒 SEC User-Agent Credentials")
        sec_email = st.text_input(
            "SEC User-Agent Email",
            value=config["sec"]["user_agent"].split()[-1] if config["sec"]["user_agent"] else "german.oviedo.b@gmail.com",
            help="Email address sent to SEC EDGAR in the User-Agent header."
        )
        sec_name = st.text_input(
            "SEC User-Agent Name",
            value=" ".join(config["sec"]["user_agent"].split()[:-1]) if config["sec"]["user_agent"] else "German Oviedo",
            help="Your name or organization sent in the User-Agent header."
        )
        config["sec"]["user_agent"] = f"{sec_name} {sec_email}"
        
        st.info("💡 **SEC policy** requires that your User-Agent header includes a valid name and email address. Failing to do so can result in immediate blocking.")

    with col_scope:
        st.markdown("### 📅 Scope & Forms")
        start_year = st.number_input("Start Year", min_value=2015, max_value=2030, value=config["pipeline"]["start_year"])
        end_year = st.number_input("End Year", min_value=2015, max_value=2030, value=config["pipeline"]["end_year"])
        config["pipeline"]["start_year"] = int(start_year)
        config["pipeline"]["end_year"] = int(end_year)
        
        form_types = st.multiselect("Form Types", ["10-K", "10-Q"], default=config["pipeline"]["form_types"])
        config["pipeline"]["form_types"] = form_types

    st.markdown("---")
    st.markdown("### ⚡ Actions")
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("💾 Save Configuration Only", type="secondary", use_container_width=True, key="save_config_only_btn"):
            save_config(config)
            st.toast("Configuration saved!", icon="💾")
            st.rerun()
            
    with col_btn2:
        if st.button("🔄 Save & Run Manifest Update (Incremental)", type="secondary", use_container_width=True, key="save_run_incremental_btn", help="Update the firm universe and pull any missing filing metadata using cached SEC files when possible."):
            save_config(config)
            st.toast("Configuration saved!", icon="💾")
            success, _ = run_script("00_build_firm_universe.py")
            if success:
                st.toast("Firm universe updated successfully!", icon="✅")
                with st.status("Fetching available filings from SEC EDGAR (incremental)...", expanded=True) as status:
                    st.write("Querying SEC filing indexes...")
                    success_manifest, _ = run_script("01_build_filing_manifest.py")
                    if success_manifest:
                        status.update(label="Filing discovery complete!", state="complete")
                        st.toast("Discovered available filings!", icon="🚀")
                        st.rerun()
                        
    with col_btn3:
        if st.button("🔥 Force Re-fetch All Metadata from SEC", type="primary", use_container_width=True, key="force_refetch_all_btn", help="Update the firm universe and force re-download all CIK JSON filings from SEC EDGAR, ignoring local caches."):
            save_config(config)
            st.toast("Configuration saved!", icon="💾")
            success, _ = run_script("00_build_firm_universe.py")
            if success:
                st.toast("Firm universe updated successfully!", icon="✅")
                with st.status("Fetching available filings from SEC EDGAR (force refetch)...", expanded=True) as status:
                    st.write("Querying SEC filing indexes...")
                    success_manifest, _ = run_script("01_build_filing_manifest.py", args=["--force"])
                    if success_manifest:
                        status.update(label="Filing discovery complete!", state="complete")
                        st.toast("Discovered available filings!", icon="🚀")
                        st.rerun()

with tab_universe:
    st.subheader("🏢 Active Company Universe")
    st.markdown("Manage the list of target companies for AI disclosure analysis. You can add companies individually, massively by presets, or search by name/industry keywords.")
    
    # Load SEC directory mappings (cached)
    sec_tickers = get_sec_tickers(user_agent=config["sec"]["user_agent"])
    sec_lookup = {}
    for entry in sec_tickers.values():
        sec_lookup[entry["ticker"].upper()] = {
            "cik": str(entry["cik_str"]).zfill(10),
            "company_name": entry["title"]
        }

    col_univ_left, col_univ_right = st.columns([1, 1])
    
    with col_univ_left:
        st.markdown("### 📋 Active Universe")
        st.write(f"Currently configured: **{len(st.session_state.active_tickers)}** companies.")
        
        if not st.session_state.active_tickers:
            st.info("No companies in the active universe yet. Add some companies using the controls on the right!")
        else:
            # Build details of active tickers
            active_details = []
            for tk in st.session_state.active_tickers:
                tk_upper = tk.upper()
                if tk_upper in sec_lookup:
                    info = sec_lookup[tk_upper]
                    active_details.append({
                        "Ticker": tk_upper,
                        "CIK": info["cik"],
                        "Company Name": info["company_name"]
                    })
                else:
                    active_details.append({
                        "Ticker": tk_upper,
                        "CIK": "Unknown",
                        "Company Name": "Unknown"
                    })
            active_df = pd.DataFrame(active_details)
            
            st.write("Use checkboxes to select companies for deletion:")
            # Display active universe in an interactive dataframe with row selection
            event_active = st.dataframe(
                active_df,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="multi-row",
                key="active_universe_table"
            )
            
            selected_active_indices = event_active.selection.rows
            
            col_act_left, col_act_right = st.columns(2)
            with col_act_left:
                remove_selected_label = f"🗑️ Remove Selected ({len(selected_active_indices)})" if selected_active_indices else "🗑️ Remove Selected"
                if st.button(remove_selected_label, type="secondary", disabled=not selected_active_indices, use_container_width=True, key="remove_selected_active_btn"):
                    tickers_to_remove = active_df.iloc[selected_active_indices]["Ticker"].tolist()
                    for tk in tickers_to_remove:
                        if tk in st.session_state.active_tickers:
                            st.session_state.active_tickers.remove(tk)
                    st.toast(f"Removed {len(tickers_to_remove)} tickers!", icon="🗑️")
                    st.rerun()
            with col_act_right:
                if st.button("🗑️ Clear Active Universe", use_container_width=True, key="clear_univ_btn"):
                    st.session_state.active_tickers = []
                    st.rerun()
                    
            # Collapsed raw tickers list
            with st.expander("📋 View/Copy Raw Tickers List", expanded=False):
                raw_tks_str = ", ".join(sorted(st.session_state.active_tickers))
                st.code(raw_tks_str, language="text")
                st.caption("Double click/select all to copy the raw ticker list.")
        
        # Display registered universe summary (from the built Parquet, if it exists)
        if universe_path.exists():
            universe_df = pd.read_parquet(universe_path)
            st.markdown("#### 🔍 Built Universe Data:")
            st.dataframe(universe_df, use_container_width=True, hide_index=True)
            
        # Large primary save button
        st.markdown("---")
        st.markdown("#### 🔄 Apply Universe Changes")
        st.write("After changing tickers, you must save and update to pull company metadata and build the filing manifest:")
        if st.button("💾 Save Config & Update Universe", type="primary", use_container_width=True, key="save_univ_btn"):
            save_config(config)
            st.success("Configuration saved!")
            success, _ = run_script("00_build_firm_universe.py")
            if success:
                st.toast("Firm universe updated successfully!", icon="✅")
                with st.status("Fetching available filings from SEC EDGAR...", expanded=True) as status:
                    st.write("Querying SEC filing indexes...")
                    success_manifest, _ = run_script("01_build_filing_manifest.py")
                    if success_manifest:
                        status.update(label="Filing discovery complete!", state="complete")
                        st.toast("Discovered available filings!", icon="🚀")
                        st.rerun()
                        
    with col_univ_right:
        st.markdown("### ➕ Add Companies")
        
        # Expanders for different add methods
        with st.expander("🔍 Search SEC Directory (Typeahead)", expanded=True):
            options_map = {}
            for entry in sec_tickers.values():
                tk = entry["ticker"].upper()
                title = entry["title"]
                options_map[f"{tk} - {title}"] = tk

            options_list = [""] + sorted(list(options_map.keys()))

            selected_to_add = st.selectbox(
                "Search SEC Directory",
                options=options_list,
                index=0,
                key="tab_search_selectbox",
                help="Type to search company name or ticker"
            )
            
            if selected_to_add:
                tk = options_map[selected_to_add]
                info = sec_lookup.get(tk, {"cik": "Unknown", "company_name": "Unknown"})
                st.markdown(f"""
                **Selected Company Details:**
                * **Ticker:** `{tk}`
                * **CIK:** `{info['cik']}`
                * **Company Name:** `{info['company_name']}`
                """)
                
                if st.button("➕ Add Selected Company", type="primary", use_container_width=True, key="add_selected_btn"):
                    if tk not in st.session_state.active_tickers:
                        st.session_state.active_tickers.append(tk)
                        st.toast(f"Added {tk}!", icon="✅")
                        st.session_state.tab_search_selectbox = ""
                        st.rerun()
                    else:
                        st.warning(f"{tk} is already in the active universe.")
                        
        with st.expander("✨ Add Industry Presets", expanded=True):
            PRESETS = {
                "Big Tech (Hyperscalers)": ["AAPL", "GOOGL", "META", "AMZN", "MSFT", "TSLA", "NFLX"],
                "Software & SaaS": ["CRM", "ADBE", "ORCL", "PLTR", "SNOW", "INTU", "PANW", "WDAY", "NOW", "DDOG", "NET", "CRWD", "OKTA", "ZS", "TEAM", "ADSK"],
                "Semiconductors & Hardware": ["NVDA", "AMD", "QCOM", "INTC", "AVGO", "TXN", "MU", "AMAT", "LRCX", "KLAC", "ADI", "ASML", "TSM", "MRVL"],
                "Financial Services & Payments": ["JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA", "PYPL", "SQ", "AXP", "DFS", "COF", "BLK", "MSCI"],
                "Healthcare & Biotech": ["JNJ", "PFE", "LLY", "MRK", "ABBV", "AMGN", "GILD", "BMY", "TMO", "ISRG", "REGN", "VRTX", "MRNA", "BIIB"],
                "Energy & Utilities": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "NEE", "DUK", "SO", "AEP", "SRE"],
                "Retail & Consumer Goods": ["WMT", "TGT", "COST", "HD", "LOW", "NKE", "SBUX", "MCD", "KO", "PEP", "PG", "EL", "UL", "CL"],
                "Aerospace & Defense": ["LMT", "RTX", "BA", "NOC", "GD", "LHX", "HWM", "TDG"],
                "Automotive & Transportation": ["TSLA", "F", "GM", "TM", "HMC", "RIVN", "UPS", "FDX", "UNP", "CSX"],
                "Telecommunications": ["T", "VZ", "TMUS", "CHTR", "CMCSA"],
                "Industrials & Manufacturing": ["GE", "HON", "MMM", "CAT", "DE", "ITW"]
            }
            
            preset_choice = st.selectbox("Select Industry Sector", options=list(PRESETS.keys()), key="preset_sector_selectbox")
            
            preset_tickers = PRESETS[preset_choice]
            tickers_to_add_preset = st.multiselect(
                "Select Tickers to Add",
                options=preset_tickers,
                default=preset_tickers,
                key="preset_tickers_multiselect"
            )
            
            if st.button("➕ Add Preset Tickers", use_container_width=True, key="add_preset_btn"):
                if tickers_to_add_preset:
                    added_count = 0
                    for tk in tickers_to_add_preset:
                        if tk not in st.session_state.active_tickers:
                            st.session_state.active_tickers.append(tk)
                            added_count += 1
                    if added_count > 0:
                        st.toast(f"Added {added_count} tickers from {preset_choice} preset!", icon="✅")
                        st.rerun()
                    else:
                        st.warning("All selected tickers are already in the universe.")
                else:
                    st.error("Please select at least one ticker to add.")
                    
        with st.expander("⚡ Search SEC Directory by Keyword (Mass Add)", expanded=True):
            st.write("Search the 10,000+ SEC companies by name keywords (e.g. 'Software', 'Semiconductor', 'Medical', 'Bank') and add them in bulk.")
            
            # Quick search buttons
            st.write("Quick Search Keywords:")
            quick_keywords = ["Software", "Semiconductor", "Therapeutics", "Bank", "Energy", "Retail"]
            qk_cols = st.columns(6)
            for idx, qk in enumerate(quick_keywords):
                if qk_cols[idx].button(qk, key=f"quick_key_{qk}", use_container_width=True):
                    st.session_state.search_keyword_input = qk
                    st.rerun()
            
            keyword = st.text_input("Enter keyword (case-insensitive):", value="", key="search_keyword_input")
            
            if keyword:
                keyword_upper = keyword.upper()
                matching_firms = []
                for entry in sec_tickers.values():
                    title = entry["title"].upper()
                    tk = entry["ticker"].upper()
                    if keyword_upper in title or keyword_upper in tk:
                        matching_firms.append({
                            "Ticker": tk,
                            "CIK": str(entry["cik_str"]).zfill(10),
                            "Company Title": entry["title"]
                        })
                        
                st.write(f"Found **{len(matching_firms)}** matching companies in SEC database.")
                
                if matching_firms:
                    match_df = pd.DataFrame(matching_firms)
                    
                    st.write("Select companies to add, or add all matched companies:")
                    
                    event_search = st.dataframe(
                        match_df,
                        use_container_width=True,
                        hide_index=True,
                        on_select="rerun",
                        selection_mode="multi-row",
                        key="keyword_search_table"
                    )
                    
                    selected_search_indices = event_search.selection.rows
                    
                    col_search_left, col_search_right = st.columns(2)
                    with col_search_left:
                        add_selected_label = f"➕ Add Selected ({len(selected_search_indices)})" if selected_search_indices else "➕ Add Selected"
                        if st.button(add_selected_label, type="primary", disabled=not selected_search_indices, use_container_width=True, key="add_selected_match_btn"):
                            added_count = 0
                            tickers_to_add = match_df.iloc[selected_search_indices]["Ticker"].tolist()
                            for tk in tickers_to_add:
                                if tk not in st.session_state.active_tickers:
                                    st.session_state.active_tickers.append(tk)
                                    added_count += 1
                            st.toast(f"Added {added_count} selected companies!", icon="✅")
                            st.rerun()
                            
                    with col_search_right:
                        if st.button(f"➕ Add All {len(matching_firms)} Matches", use_container_width=True, key="add_all_match_btn"):
                            added_count = 0
                            for _, row_data in match_df.iterrows():
                                tk = row_data["Ticker"].upper()
                                if tk not in st.session_state.active_tickers:
                                    st.session_state.active_tickers.append(tk)
                                    added_count += 1
                            st.toast(f"Added {added_count} companies!", icon="✅")
                            st.rerun()
 
        with st.expander("📝 Bulk Add / Paste Tickers"):
            bulk_input = st.text_area("Paste comma-separated or new-line-separated ticker symbols (e.g., AAPL, MSFT, GOOGL):", height=100, key="bulk_paste_input")
            if st.button("➕ Add Bulk Tickers", use_container_width=True, key="add_bulk_btn"):
                if bulk_input:
                    # Parse using regex to catch spaces, commas, new lines
                    raw_tickers = re.split(r'[\s,;\n\r]+', bulk_input)
                    added_count = 0
                    for raw_tk in raw_tickers:
                        tk = raw_tk.strip().upper()
                        if tk and tk not in st.session_state.active_tickers:
                            st.session_state.active_tickers.append(tk)
                            added_count += 1
                    if added_count > 0:
                        st.toast(f"Added {added_count} bulk tickers!", icon="✅")
                        # Clear inputs
                        st.session_state.bulk_paste_input = ""
                        st.rerun()
                    else:
                        st.warning("No new tickers were added.")

with tab1:
    st.subheader("Filing Discovery and Download Manifest")
    
    # Check if universe exists
    if not universe_path.exists():
        st.warning("⚠️ No firm universe exists yet. Add tickers in the '🏢 Universe Management' tab and run the universe update to get started.")
    else:
        universe_df = pd.read_parquet(universe_path)
        
        # Display registered universe
        with st.expander(f"Registered Universe ({len(universe_df)} Firms)"):
            st.dataframe(universe_df, use_container_width=True)
            
        # Discover Filings button
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔍 Force-Discover SEC Filings", type="secondary", use_container_width=True, help="Fetch latest index for all target firms from the SEC website"):
                save_config(config)
                # Run build manifest
                success, _ = run_script("01_build_filing_manifest.py", args=["--force"])
                if success:
                    st.toast("Discovered available filings!", icon="🚀")
                    st.rerun()
        
        if manifest_path.exists():
            manifest_df = pd.read_parquet(manifest_path)
            
            # Extract year for matrix grouping
            def get_year(row):
                fd = row["filing_date"]
                if hasattr(fd, "year"):
                    return fd.year
                try:
                    return pd.to_datetime(str(fd).split()[0]).year
                except Exception:
                    return None
            
            manifest_df["year"] = manifest_df.apply(get_year, axis=1)
            
            # Filter layout values
            manifest_years = sorted([y for y in manifest_df["year"].unique() if y is not None])
            manifest_tickers = sorted(manifest_df["ticker"].unique())
            
            # Helper to format status cell
            def get_cell_status(subset):
                if subset.empty:
                    return "—"
                parts = []
                for _, row in subset.iterrows():
                    form = row["form_type"]
                    dl = row["download_status"]
                    ps = row["parse_status"]
                    pf = row["prefilter_status"]
                    
                    if dl == "pending":
                        icon = "📥" # Pending
                    elif dl == "selected":
                        icon = "⏳" # Selected
                    elif dl == "completed":
                        if ps == "pending":
                            icon = "💾" # Downloaded
                        elif ps.startswith("failed"):
                            icon = "⚠️" # Parse failed
                        elif ps == "completed":
                            if pf == "pending":
                                icon = "🔍" # Parsed
                            elif pf in ("matched", "no_matches"):
                                icon = "🧩" # Chunked
                            else:
                                icon = "🔍"
                        else:
                            icon = "💾"
                    else:
                        icon = "❓"
                    parts.append(f"**{form}** {icon}")
                return " | ".join(parts)

            # Build Matrix Dataframe
            matrix_rows = []
            for ticker in manifest_tickers:
                row_dict = {"Company/Ticker": ticker}
                for yr in manifest_years:
                    sub = manifest_df[(manifest_df["ticker"] == ticker) & (manifest_df["year"] == yr)]
                    row_dict[str(yr)] = get_cell_status(sub)
                matrix_rows.append(row_dict)
            
            matrix_df = pd.DataFrame(matrix_rows)
            
            st.markdown("### 📊 Filing Availability & Status Matrix")
            st.markdown("This matrix shows which filings were found in the SEC index for each year and their current processing status:")
            
            st.dataframe(
                matrix_df,
                use_container_width=True,
                hide_index=True
            )
            
            # Beautiful HTML Legend
            st.markdown("""
            <div style='background-color: #f8f9fa; padding: 12px 18px; border-radius: 8px; border: 1px solid #e9ecef; margin-bottom: 20px; font-size: 0.9rem; line-height: 1.6;'>
                <strong>Pipeline Status Legend:</strong> &nbsp;&nbsp;&nbsp;&nbsp;
                <span>📥 <strong>Pending</strong> (Available on SEC, not downloaded)</span> &nbsp;&nbsp;|&nbsp;&nbsp;
                <span>⏳ <strong>Selected</strong> (Marked for processing)</span> &nbsp;&nbsp;|&nbsp;&nbsp;
                <span>💾 <strong>Downloaded</strong> (HTML saved locally)</span> &nbsp;&nbsp;|&nbsp;&nbsp;
                <span>🔍 <strong>Parsed</strong> (Text extracted to Markdown)</span> &nbsp;&nbsp;|&nbsp;&nbsp;
                <span>🧩 <strong>Chunked</strong> (AI mentions prefiltered &amp; candidates saved)</span> &nbsp;&nbsp;|&nbsp;&nbsp;
                <span>⚠️ <strong>Failed</strong> (Extraction error)</span> &nbsp;&nbsp;|&nbsp;&nbsp;
                <span>— <strong>N/A</strong> (Did not file / not found)</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Interactive Batch Selection Rules
            st.markdown("---")
            st.subheader("⚡ Batch Selection Controls")
            st.write("Easily select or unselect groups of filings for downloading and parsing:")
            
            col_sel1, col_sel2, col_sel3 = st.columns(3)
            with col_sel1:
                sel_tickers = st.multiselect("Filter by Tickers", manifest_tickers, help="Leave empty for all")
            with col_sel2:
                sel_years = st.multiselect("Filter by Years", [str(y) for y in manifest_years], help="Leave empty for all")
            with col_sel3:
                sel_forms = st.multiselect("Filter by Form Types", sorted(manifest_df["form_type"].unique()), help="Leave empty for all")
                
            # Selection action buttons
            action_col1, action_col2, action_col3 = st.columns(3)
            
            def matches_filter(row):
                if sel_tickers and row["ticker"] not in sel_tickers:
                    return False
                if sel_years and str(row["year"]) not in sel_years:
                    return False
                if sel_forms and row["form_type"] not in sel_forms:
                    return False
                return True
                
            with action_col1:
                if st.button("👉 Mark Matching as Selected", type="primary", use_container_width=True, help="Mark filtered filings for download and section extraction"):
                    selected_count = 0
                    for idx, row in manifest_df.iterrows():
                        if matches_filter(row):
                            # We only select pending filings (completed ones don't need re-download/re-parse)
                            if row["download_status"] == "pending":
                                manifest_df.at[idx, "download_status"] = "selected"
                                manifest_df.at[idx, "batch_id"] = "streamlit_batch"
                                manifest_df.at[idx, "updated_at"] = datetime.now()
                                selected_count += 1
                    if selected_count > 0:
                        manifest_df.drop(columns=["year"], errors="ignore").to_parquet(manifest_path, index=False)
                        st.success(f"Selected {selected_count} filings!")
                        st.rerun()
                    else:
                        st.warning("No pending filings matched the criteria.")
                        
            with action_col2:
                if st.button("👈 Unselect Matching", use_container_width=True, help="Remove selection mark from matching filings"):
                    unselected_count = 0
                    for idx, row in manifest_df.iterrows():
                        if matches_filter(row):
                            if row["download_status"] == "selected":
                                manifest_df.at[idx, "download_status"] = "pending"
                                manifest_df.at[idx, "batch_id"] = ""
                                manifest_df.at[idx, "updated_at"] = datetime.now()
                                unselected_count += 1
                    if unselected_count > 0:
                        manifest_df.drop(columns=["year"], errors="ignore").to_parquet(manifest_path, index=False)
                        st.success(f"Unselected {unselected_count} filings!")
                        st.rerun()
                    else:
                        st.warning("No selected filings matched the criteria.")
                        
            with action_col3:
                if st.button("🧹 Reset All to Pending", use_container_width=True, help="Clear all selections and reset failed download states back to pending"):
                    manifest_df.loc[manifest_df["download_status"] == "selected", "download_status"] = "pending"
                    manifest_df.loc[manifest_df["download_status"].str.startswith("failed", na=False), "download_status"] = "pending"
                    manifest_df["batch_id"] = ""
                    manifest_df["updated_at"] = datetime.now()
                    manifest_df.drop(columns=["year"], errors="ignore").to_parquet(manifest_path, index=False)
                    st.success("All selection states reset!")
                    st.rerun()
            
            # Expander for Detailed Flat Table
            with st.expander("🔍 View Detailed Flat Filing Table"):
                display_cols = ["ticker", "filing_date", "form_type", "download_status", "parse_status", "prefilter_status", "accession_number", "sec_url"]
                
                # Use selection mode to allow checking rows directly (requires Streamlit >= 1.35.0)
                event = st.dataframe(
                    manifest_df[display_cols],
                    column_config={
                        "accession_number": st.column_config.TextColumn("Accession Number"),
                        "ticker": st.column_config.TextColumn("Ticker"),
                        "filing_date": st.column_config.DateColumn("Filing Date"),
                        "form_type": st.column_config.TextColumn("Form Type"),
                        "download_status": st.column_config.TextColumn("Download Status"),
                        "parse_status": st.column_config.TextColumn("Parse Status"),
                        "prefilter_status": st.column_config.TextColumn("Prefilter Status"),
                        "sec_url": st.column_config.LinkColumn("SEC Link")
                    },
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="multi-row"
                )
                
                selected_indices = event.selection.rows
                if selected_indices:
                    st.markdown(f"**Selected {len(selected_indices)} row(s) in the table above:**")
                    col_act1, col_act2 = st.columns(2)
                    with col_act1:
                        if st.button("👉 Mark Selected Rows as Selected for Processing", type="primary", use_container_width=True):
                            selected_accessions = manifest_df.iloc[selected_indices]["accession_number"].tolist()
                            manifest_df.loc[manifest_df["accession_number"].isin(selected_accessions) & (manifest_df["download_status"] == "pending"), "download_status"] = "selected"
                            manifest_df.loc[manifest_df["accession_number"].isin(selected_accessions), "batch_id"] = "streamlit_batch"
                            manifest_df.loc[manifest_df["accession_number"].isin(selected_accessions), "updated_at"] = datetime.now()
                            manifest_df.drop(columns=["year"], errors="ignore").to_parquet(manifest_path, index=False)
                            st.success(f"Selected {len(selected_accessions)} filings!")
                            st.rerun()
                    with col_act2:
                        if st.button("👈 Mark Selected Rows as Pending (Unselect)", use_container_width=True):
                            selected_accessions = manifest_df.iloc[selected_indices]["accession_number"].tolist()
                            manifest_df.loc[manifest_df["accession_number"].isin(selected_accessions) & (manifest_df["download_status"] == "selected"), "download_status"] = "pending"
                            manifest_df.loc[manifest_df["accession_number"].isin(selected_accessions), "batch_id"] = ""
                            manifest_df.loc[manifest_df["accession_number"].isin(selected_accessions), "updated_at"] = datetime.now()
                            manifest_df.drop(columns=["year"], errors="ignore").to_parquet(manifest_path, index=False)
                            st.success(f"Unselected {len(selected_accessions)} filings!")
                            st.rerun()
                
            # Run download and processing pipelines
            st.markdown("---")
            st.subheader("⚡ Pipeline Control Panel")
            st.write("Process the selected filings end-to-end (Download -> Section Extract -> AI Prefilter -> Candidate Chunking):")
            
            # Count selected filings
            selected_count = len(manifest_df[manifest_df["download_status"] == "selected"])
            st.info(f"Currently selected for processing: **{selected_count}** filings.")
            
            if st.button("🚀 Run Processing Pipeline on Selected Filings", type="primary", disabled=(selected_count == 0)):
                # Execute sequential pipeline scripts
                steps = [
                    ("03_download_selected_filings.py", "Downloading files..."),
                    ("04_extract_sections.py", "Extracting sections..."),
                    ("05_prefilter_ai_mentions.py", "Prefiltering AI mentions..."),
                    ("06_chunk_candidates.py", "Chunking candidates...")
                ]
                
                success_all = True
                for script, desc in steps:
                    st.info(f"Executing: {desc}")
                    success, _ = run_script(script)
                    if not success:
                        success_all = False
                        st.error(f"Pipeline stopped at {script}")
                        break
                        
                if success_all:
                    st.balloons()
                    st.success("Pipeline ran successfully end-to-end! Chunks are ready in the next tab.")
                    st.rerun()
        else:
            st.info("💡 Click 'Force-Discover SEC Filings' to pull the list of matching reports from the SEC index.")

with tab2:
    st.subheader("🧩 Candidate Chunks Explorer")
    
    if not chunks_path.exists():
        st.info("💡 Candidate chunks dataset (`ai_candidate_chunks.parquet`) has not been generated yet. Complete the pipeline in Tab 1 first.")
    else:
        chunks_df = pd.read_parquet(chunks_path)
        
        # Display Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Candidate Chunks", len(chunks_df))
        with col2:
            st.metric("Unique Text Blocks", chunks_df["text_hash"].nunique())
        with col3:
            st.metric("Represented Firms", chunks_df["ticker"].nunique())
            
        # Filters
        st.markdown("### Search & Filter Chunks:")
        f_col1, f_col2, f_col3 = st.columns([1, 1, 2])
        with f_col1:
            ticker_filter = st.multiselect("Filter Tickers", sorted(chunks_df["ticker"].unique()))
        with f_col2:
            section_filter = st.multiselect("Filter Section Areas", sorted(chunks_df["section_name"].unique()))
        with f_col3:
            search_query = st.text_input("🔍 Search within chunk text (case-insensitive)")
            
        # Apply filters
        filtered_df = chunks_df
        if ticker_filter:
            filtered_df = filtered_df[filtered_df["ticker"].isin(ticker_filter)]
        if section_filter:
            filtered_df = filtered_df[filtered_df["section_name"].isin(section_filter)]
        if search_query:
            filtered_df = filtered_df[filtered_df["chunk_text"].str.contains(search_query, case=False, na=False)]
            
        # Display list
        st.write(f"Showing {len(filtered_df)} matching chunks:")
        
        for idx, row in filtered_df.head(50).iterrows():
            with st.container():
                st.markdown(f"**Firm**: `{row['ticker']}` | **Filing Date**: `{row['filing_date']}` | **Section**: `{row['section_name']}` | **Families**: `{row['keyword_family']}` (Keywords: {row['ai_keyword_count']})")
                st.text_area(f"Chunk ID: {row['chunk_id']}", value=row["chunk_text"], height=120, disabled=True, key=f"chunk_{idx}")
                st.markdown("---")

with tab_logs:
    st.subheader("📋 Centralized Pipeline Logs & Diagnostics")
    st.markdown("Explore and query the pipeline logs. Logs are stored in JSONL format, allowing SQL queries directly via DuckDB.")
    
    log_file = Path(config["paths"]["interim_manifests"]) / "pipeline_log.jsonl"
    
    if not log_file.exists():
        st.info("No log entries recorded yet. Run the pipeline to generate logs.")
    else:
        try:
            # Read JSONL file
            log_df = pd.read_json(log_file, lines=True)
            # Reorder columns for better presentation
            cols = ["timestamp", "pipeline_step", "level", "message", "ticker", "cik", "accession_number", "duration_seconds", "details"]
            log_df = log_df[[c for c in cols if c in log_df.columns]]
            
            # Metric row
            col_tot, col_succ, col_warn, col_err = st.columns(4)
            with col_tot:
                st.metric("Total Log Entries", len(log_df))
            with col_succ:
                succ_count = len(log_df[log_df["level"] == "SUCCESS"]) if "level" in log_df.columns else 0
                st.metric("Successes", succ_count)
            with col_warn:
                warn_count = len(log_df[log_df["level"] == "WARNING"]) if "level" in log_df.columns else 0
                st.metric("Warnings", warn_count)
            with col_err:
                err_count = len(log_df[log_df["level"] == "ERROR"]) if "level" in log_df.columns else 0
                st.metric("Errors", err_count)
                
            # Filters
            st.markdown("### 🔍 Search & Filter Logs")
            fl_col1, fl_col2, fl_col3 = st.columns([1, 1, 2])
            with fl_col1:
                levels = sorted(log_df["level"].unique()) if "level" in log_df.columns else []
                selected_levels = st.multiselect("Log Level", levels, default=[])
            with fl_col2:
                steps = sorted(log_df["pipeline_step"].unique()) if "pipeline_step" in log_df.columns else []
                selected_steps = st.multiselect("Pipeline Step", steps, default=[])
            with fl_col3:
                search_text = st.text_input("Search Messages", "")
                
            # Apply filters
            filtered_logs = log_df
            if selected_levels:
                filtered_logs = filtered_logs[filtered_logs["level"].isin(selected_levels)]
            if selected_steps:
                filtered_logs = filtered_logs[filtered_logs["pipeline_step"].isin(selected_steps)]
            if search_text:
                filtered_logs = filtered_logs[
                    filtered_logs["message"].astype(str).str.contains(search_text, case=False, na=False) | 
                    filtered_logs["ticker"].astype(str).str.contains(search_text, case=False, na=False)
                ]
                
            st.dataframe(filtered_logs, use_container_width=True, hide_index=True)
            
            # SQL Console
            st.markdown("### 📊 Interactive SQL Console (DuckDB)")
            st.markdown("Write standard SQL queries against the logs. Refer to the table as `log_df`.")
            
            default_query = "SELECT timestamp, pipeline_step, level, message, ticker FROM log_df ORDER BY timestamp DESC LIMIT 50"
            sql_query = st.text_area("SQL Query", value=default_query, height=100)
            
            if st.button("Execute Query", type="primary"):
                try:
                    import duckdb
                    result_df = duckdb.query(sql_query).df()
                    st.success("Query executed successfully!")
                    st.dataframe(result_df, use_container_width=True, hide_index=True)
                except Exception as ex:
                    st.error(f"SQL Error: {ex}")
        except Exception as e:
            st.error(f"Error loading logs: {e}")

with tab3:
    st.subheader("📘 Project Architecture")
    
    # Read description.md if it exists
    desc_path = Path("docs/description.md")
    if desc_path.exists():
        with open(desc_path, "r") as f:
            desc_content = f.read()
        # Clean lines number prefixes if they were written
        cleaned_desc = re.sub(r"^\d+:\s", "", desc_content, flags=re.MULTILINE)
        st.markdown(cleaned_desc)
    else:
        st.info("Documentation missing at docs/description.md.")
