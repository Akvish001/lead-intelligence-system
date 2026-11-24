import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MAIN_PY = APP_DIR / "main.py"
REQUIRED_COLUMNS = ["name", "company", "company_size", "industry", "source", "last_interaction_date"]

st.set_page_config(page_title="Lead Intelligence System", page_icon="🎯", layout="wide")
st.title("🎯 Lead Intelligence System")
st.caption("Upload leads or enter one manually, then run your existing qualification pipeline.")

with st.expander("Required CSV columns"):
    st.code(",".join(REQUIRED_COLUMNS))

uploaded_file = st.file_uploader("Upload leads CSV", type=["csv"])
st.subheader("Or enter one lead manually")

with st.form("manual_lead"):
    left, right = st.columns(2)
    with left:
        name = st.text_input("Lead name")
        company = st.text_input("Company")
        company_size = st.selectbox("Company size", ["1-10", "11-50", "51-200", "201-500", "501-1000", "1001-5000", "5000+", "Unknown"])
    with right:
        industry = st.text_input("Industry")
        source = st.selectbox("Lead source", ["Website", "LinkedIn", "Referral", "Event", "Cold outreach", "Inbound", "Other"])
        last_interaction_date = st.date_input("Last interaction date")
    add_manual = st.form_submit_button("Add manual lead")

if "saved_manual_lead" not in st.session_state:
    st.session_state.saved_manual_lead = None

if add_manual:
    if not name.strip() or not company.strip():
        st.error("Enter at least the lead name and company.")
    else:
        st.session_state.saved_manual_lead = {
            "name": name.strip(), "company": company.strip(), "company_size": company_size,
            "industry": industry.strip() or "Unknown", "source": source,
            "last_interaction_date": last_interaction_date.isoformat()
        }
        st.success("Manual lead is ready.")

input_df = None
if uploaded_file is not None:
    try:
        input_df = pd.read_csv(uploaded_file)
        st.subheader("CSV preview")
        st.dataframe(input_df.head(10), use_container_width=True)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")

if st.session_state.saved_manual_lead:
    st.write("Manual lead:")
    st.dataframe(pd.DataFrame([st.session_state.saved_manual_lead]), use_container_width=True)

if st.button("🚀 Run qualification", type="primary"):
    if input_df is None and not st.session_state.saved_manual_lead:
        st.warning("Upload a CSV or add a manual lead first.")
        st.stop()

    if input_df is not None and st.session_state.saved_manual_lead:
        input_df = pd.concat([input_df, pd.DataFrame([st.session_state.saved_manual_lead])], ignore_index=True)
    elif input_df is None:
        input_df = pd.DataFrame([st.session_state.saved_manual_lead])

    missing = [col for col in REQUIRED_COLUMNS if col not in input_df.columns]
    if missing:
        st.error("Missing required CSV columns: " + ", ".join(missing))
        st.info("Rename your columns to match the required format shown above.")
        st.stop()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        input_path = temp_dir / "input_leads.csv"
        output_path = temp_dir / "output_report.json"
        input_df.to_csv(input_path, index=False)
        command = [sys.executable, str(MAIN_PY), "--input", str(input_path), "--output", str(output_path)]

        with st.spinner("Processing leads..."):
            try:
                result = subprocess.run(command, cwd=str(APP_DIR), capture_output=True, text=True, timeout=600)
            except subprocess.TimeoutExpired:
                st.error("Processing exceeded 10 minutes and was stopped.")
                st.stop()

        with st.expander("Pipeline logs"):
            st.text((result.stdout + "\n" + result.stderr)[-16000:])

        if result.returncode != 0:
            st.error("Pipeline failed. Review the logs above.")
            st.stop()
        if not output_path.exists():
            st.error("Pipeline did not create output_report.json.")
            st.stop()

        report_text = output_path.read_text(encoding="utf-8")
        try:
            report = json.loads(report_text)
        except json.JSONDecodeError as exc:
            st.error(f"Could not parse report JSON: {exc}")
            st.stop()

        st.session_state.report = report
        st.session_state.report_text = report_text

if "report" in st.session_state:
    report = st.session_state.report
    st.divider()
    st.header("📊 Results")

    if isinstance(report, dict):
        stats = report.get("stats") or report.get("statistics") or report.get("summary") or {}
        if isinstance(stats, dict) and stats:
            known = [("total_leads", "Total leads"), ("total", "Total leads"), ("qualified", "Qualified"),
                     ("rejected", "Rejected"), ("review", "Needs review")]
            metrics = [(label, stats[key]) for key, label in known if key in stats]
            if metrics:
                cols = st.columns(min(4, len(metrics)))
                for i, (label, value) in enumerate(metrics):
                    cols[i % len(cols)].metric(label, value)

        rows = next((report[key] for key in ("leads", "results", "records", "items", "priority_queue")
                     if isinstance(report.get(key), list)), None)
        if rows is not None:
            result_df = pd.json_normalize(rows)

            st.subheader("Lead results")
            st.dataframe(
                result_df,
                width="stretch",
                hide_index=True
            )

            st.download_button(
                "Download results CSV",
                result_df.to_csv(index=False).encode("utf-8"),
                file_name="lead_qualification_results.csv",
                mime="text/csv"
            )
    else:
        st.json(report)

        st.download_button("Download JSON report", st.session_state.report_text.encode("utf-8"),
                           file_name="output_report.json", mime="application/json")

st.divider()
st.caption("Keep your .env file private. Never share or submit your API key.")
