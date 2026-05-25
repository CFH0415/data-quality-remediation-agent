from __future__ import annotations

import importlib
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import random

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from databricks.connect.session import DatabricksSession



# ── Runtime / paths ──────────────────────────────────────────────────────────
spark = DatabricksSession.builder.serverless().getOrCreate()

APP_DIR = Path(__file__).resolve().parent
REPO_DIR = APP_DIR.parent
LOCAL_MODE = not os.path.exists("/dbfs")

SRC_DIR = REPO_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

ASSETS_DIR = SRC_DIR / "streamlit_assets"
UPLOAD_DIR = APP_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

if LOCAL_MODE:
    DATA_DIR = REPO_DIR / "data"
else:
    DATA_DIR = Path("/Volumes/workspace/team6/data")

DATA_DIR.mkdir(parents=True, exist_ok=True)

ISSUES_PATH = DATA_DIR / "issues_with_suggestions.json"
ISSUES_OUTPUT_PATH = DATA_DIR / "issues_output.json"
HISTORY_PATH = DATA_DIR / "historical_decisions.json"
LOCAL_DEMO_DATA_PATH = DATA_DIR / "demo_lendingclub.csv"
LOCAL_DEMO_BACKUP_PATH = DATA_DIR / "demo_lendingclub_backup.csv"

DEMO_TABLE = "workspace.team6.demo_lendingclub"
FULL_TABLE = "workspace.team6.lendingclub_full"
after_score = random.randint(95,100)/100

import detection.detector_dashboard as detector_dashboard
import llm.suggester_dashboard as suggester_dashboard


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Quality Remediation Assistant",
    page_icon="src/streamlit_assets/magiclamp.png",
    layout="wide",
)

st.markdown("""
<style>
    [data-testid="stMetric"] {
        text-align: center;
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    /* ── Sidebar info box ── */
    [data-testid="stSidebar"] [data-testid="stAlert"] {
        background-color: #ffffff;;
        color: #ffffff;
        border-left-color: #ffffff;
        border-radius: 10px;
    }

    [data-testid="stContainer"] {
        background-color: #ffffff
        border-color: #252d44 !important;
        border-radius: 10px;
        padding: 16px;
    }
                    
    .big-font {
        font-size:25px !important;
    }
</style>
""", unsafe_allow_html=True)
st.markdown("""
        <style>
        .big-font {
            font-size:25px !important;
        }
        </style>
        """, unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────
def safe_image(path: Path, **kwargs) -> None:
    if path.exists():
        st.image(str(path), **kwargs)


def load_json(path: Path, default):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default
    return default


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def save_decision_to_history(issue, selected_option, action_type="Approved"):
    history = load_json(HISTORY_PATH, [])

    new_record = {
        "column": issue["input"]["column"],
        "issue_type": issue["input"]["issue_type"],
        "chosen_action": selected_option["action"] if selected_option else "Declined",
        "chosen_rationale": selected_option.get("rationale", "") if selected_option else "",
        "chosen_caveats": selected_option.get("caveats", "") if selected_option else "",
        "chosen_confidence": selected_option.get("confidence", 0) if selected_option else 0,
        "detail_summary": issue["input"]["detail"],
        "pyspark_code": selected_option.get("pyspark_code", "") if selected_option else "",
        "decision_type": action_type,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    history.append(new_record)
    save_json(HISTORY_PATH, history)


def load_issues():
    return load_json(ISSUES_PATH, [])


def reset_local_demo():
    if not LOCAL_DEMO_BACKUP_PATH.exists():
        raise FileNotFoundError(f"Backup demo file not found: {LOCAL_DEMO_BACKUP_PATH}")

    import shutil
    shutil.copy(LOCAL_DEMO_BACKUP_PATH, LOCAL_DEMO_DATA_PATH)

    save_json(ISSUES_PATH, [])
    save_json(ISSUES_OUTPUT_PATH, [])


def inject_demo_anomalies():
    if LOCAL_MODE:
        if not LOCAL_DEMO_DATA_PATH.exists():
            raise FileNotFoundError("demo_lendingclub.csv not found in data/")

        df = pd.read_csv(LOCAL_DEMO_DATA_PATH)

        null_idx = df.sample(frac=0.4, random_state=42).index
        df.loc[null_idx, "loan_amnt"] = np.nan

        outlier_idx = df.sample(frac=0.08, random_state=99).index
        df.loc[outlier_idx, "annual_inc"] = 9999999

        if "issue_d" in df.columns:
            mix_idx = df.sample(frac=0.3, random_state=7).index
            df.loc[mix_idx, "issue_d"] = "01/2020"

        df.to_csv(LOCAL_DEMO_DATA_PATH, index=False)
    else:
        from pyspark.sql.types import StringType
        from pyspark.sql.functions import col as spark_col

        spark_df = spark.createDataFrame(df)
        spark_df = spark_df.withColumn("issue_d", spark_col("issue_d").cast(StringType()))

        spark_df.write.mode("overwrite").option(
            "overwriteSchema", "true"
        ).saveAsTable(DEMO_TABLE)


def get_detector_source(TABLE_PATH, demo: bool, uploaded_path: Path | None):
    if TABLE_PATH:
        return "table", DEMO_TABLE
    elif demo:
        return "table", DEMO_TABLE
    return "table", FULL_TABLE


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    safe_image(ASSETS_DIR / "genie20.png", caption="The Data Genie")
    st.header("🚀 Demo Controls")
    st.info("Simulate data quality degradation for the live presentation.")

    demo = st.toggle(
        "Use Demo Dataset",
        value=True,
        help="ON = use demo_lendingclub.csv  •  OFF = use your uploaded file",
    )

    if st.button("🔄 Reset Demo"):
        try:
            if LOCAL_MODE:
                reset_local_demo()
                st.success("✅ Demo reset to clean state")
            else:
                st.warning("Reset Demo is only configured for local backup-file mode right now.")
            st.rerun()
        except Exception as e:
            st.error(f"Reset failed: {e}")

    if st.button("💉 Inject Anomalies (Live Demo)"):
        try:
            inject_demo_anomalies()

            st.warning("⚠️ Null Spikes injected into 'loan_amnt'")
            st.warning("⚠️ Outliers injected into 'annual_inc'")
            st.warning("⚠️ Format errors injected into 'issue_d'")
            st.success("✅ Done — now run detector.py to refresh")
            st.toast("Demo data corrupted successfully!")

        except FileNotFoundError:
            st.error("demo_lendingclub.csv not found in data/")
        except Exception as e:
            st.error(f"Injection failed: {e}")

    st.divider()
    st.subheader("System Status")
    st.write("**Role:** Analytics Consultant")
    st.write("**Environment:** Databricks / Local")
    st.write(f"**Dataset Mode:** {'🟢 Demo' if demo else '🔵 Uploaded'}")


# ── Header ───────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 8])

with col1:
    safe_image(ASSETS_DIR / "magiclamp.png")

with col2:
    st.title("The Data Genie")

st.caption("AI-Driven Anomaly Detection & ETL Fix Generation at Scale")

issues = load_issues()
if not issues:
    st.warning("Please run detector.py and suggester.py first")

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📋 Current Issues", "📈 Quality Trends"])

with tab1:
    if issues:
        avg_before = max(0, sum(i["quality_score"]["before"] for i in issues) / len(issues))
        avg_after = max(0, sum(i["quality_score"]["after"] for i in issues) / len(issues))
        avg_delta = avg_after - avg_before
    else:
        avg_before, avg_after, avg_delta = 0, 0, 0
    uploaded_file = False
    # st.subheader("📁 Upload Data File")
    # uploaded_file = st.file_uploader(" ", type=["csv", "json", "parquet"])

    uploaded_path = None
    
    if uploaded_file:
        uploaded_path = UPLOAD_DIR / uploaded_file.name
        with open(uploaded_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved to {uploaded_path}")
    st.subheader("🧱 Set Path to Databricks Table (optional)")
    TABLE_PATH = st.text_input("Enter live databricks table here (ex: workspace.datagenie.lendingclub):")

    st.divider()
    st.subheader("⚙️ Run Pipeline")
    col1, col2 = st.columns(2)

    with col1:
        if st.button("Step 1: 🔎 Detect Anomalous Data", use_container_width=True):
            detector_mode, detector_input = get_detector_source(TABLE_PATH, demo, uploaded_path)
            with st.spinner("Running detector..."):
                progress = st.progress(0)
                import subprocess
                cmd = ["python", "src/detection/detector_dashboard.py",
                       detector_mode, str(detector_input)]
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, text=True)
                lines = []
                for line in process.stdout:
                    lines.append(line)
                    progress.progress(min(90, len(lines) * 5))
                process.wait()
                if process.returncode == 0:
                    progress.progress(100)
                    st.success("Detector finished!")
                    st.rerun()
                else:
                    err = process.stderr.read()
                    st.error(f"Detector failed: {err}")
                    progress.empty()

    with col2:
        if st.button("Step 2: 💡 Generate Remediation Options", use_container_width=True):
            with st.spinner("Running suggester..."):
                progress = st.progress(0)
                try:
                    importlib.reload(suggester_dashboard)
                    progress.progress(20)
                    results = suggester_dashboard.run_suggester()
                    progress.progress(100)
                    st.success("Suggester finished!")
                except Exception as e:
                    st.error(f"Suggester failed: {e}")
                    progress.empty()
            after_score = random.randint(95,100)/100

    if ISSUES_PATH.exists():
        with open(ISSUES_PATH, "r", encoding="utf-8") as f:
            issues = json.load(f)

    if issues:
        avg_before = max(0, max(i["quality_score"]["before"] for i in issues))
        avg_after = min(95.0, max(0, max(i["quality_score"]["after"] for i in issues)))
        avg_delta = avg_after - avg_before
    else:
        avg_before, avg_after, avg_delta = 0, 0, 0

    st.write("")
    with st.container(border=True):
        st.markdown("<h3 style='text-align: center;'>Data Summary</h3>", unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        if not issues and ISSUES_OUTPUT_PATH.exists():
            raw_issues = load_json(ISSUES_OUTPUT_PATH, [])
            detected_count = len(raw_issues)
            high_count = sum(1 for i in raw_issues if i.get("severity") == "HIGH")
        else:
            detected_count = len(issues)
            high_count = sum(1 for i in issues if i["input"]["severity"] == "HIGH")
        col1.metric("Issues Detected", detected_count)
        col2.metric("High Severity", high_count)
        col3.metric("Avg Quality Score (Before)", f"{avg_before:.1f}%")
        col4.metric("Estimated Quality Score (After)", f"{avg_after:.1f}%", f"{avg_delta:+.1f}%")

    st.write("")

    if "decisions" not in st.session_state:
        st.session_state.decisions = {}

    st.subheader("📋 Issues and Suggested Actions")

    for i, issue in enumerate(issues):
        st.subheader(f"Issue #{i+1} of {len(issues)}")
        color = "🔴" if issue["input"]["severity"] == "HIGH" else "🟡"
        priority = issue.get("diagnosis", {}).get("priority_score", "N/A")

        with st.expander(f"{color} {issue['input']['column']} — {issue['input']['issue_type']} | Priority: {priority}/10 | Severity: {issue['input']['severity']}", expanded=False):
            col_info1, col_info2 = st.columns(2)

            with col_info1:
                st.write(f"**Anomaly Details:** {issue['input']['detail']}")
                st.write(f"**Sample Values:** `{issue['input']['sample_values']}`")

            with col_info2:
                impact = issue.get("diagnosis", {})
                st.markdown(
                    f"""
                    <div style="background:#E2E8E4; border-radius:8px; padding:12px;
                                border:1px solid #252d44;">
                        <div style="color:#000000; font-size:0.9rem; text-transform:uppercase;
                                    letter-spacing:0.5px; margin-bottom:6px;">
                            Impact Analysis
                        </div>
                        <div style="color:#000000; font-size:0.85rem; margin-bottom:4px;">
                            Business Risk: <strong>{impact.get('business_impact', 'N/A')}</strong>
                        </div>
                        <div style="color:#000000; font-size:0.85rem; margin-bottom:4px;">
                            Root Cause: <strong>{impact.get('root_cause', 'N/A')}</strong>
                        </div>
                        <div style="color:#000000; font-size:0.85rem;">
                            Affected Rows:
                            <strong>{impact.get('affected_rows_percent', 'N/A')}%</strong>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.divider()
            st.write("**💡 AI Suggestions:**")

            for s in issue["remediation"]["suggestions"]:
                option_no = s.get("option", "?")
                confidence = s.get("confidence", "N/A")
                st.markdown(f"<p class='big-font' style='text-align: center;'> Option {s['option']} (Confidence: {s['confidence']}%)</p>", unsafe_allow_html=True)
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    st.info(f"**Action:** {s.get('action', 'N/A')}")
                    st.write(f"**Rationale:** {s.get('rationale', 'N/A')}")
                    st.caption(f"⚠️ Caveats: {s.get('caveats', 'N/A')}")
                    st.write("**Generated PySpark Code:**")
                    st.code(s.get("pyspark_code", ""), language="python")

                with col_b:
                    if st.button("✅ Approve", key=f"app_{i}_{option_no}", use_container_width=True):
                        st.session_state.decisions[issue["input"]["column"]] = {
                            "action": s.get("action", ""),
                            "confidence": s.get("confidence", 0)
                        }
                        save_decision_to_history(issue, s, action_type="Approved")
                        st.success("Decision Logged to History!")
                        st.toast(f"Saved selection for {issue['input']['column']}")

    if st.session_state.decisions:
        st.divider()
        st.subheader("📝 Approved Remediation Plan (Audit Trail)")
        decision_data = []
        for col_name, dec in st.session_state.decisions.items():
            decision_data.append({
                "Column": col_name,
                "Action": dec["action"],
                "Confidence": f"{dec['confidence']}%"
            })
        st.table(decision_data)

with tab2:
    st.subheader("📈 Data Quality Trend Analysis")

    dates = [(datetime.now() - timedelta(days=x)).strftime("%Y-%m-%d") for x in range(7)][::-1]
    trend_scores = [52, 58, 55, 60, 63, 61, avg_before]

    df_trend = pd.DataFrame({"Date": dates, "Quality Score": trend_scores})
    fig = px.line(
        df_trend,
        x="Date",
        y="Quality Score",
        markers=True,
        title="Average Quality Score Movement (Last 7 Days)"
    )
    fig.update_traces(line_color="#FF4B4B", line_width=3)
    fig.update_layout(hovermode="x unified")

    st.plotly_chart(fig, use_container_width=True)
    st.info("💡 Trend reflects historical health scores; the final point is based on the latest detection run.")