# How to Use The Data Genie

## Requirements

- Python 3.10+
- Databricks account (for cloud deployment)
- OpenAI API key

---

## Installation

**Step 1: Clone the repository**
```bash
git clone https://github.com/tom666d/big_data_team_6.git
cd big_data_team_6
```

**Step 2: Install dependencies**
```bash
pip install pyspark openai streamlit pandas python-dotenv databricks-connect plotly numpy
```

**Step 3: Set up environment variables**

Create a `.env` file in the root directory:
```
OPENAI_API_KEY=your_openai_api_key_here
```
Never commit this file to GitHub. It is already included in `.gitignore`.

---

## Running Locally

The pipeline has three steps. Run them in order:

**Step 1: Detect anomalies**
```bash
python src/detection/detector_dashboard.py
```
This scans the dataset using Apache Spark and outputs detected issues to `data/issues_output.json`.

**Step 2: Generate AI remediation suggestions**
```bash
python src/llm/suggester_dashboard.py
```
This runs the multi-agent pipeline (Router → Agent 1 → Agent 2 → Critic) and outputs suggestions to `data/issues_with_suggestions.json`.

**Step 3: Launch the dashboard**
```bash
streamlit run streamlit_app/app.py
```
Open `http://localhost:8501` in your browser to access the human review dashboard.

---

## Running on Databricks

**Prerequisites:**
- Databricks workspace with serverless compute enabled
- Databricks CLI configured locally
- Data tables uploaded to your Databricks catalog

**Setup:**
1. Connect your Databricks workspace to this GitHub repository via Databricks Repos
2. Upload the Lending Club dataset to your Databricks catalog (e.g. `workspace.team6.lendingclub_full`)
3. Set `OPENAI_API_KEY` as a Databricks secret
4. Deploy the Streamlit app via Databricks Apps

**Data table paths (update if needed):**
```python
DEMO_TABLE = "workspace.team6.demo_lendingclub"   # 100K rows for demo
FULL_TABLE = "workspace.team6.lendingclub_full"    # 2.5M rows full dataset
```

**Output files are written to:**
```
/Volumes/workspace/team6/data/issues_output.json
/Volumes/workspace/team6/data/issues_with_suggestions.json
/Volumes/workspace/team6/data/historical_decisions.json
```

---

## Demo Workflow

To run the live demo:

1. Open the dashboard
2. Click **Reset Demo** — restores the clean dataset
3. Click **Inject Anomalies** — simulates three data quality failures:
   - 40% null spike in `loan_amnt`
   - 8% statistical outliers in `annual_inc`
   - 30% format inconsistency in `issue_d`
4. Click **Step 1: Detect Anomalous Data** — Spark scans and flags issues
5. Click **Step 2: Generate Remediation Options** — multi-agent pipeline runs
6. Review each issue, read the AI diagnosis and generated PySpark code
7. Click **Approve** to log the decision to the audit trail and feedback loop

---

## File Structure

```
big_data_team_6/
├── src/
│   ├── detection/
│   │   └── detector_dashboard.py     ← Spark anomaly detection
│   ├── llm/
│   │   └── suggester_dashboard.py    ← Multi-agent AI pipeline
│   └── streamlit_assets/             ← UI images and icons
├── streamlit_app/
│   └── app.py                        ← Streamlit dashboard
├── data/
│   ├── demo_lendingclub.csv          ← Local demo dataset
│   ├── demo_lendingclub_backup.csv   ← Clean backup for reset
│   ├── issues_output.json            ← Detector output
│   ├── issues_with_suggestions.json  ← Suggester output
│   └── historical_decisions.json     ← Feedback loop storage
├── .env                              ← API keys (not on GitHub)
├── .gitignore
└── README.md
```

---

## Configuration

| Parameter | Location | Description |
|-----------|----------|-------------|
| `OPENAI_API_KEY` | `.env` | OpenAI API key for GPT-4o-mini |
| `DEMO_TABLE` | `app.py` | Databricks demo table path |
| `FULL_TABLE` | `app.py` | Databricks full dataset table path |
| `null_rate > 0.3` | `detector_dashboard.py` | Threshold for null spike detection |
| `outlier_rate > 0.05` | `detector_dashboard.py` | Threshold for outlier detection |
| `max_retries = 2` | `suggester_dashboard.py` | Critic Agent retry limit |

---

## Troubleshooting

**Detector failed: session_id is no longer usable**
→ Databricks session timed out. The app uses subprocess to spawn a new session automatically. If the error persists, restart the Databricks app.

**issues_output.json not found**
→ Run Step 1 (detector) before Step 2 (suggester).

**OPENAI_API_KEY not set**
→ Make sure your `.env` file exists in the root directory and contains a valid API key.

**Reset Demo only works in local mode**
→ In Databricks mode, manually restore the demo table from your backup.