# Data Quality Remediation Assistant

**MSBA 6431: Big Data** | Team project · My contribution: AI agent pipeline (`src/llm/suggestion.py`)

> An end-to-end data quality remediation system for large-scale lending datasets. The pipeline detects data quality issues using Apache Spark, diagnoses root causes and business impact using a two-agent LLM system, and generates ranked remediation options with executable PySpark fix code — informed by a human-in-the-loop historical feedback loop.

---

## System Architecture

```
Spark Issue Detector (teammate)
          ↓
    issues_output.json
          ↓
  ┌───────────────────┐
  │   Agent 1         │  ← Diagnosis
  │   GPT-4o-mini     │    root_cause, business_impact
  │   + Python logic  │    priority_score, affected_rows_percent
  └───────────────────┘
          ↓
  ┌───────────────────┐
  │   Agent 2         │  ← Remediation
  │   GPT-4o-mini     │    3 options + confidence scores
  │   + history loop  │    + runnable PySpark fix code
  └───────────────────┘
          ↓
  issues_with_suggestions.json
          ↓
  Streamlit UI (teammate)
          ↓
  User selects option → historical_decisions.json (feedback loop)
```

---

## My Contribution

I built the core AI engine: `src/llm/suggestion.py`

| Component | What I built |
|---|---|
| Agent 1 — Diagnosis | GPT-4o-mini agent that generates root cause and business impact for each detected issue |
| Deterministic scoring | Priority score formula and affected-rows-percent parser — computed by Python, not LLM |
| Agent 2 — Remediation | GPT-4o-mini agent that generates 3 ranked remediation options with runnable PySpark code |
| History loop | Loads past human decisions by issue type, feeds the last 5 records into Agent 2's prompt |
| Quality score | Computes dataset-level before/after/delta quality score across all detected issues |

Teammates handled: Spark-based issue detection, Snowflake integration, and Streamlit UI.

---

## AI Agent Design

### Agent 1 — Diagnosis

Takes a detected issue (column, issue_type, severity, detail, sample_values) and returns:

- `root_cause` — upstream or operational explanation (LLM-generated)
- `business_impact` — downstream effect on analytics or modeling (LLM-generated)
- `priority_score` — deterministic formula, **not LLM-generated**
- `affected_rows_percent` — parsed from issue detail text, **not LLM-generated**

**Why hybrid (LLM + deterministic)?**

Letting the LLM estimate a priority score introduces inconsistency — the same issue described differently might get different scores. By computing it deterministically, priority scores are stable, auditable, and comparable across issues.

```python
# Priority score = 0.6 × severity_score + 0.4 × affected_rows_score
# Severity map: LOW=3, MEDIUM=6, HIGH=9
# Affected rows: <5%=1, <15%=3, <30%=5, <50%=7, >=50%=9
priority_score = round(0.6 * severity_score + 0.4 * affected_score)
```

---

### Agent 2 — Remediation

Takes the original issue + Agent 1 diagnosis + historical context and returns 3 options:

| Option | Description |
|---|---|
| Option 1 | Primary recommended fix (highest confidence) |
| Option 2 | Alternative approach with different trade-offs |
| Option 3 | Always "Decline changes" — decision captured for audit |

Each option includes:
- `action` — short remediation title
- `confidence` — 0–100, with defined bands (80–100: standard practice; 60–79: context-dependent; below 60: risky)
- `rationale` — why this fix is appropriate
- `caveats` — assumptions, limitations, or risks
- `pyspark_code` — runnable PySpark code referencing `df`

**Example output:**
```json
{
  "option": 1,
  "action": "Impute Loan Amount with Mean",
  "confidence": 85,
  "rationale": "Replacing null values with the mean maintains overall data integrity.",
  "caveats": "Assumes mean is representative and does not account for outliers.",
  "pyspark_code": "df = df.fillna({'loan_amnt': df.agg({'loan_amnt': 'mean'}).first()[0]})"
}
```

---

### History Loop — Human-in-the-Loop Feedback

After a user selects a remediation option in the UI, their decision is saved to `historical_decisions.json`, keyed by `issue_type`.

On the next run, Agent 2 receives the last 5 decisions for the same issue type as part of its prompt:

```
Historical decisions for this issue type:
1. Column: loan_amnt; Chosen Action: Impute with Mean; Confidence: 85; ...
2. Column: funded_amnt; Chosen Action: Cap outliers to upper limit; ...
```

This means the system **learns from past human decisions** — if your team consistently prefers mean imputation over median for null spikes, Agent 2 will start weighting that preference in future suggestions.

```python
def format_historical_context(issue_type, history):
    records = history.get(issue_type, [])
    # Returns last 5 decisions for this issue type
    # Fed directly into Agent 2 prompt
```

---

## Data Flow

```
Input:  issues_output.json        (from Spark detector)
        df_shape.json             (total rows + columns)
        historical_decisions.json (past human choices, may be empty on first run)

Output: issues_with_suggestions.json
        └── quality_score: { before, after, delta }
        └── issues[]: sorted by priority_score descending
            ├── input:     original detected issue
            ├── diagnosis: root_cause, business_impact, priority_score, affected_rows_percent
            └── remediation: suggestions[option 1, 2, 3]
```

**Quality score logic:**
```
before = % of rows NOT affected by any detected issue
after  = 100 (assumes all issues fully resolved)
delta  = after - before
```

---

## Example Input / Output

**Input** (from Spark detector):
```json
{
  "column": "loan_amnt",
  "issue_type": "Null Spike",
  "severity": "HIGH",
  "detail": "Null rate: 40% (903898/2260701 rows)",
  "sample_values": "[11000.0, 25000.0, 1500.0]"
}
```

**Output** (from agent pipeline):
```json
{
  "quality_score": { "before": 50.0, "after": 100.0, "delta": 50.0 },
  "issues": [{
    "diagnosis": {
      "root_cause": "Possible data entry errors in the loan application process.",
      "business_impact": "High null rates impair risk modeling and lending decisions.",
      "priority_score": 8,
      "affected_rows_percent": 40.0
    },
    "remediation": {
      "suggestions": [
        { "option": 1, "action": "Impute with Mean", "confidence": 85, "pyspark_code": "..." },
        { "option": 2, "action": "Impute with Median", "confidence": 65, "pyspark_code": "..." },
        { "option": 3, "action": "Decline changes", "confidence": 100, "pyspark_code": "# No change applied to df" }
      ]
    }
  }]
}
```

---

## Repo Structure

```
big_data_team_6/
│
├── src/
│   └── llm/
│       └── suggestion.py        ← My contribution: two-agent pipeline + history loop
│
├── notebooks/                   ← EDA and Spark pipeline (teammates)
├── data/
│   ├── issues_output.json       ← Input: detected issues from Spark
│   ├── df_shape.json            ← Input: dataset dimensions
│   ├── historical_decisions.json← Feedback loop: grows with each user decision
│   └── issues_with_suggestions.json ← Output: full agent results
│
├── flyer/                       ← Project presentation materials
├── requirements.txt
└── README.md
```

---

## Tech Stack

`Python` · `GPT-4o-mini` · `OpenAI API` · `Apache Spark` · `Snowflake` · `Streamlit`
