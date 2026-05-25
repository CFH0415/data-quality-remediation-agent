from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, when, trim
from pyspark.sql.functions import expr
from pyspark.sql.types import DoubleType
import json
import os
import sys
from pathlib import Path
from databricks.connect.session import DatabricksSession


# ── Start Spark ──────────────────────────────────────────────────────────────
spark = DatabricksSession.builder.serverless().getOrCreate()


# ── Helper: parse → numeric ───────────────────────────────
def parse_emp_length(df, col_name="emp_length"):
    return df.withColumn(
        col_name + "_numeric",
        when(trim(col(col_name)) == "10+ years", 10.0)
        .when(trim(col(col_name)).rlike(r"^< *1 year$"), 0.0)
        .when(
            trim(col(col_name)).rlike(r"^(\d+) years?$"),
            regexp_extract(trim(col(col_name)), r"^(\d+)", 1).cast(DoubleType())
        )
        .otherwise(None)
    )

def coerce_numeric_columns(df, columns):
    for c in columns:
        if c in df.columns:
            df = df.withColumn(
                c,
                expr(f"try_cast(regexp_replace(cast(`{c}` as string), '[^0-9.\\-]', '') as double)")
            )
    return df

def run_detector(input_mode=None, input_source=None):
    # ── Load data ────────────────────────────────────────────────────────────
    LOCAL_MODE = not os.path.exists("/dbfs")

    if input_mode is None or input_source is None:
        if LOCAL_MODE:
            # Local VS Code run
            # Default: local demo CSV
            if len(sys.argv) >= 3:
                input_mode = sys.argv[1]      # "file" or "table"
                input_source = sys.argv[2]
            else:
                input_mode = "file"
                input_source = "data/demo_lendingclub.csv"
        else:
            # Databricks run
            # Default: demo table unless app passes something else
            if len(sys.argv) >= 3:
                input_mode = sys.argv[1]
                input_source = sys.argv[2]
            else:
                input_mode = "table"
                input_source = "workspace.team6.demo_lendingclub"

    if input_mode == "table":
        df = spark.table(input_source)
    elif input_mode == "file":
        df = spark.read.csv(input_source, header=True, inferSchema=True)
    else:
        raise ValueError(f"Unsupported input_mode: {input_mode}")

    print(f"[OK] Step 1: Data loaded successfully from {input_mode}: {input_source}")

    total = df.count()
    issues = []

    # ── Parse ────────────────────────────────────────────────────
    df = parse_emp_length(df)
    print("[OK] Step 2: emp_length parsed to numeric (emp_length_numeric)")
   
    df = coerce_numeric_columns(
        df,
        ["loan_amnt", "funded_amnt", "funded_amnt_inv", "int_rate", "annual_inc"]
    )

    # ── Detection 1: Null Spike ─────────────────────────────────────────────
    print("\n[SCAN] Detecting Null Spikes...")

    NULL_COLS = [
        "loan_amnt", "funded_amnt", "int_rate", "grade",
        "emp_length", "annual_inc", "loan_status", "purpose",
        "revol_util", "tot_cur_bal"
    ]

    for col_name in NULL_COLS:
        if col_name not in df.columns:
            print(f"  [SKIP]  {col_name}: column not found, skipping")
            continue

        null_count = df.filter(col(col_name).isNull()).count()
        null_rate = null_count / total

        if null_rate > 0.3:
            sample = (
                df.select(col_name)
                .dropna()
                .limit(3)
                .toPandas()[col_name]
                .tolist()
            )
            issues.append({
                "column": col_name,
                "issue_type": "Null Spike",
                "severity": "HIGH",
                "detail": f"Null rate: {null_rate:.0%} ({null_count}/{total} rows)",
                "sample_values": str(sample)
            })
            print(f"  [WARN]  {col_name}: null rate = {null_rate:.0%}")
        else:
            print(f"  [OK] {col_name}: null rate = {null_rate:.0%}, within threshold")

    # ── Detection 2: Statistical Outlier ────────────────────────────────────
    print("\n[SCAN] Detecting Statistical Outliers (IQR)...")

    NUMERIC_COLS = [
        "loan_amnt",
        "funded_amnt",
        "funded_amnt_inv",
        "int_rate",
        "emp_length_numeric",
        "annual_inc",
    ]

    for col_name in NUMERIC_COLS:
        if col_name not in df.columns:
            print(f"  [SKIP]  {col_name}: column not found, skipping")
            continue

        quantiles = df.select(col_name).dropna().approxQuantile(col_name, [0.25, 0.75], 0.01)
        if len(quantiles) < 2:
            print(f"  [SKIP]  {col_name}: insufficient quantiles")
            continue

        q1, q3 = quantiles
        iqr = q3 - q1
        if iqr == 0:
            print(f"  [SKIP]  {col_name}: IQR is 0")
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outlier_count = df.filter(
            col(col_name).isNotNull() &
            ((col(col_name) < lower) | (col(col_name) > upper))
        ).count()

        outlier_rate = outlier_count / total

        if outlier_rate > 0.05:
            sample = (
                df.filter((col(col_name) < lower) | (col(col_name) > upper))
                .select(col_name)
                .limit(5)
                .toPandas()[col_name]
                .tolist()
            )
            issues.append({
                "column": col_name,
                "issue_type": "Statistical Outlier",
                "severity": "HIGH" if outlier_rate > 0.20 else "MEDIUM",
                "detail": (
                    f"{outlier_count} outlier(s) ({outlier_rate:.1%} of rows). "
                    f"IQR bounds: [{lower:.2f}, {upper:.2f}]"
                ),
                "sample_values": str(sample)
            })
            print(
                f"  [WARN]  {col_name}: {outlier_count} outliers ({outlier_rate:.1%}) "
                f"| bounds [{lower:.2f}, {upper:.2f}]"
            )
        else:
            print(f"  [OK] {col_name}: clean ({outlier_count} outliers, below threshold)")

    # ── Detection 3: Format Inconsistency ───────────────────────────────────
    print("\n[SCAN] Detecting Format Inconsistencies...")
    if "issue_d" in df.columns:
        iso_pattern = r"^[A-Za-z]{3}-\d{4}$"
        us_pattern = r"^\d{2}/\d{4}$"

        iso_count = df.filter(regexp_extract(col("issue_d"), iso_pattern, 0) != "").count()
        us_count = df.filter(regexp_extract(col("issue_d"), us_pattern, 0) != "").count()

        if iso_count > 0 and us_count > 0:
            sample = df.select("issue_d").limit(5).toPandas()["issue_d"].tolist()
            issues.append({
                "column": "issue_d",
                "issue_type": "Format Inconsistency",
                "severity": "MEDIUM",
                "detail": (
                    f"Mixed formats: {iso_count} rows as YYYY-MM-DD, "
                    f"{us_count} rows as MM/DD/YYYY"
                ),
                "sample_values": str(sample)
            })
            print("  [WARN]  issue_d: mixed date formats detected")
        else:
            print("  [OK] issue_d: format consistent, no mixed formats detected")
    else:
        print("  [SKIP]  issue_d: column not found, skipping")

    # ── Compute Data Quality Score ──────────────────────────────────────────
    deductions = {"HIGH": 8, "MEDIUM": 4, "LOW": 2}
    raw_score = 100 - sum(deductions.get(i["severity"], 4) for i in issues)
    quality_score = max(0, min(100, raw_score))

    print(f"\n[SCORE] Data Quality Score: {quality_score}/100  |  {len(issues)} issue(s) found")

    # ── Output ──────────────────────────────────────────────────────────────
    print(f"\n[OK] Detection complete — {len(issues)} issue(s) found")
    for idx, issue in enumerate(issues):
        print(f" Issue {idx + 1}: [{issue['severity']}] {issue['column']} - {issue['issue_type']}")

    if LOCAL_MODE:
        output_dir = Path("data")
    else:
        output_dir = Path("/Volumes/workspace/team6/data")

    output_dir.mkdir(parents=True, exist_ok=True)

    output_issues = output_dir / "issues_output.json"
    output_quality = output_dir / "quality_score.json"
    output_shape = output_dir / "df_shape.json"

    with open(output_issues, "w", encoding="utf-8") as f:
        json.dump(issues, f, indent=2)

    with open(output_quality, "w", encoding="utf-8") as f:
        json.dump({"quality_score": quality_score}, f, indent=2)

    with open(output_shape, "w", encoding="utf-8") as f:
        json.dump({"total_rows": total, "total_columns": len(df.columns)}, f, indent=2)

    print("\n[OK] issues_output.json saved")
    print("[OK] quality_score.json saved")
    print("[OK] df_shape.json saved")

    return {
        "issues": issues,
        "quality_score": quality_score,
        "total_rows": total,
        "total_columns": len(df.columns),
    }


if __name__ == "__main__":
    run_detector()