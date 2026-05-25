import json
import os
from anthropic import Anthropic

from dotenv import load_dotenv
load_dotenv("C:\Users\seanc\big_data_team_6\src\.env")

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-6"  # smart + cost-efficient for agentic tasks


def run_agent_1_diagnosis(issue: dict) -> dict:
    """Agent 1: Diagnose the issue — root cause, priority, impact."""
    prompt = f"""You are a data quality analyst. Analyze this data quality issue and return ONLY valid JSON, no preamble.

Issue detected:
{json.dumps(issue, indent=2)}

Return this exact JSON structure:
{{
  "root_cause": "string — concise explanation of why this issue likely occurred",
  "priority_score": <integer 1-10, 10 = most urgent>,
  "affected_records_pct": <float, estimated % of records affected>,
  "urgency": "low|medium|high|critical",
  "business_impact": "string — what downstream effect this has on analytics or models"
}}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        system="You are an expert data quality analyst. Always respond with valid JSON only.",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def run_agent_2_remediation(issue: dict, diagnosis: dict) -> dict:
    """Agent 2: Generate a runnable PySpark fix based on Agent 1's diagnosis."""
    prompt = f"""You are a PySpark data engineer. Given the issue and diagnosis below, generate a runnable PySpark fix.

Original Issue:
{json.dumps(issue, indent=2)}

Agent 1 Diagnosis:
{json.dumps(diagnosis, indent=2)}

Return ONLY valid JSON with this exact structure:
{{
  "fix_description": "string — what the fix does in plain English",
  "pyspark_code": "string — complete, runnable PySpark code. Use 'df' as the DataFrame variable.",
  "confidence": "low|medium|high",
  "caveats": "string — any assumptions or limitations of this fix"
}}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system="You are an expert PySpark engineer. Always respond with valid JSON only. Escape newlines in code strings as \\n.",
        messages=[{"role": "user", "content": prompt}]
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())


def process_issues(input_path: str, output_path: str):
    with open(input_path, "r") as f:
        issues = json.load(f)

    enriched = []
    for i, issue in enumerate(issues):
        print(f"Processing issue {i+1}/{len(issues)}: {issue.get('type', 'unknown')} on column '{issue.get('column', '?')}'")

        try:
            diagnosis = run_agent_1_diagnosis(issue)
            print(f"  ✓ Agent 1 done — priority: {diagnosis.get('priority_score')}/10")

            remediation = run_agent_2_remediation(issue, diagnosis)
            print(f"  ✓ Agent 2 done — confidence: {remediation.get('confidence')}")

            enriched.append({
                **issue,
                "diagnosis": diagnosis,
                "remediation": remediation
            })

        except Exception as e:
            print(f"  ✗ Error on issue {i+1}: {e}")
            enriched.append({
                **issue,
                "diagnosis": {"error": str(e)},
                "remediation": {"error": str(e)}
            })

    with open(output_path, "w") as f:
        json.dump(enriched, f, indent=2)

    print(f"\nDone. Output written to {output_path}")


if __name__ == "__main__":
    process_issues(
        input_path="data/issues_output.json",
        output_path="data/issues_with_suggestions.json"
    )