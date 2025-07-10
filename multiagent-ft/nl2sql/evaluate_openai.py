from pathlib import Path
import argparse, json, sqlite3, sqlglot, re
from datasets import Dataset
from openai import OpenAI
import common

# ─── OpenAI Setup ───────────────────────────────────────────────────────────
client = OpenAI()

# ─── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="../../spider/dev.json")
parser.add_argument("--db_root", default="../../spider/database")
parser.add_argument("--limit", type=int, default=500)
args = parser.parse_args()

# ─── Dataset ─────────────────────────────────────────────────────────────────
DEV_ROOT = Path(args.data).parent
with open(args.data, "r", encoding="utf-8") as f:
    raw = json.load(f)

for ex in raw:
    ex["SQL"] = ex["query"]
    for k in ["query_toks", "query_toks_no_value", "question_toks", "sql"]:
        ex.pop(k, None)

ds = Dataset.from_list(raw)
ds = ds.map(lambda e: common.attach_schema_json(e, DEV_ROOT))

# ─── Helpers ─────────────────────────────────────────────────────────────────
def call_openai(prompt: str, iter, system_msg="You are a helpful assistant", max_tokens=512):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens
    )
    print(f"\n{iter}\n\nPROMPT =====\n{prompt}\n\nGEN =====\n{response.choices[0].message.content.strip()}\n")
    return response.choices[0].message.content.strip()

def clean_sql(raw: str) -> str:
    # Extract SQL up to newline or comment or semicolon
    m = re.search(r"(?is)\bselect\b.*?(?=\n|;|\/\*|\Z)", raw)
    sql = (m.group(0).strip() if m else "").rstrip(";")

    # Fix unbalanced quotes/backticks
    if sql.count('"') % 2 != 0:
        sql = sql.rsplit('"', 1)[0]
    if sql.count("'") % 2 != 0:
        sql = sql.rsplit("'", 1)[0]
    if sql.count("`") % 2 != 0:
        sql = sql.rsplit("`", 1)[0]

    return sql.strip() + ";"

def valid_sql(sql):
    try:
        sqlglot.parse_one(sql, dialect="sqlite")
        return True
    except sqlglot.errors.ParseError:
        return False

def execute_sql(db_id, sql):
    db_file = Path(args.db_root) / db_id / f"{db_id}.sqlite"
    try:
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return rows
    except sqlite3.Error:
        return None
    finally:
        conn.close()

# ─── Multi-Agent Execution ───────────────────────────────────────────────────
results = []
print(f"⚙️  Evaluating on {args.limit or 'ALL'} examples…")
iter = 0
for ex in ds.select(range(args.limit or len(ds))):
    # Step 1: Generate Query Plan
    plan_prompt = common.build_query_plan_prompt(ex)
    query_plan = call_openai(plan_prompt, iter, system_msg="You are an expert query planner. Output only the query plan.")
    
    # Step 2: Generate SQL from Query Plan
    sql_prompt = common.build_sql_from_plan_prompt(ex, query_plan)
    sql = call_openai(sql_prompt, iter, system_msg="You are a SQLite expert. Output only the SQL query.")
    iter += 1
    # Clean SQL
    sql = sql.strip().split(";")[0].strip() + ";"
    sql = clean_sql(sql)
    
    # Gold SQL
    gold = ex["SQL"].strip()

    exact = sql.strip(";").strip().lower() == gold.lower()
    valid = valid_sql(sql)

    pred_rows = execute_sql(ex["db_id"], sql) if valid else None
    gold_rows = execute_sql(ex["db_id"], gold)
    exec_match = (pred_rows is not None) and (pred_rows == gold_rows)

    results.append({
        "question": ex["question"],
        "query_plan": query_plan,
        "pred": sql,
        "gold": gold,
        "exact": exact,
        "valid": valid,
        "exec_match": exec_match
    })

# ─── Aggregates ──────────────────────────────────────────────────────────────
agg = {
    "exact_match": sum(r["exact"] for r in results) / len(results),
    "valid_sql": sum(r["valid"] for r in results) / len(results),
    "execution_match": sum(r["exec_match"] for r in results) / len(results)
}

print(json.dumps(agg, indent=2))
with open("result_jsons/evaluation_results_openai.json", "w") as f:
    json.dump({"aggregates": agg, "examples": results}, f, indent=2)

print("✅ Saved to evaluation_results_openai.json")
