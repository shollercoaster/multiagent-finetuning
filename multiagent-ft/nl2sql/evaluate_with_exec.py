"""Evaluate single‑agent LoRA model (Spider or Bird‑SQL) with exact‑, valid‑,
and execution‑match metrics."""
from pathlib import Path
import argparse, json, sqlite3, torch, sqlglot
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import common
from sqlglot.errors import ParseError, TokenError
import re, textwrap

# ─── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="../../spider/dev.json")
parser.add_argument("--db_root", default="../../spider/database",
                    help="Folder that holds <db_id>/<db_id>.sqlite files")
parser.add_argument("--ckpt", default="./out_single_lora/final",
                    help="Path to saved LoRA adapter")
parser.add_argument("--base", default="defog/sqlcoder-7b-2",
                    help="Base model used during training")
parser.add_argument("--limit", type=int, default=10)
args = parser.parse_args()

# ─── Model ───────────────────────────────────────────────────────────────────
print("⏳ Loading base:", args.base)
base = AutoModelForCausalLM.from_pretrained(
    args.base, load_in_4bit=True, device_map="auto"
).eval()
print("⏳ Loading adapter:", args.ckpt)
model = PeftModel.from_pretrained(base, "schaturv/sqlcoder_spider_simple_prompt", adapter_name="single_agent").eval()
model.set_adapter("single_agent")
print(f"Active adapters: {model.active_adapters}")
tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
tok.pad_token = tok.eos_token

# ─── Dataset ─────────────────────────────────────────────────────────────────
DEV_ROOT = Path(args.data).parent

# Spider JSON has nested objects → read manually
with open(args.data, "r", encoding="utf-8") as f:
    raw = json.load(f)

for ex in raw:
    ex["SQL"] = ex["query"]
    del ex["query_toks"]
    del ex["query_toks_no_value"]
    del ex["question_toks"]
    del ex["sql"]

'''
for ex in raw:
    ex["SQL"] = ex.get("query", ex.get("SQL", ""))  # unify key name
    # prune heavy fields if present
    for k in ["query_toks", "query_toks_no_value", "question_toks", "sql"]:
        ex.pop(k, None)
'''
ds = Dataset.from_list(raw)
ds = ds.map(lambda e: common.attach_schema_json(e, DEV_ROOT))

# ─── Helpers ────────────────────────────────────────────────────────────────
@torch.no_grad()
@torch.no_grad()
def gen_sql(ex):
    prompt = common.build_single_code_repr_prompt(ex)

    # 1. Generate
    toks = tok(prompt, return_tensors="pt").to(model.device)
    out  = model.generate(
        **toks,
        max_new_tokens=256,
        eos_token_id=tok.eos_token_id,
    )
    dec = tok.decode(out[0], skip_special_tokens=True)

    # 2. Remove the prompt (robustly)
    gen_body = dec[len(prompt):] if dec.startswith(prompt) else dec

    # 3. Keep the *first* SELECT …;
    match = re.search(r"(?i)select\b.*?;", gen_body, re.S)
    sql = match.group(0).strip() if match else ""

    print(textwrap.dedent(f"""
        ─ PROMPT ─
        {prompt}

        ─ RAW ─
        {dec}

        ─ SQL ─
        {sql}
    """))
    return sql

def valid_sql(sql: str) -> bool:
    if not sql or sql == ";":
        return False
    try:
        sqlglot.parse_one(sql, dialect="sqlite")
        return True
    except (ParseError, TokenError):
        return False

def execute_sql(db_id, sql):
    db_file = Path(args.db_root) / db_id / f"{db_id}.sqlite"
    conn = sqlite3.connect(db_file)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return rows
    except sqlite3.Error:
        return None
    finally:
        conn.close()

# ─── Evaluation loop ────────────────────────────────────────────────────────
results = []
print(f"⚙️  Evaluating first {args.limit} examples…")

for ex in ds.select(range(args.limit)):
    pred = gen_sql(ex)
    gold = ex["SQL"].strip()
    exact = pred.lower() == gold.lower()
    valid = valid_sql(pred)

    pred_rows = execute_sql(ex["db_id"], pred) if valid else None
    gold_rows = execute_sql(ex["db_id"], gold)
    exec_match = (pred_rows is not None) and (pred_rows == gold_rows)

    results.append({
        "question": ex["question"],
        "pred": pred,
        "gold": gold,
        "exact": exact,
        "valid": valid,
        "exec_match": exec_match
    })

agg = {
    "exact_match"   : sum(r["exact"] for r in results) / len(results),
    "valid_sql"     : sum(r["valid"] for r in results) / len(results),
    "execution_match": sum(r["exec_match"] for r in results) / len(results)
}

print(json.dumps(agg, indent=2))
with open("result_jsons/evaluation_results_with_EA.json", "w") as f:
    json.dump({"aggregates": agg, "examples": results}, f, indent=2)
print("✅  Saved to evaluation_results_with_EA.json")
