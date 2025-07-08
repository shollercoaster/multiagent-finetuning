"""
Evaluate two‑agent LoRA model (schema‑agent + sql‑agent) on Spider or Bird‑SQL.

Usage
-----
python evaluate_multi_agents.py \
  --data     ../../spider/dev.json \
  --db_root  ../../spider/database \
  --schema_ckpt ./out_schema_agent/final \
  --sql_ckpt    ./out_sql_agent/final \
  --limit   100
"""
from pathlib import Path
import argparse, json, sqlite3, re, torch, sqlglot
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import common, textwrap
from transformers import StoppingCriteria, StoppingCriteriaList

# adding a custom stopping criterion
class StopOnSection(StoppingCriteria):
    def __init__(self, tokenizer, stop_str="\n###"):
        self.stop_ids = tokenizer(stop_str, add_special_tokens=False)["input_ids"]

    def __call__(self, input_ids: torch.LongTensor, scores, **kwargs):
        return input_ids[0, -len(self.stop_ids):].tolist() == self.stop_ids

# stopper = StoppingCriteriaList([StopOnSection(tok, "\n###")])

# ─── CLI ────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="../../spider/dev.json",
                    help="dev.json or dev_20240627/dev.json")
parser.add_argument("--db_root", default="../../spider/database",
                    help="Folder with <db_id>/<db_id>.sqlite files")
parser.add_argument("--schema_ckpt", default="schaturv/spider_schema_agent",
                    help="Path to trained schema‑agent adapter")
parser.add_argument("--sql_ckpt", default="schaturv/spider_sql_agent",
                    help="Path to trained sql‑agent adapter")
parser.add_argument("--base", default="defog/sqlcoder-7b-2")
parser.add_argument("--limit", type=int, default=100)
args = parser.parse_args()

# ─── Load base + adapters ──────────────────────────────────────────────────
tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
tok.pad_token = tok.eos_token
stopper = StoppingCriteriaList([StopOnSection(tok, "\n###")])
def load_adapter(path, adapter_name):
    base = AutoModelForCausalLM.from_pretrained(args.base,
                                               load_in_4bit=True,
                                               device_map="auto").eval()
    return PeftModel.from_pretrained(base, path, adapter_name=adapter_name).eval()

print("⏳ Loading schema‑agent adapter")
schema_model = load_adapter(args.schema_ckpt, "schema")
schema_model.set_adapter("schema")
print("⏳ Loading sql‑agent adapter")
sql_model    = load_adapter(args.sql_ckpt, "sql")
sql_model.set_adapter("sql")
# ─── Load & prep dataset ───────────────────────────────────────────────────
DEV_ROOT = Path(args.data).parent
with open(args.data, "r", encoding="utf-8") as f:
    raw = json.load(f)

for ex in raw:
    ex["SQL"] = ex.get("query", ex.get("SQL", ""))   # unify key
    for k in ["query_toks", "query_toks_no_value", "question_toks", "sql"]:
        ex.pop(k, None)

ds = Dataset.from_list(raw)
ds = ds.map(lambda e: common.attach_schema_json(e, DEV_ROOT))

# ─── Helpers ───────────────────────────────────────────────────────────────
@torch.no_grad()
def run_schema(ex):
    p = common.build_schema_prompt(ex)
    ids = tok(p, return_tensors="pt").to(schema_model.device)
    dec = tok.decode(schema_model.generate(**ids, max_new_tokens=64)[0],
                     skip_special_tokens=True)
    body = dec[len(p):] if dec.startswith(p) else dec
    # keep first line (needed cols)
    return body.strip().split("\n")[0].strip()

@torch.no_grad()
def run_sql(ex, needed):
    p = common.build_sql_prompt(ex, needed)
    stopper = StoppingCriteriaList([StopOnSection(tok, "\n###")])
    ids = tok(p, return_tensors="pt").to(sql_model.device)
    dec = tok.decode(sql_model.generate(**ids, stopping_criteria=stopper, max_new_tokens=128)[0],
                     skip_special_tokens=True)
    body = dec[len(p):] if dec.startswith(p) else dec
    m = re.search(r"(?is)select\b.*?(?=\/\*|;|\Z)", body)
    sql = (m.group(0).strip() if m else "").rstrip(";")
    print(textwrap.dedent(f"""
        ─ PROMPT ─
        {p}

        ─ RAW ─
        {dec}

        ─ SQL ─
        {sql}
    """))

    return sql + ";"

def valid_sql(sql):
    try:
        sqlglot.parse_one(sql, dialect="sqlite"); return True
    except sqlglot.errors.ParseError: return False

def execute_sql(db_id, sql):
    db_file = Path(args.db_root) / db_id / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(db_file) as conn:
            cur = conn.cursor(); cur.execute(sql); return cur.fetchall()
    except sqlite3.Error:
        return None

# ─── Evaluation loop ───────────────────────────────────────────────────────
results = []
for ex in ds:
    needed = run_schema(ex)
    pred   = run_sql(ex, needed)
    gold   = ex["SQL"].strip()

    exact = pred.strip(";").strip().lower() == gold.lower()
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
Path("result_jsons/evaluation_results_multi.json").write_text(
    json.dumps({"aggregates": agg, "examples": results}, indent=2)
)
print("✅ Saved to evaluation_results_multi.json")
