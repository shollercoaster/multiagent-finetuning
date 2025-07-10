from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sqlglot, sqlite3, argparse, json, torch, textwrap, re
import common

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="../../spider/dev.json")
parser.add_argument("--db_root", default="../../spider/database")
parser.add_argument("--plan_ckpt", default="schaturv/spider_plan_agent")
parser.add_argument("--sql_ckpt", default="schaturv/spider_sql_from_plan_agent")
parser.add_argument("--base", default="deepseek-ai/deepseek-coder-6.7b-instruct")
parser.add_argument("--limit", type=int, default=100)
parser.add_argument("--save", default="result_jsons/eval_results_plan_sql.json")
parser.add_argument("--run_exec", action="store_true", default=True, help="Enable execution match eval")
args = parser.parse_args()

##############################
# Load models ################
##############################

print("⏳ Loading base:", args.base)
base_plan = AutoModelForCausalLM.from_pretrained(args.base, load_in_4bit=True, device_map="auto").eval()
base_sql = AutoModelForCausalLM.from_pretrained(args.base, load_in_4bit=True, device_map="auto").eval()

print("⏳ Loading adapters")
plan_model = PeftModel.from_pretrained(base_plan, args.plan_ckpt, adapter_name="plan").eval()
sql_model = PeftModel.from_pretrained(base_sql, args.sql_ckpt, adapter_name="sql").eval()
plan_model.set_adapter("plan")
sql_model.set_adapter("sql")

tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
tok.pad_token = tok.eos_token

##############################
# Dataset ####################
##############################

with open(args.data, "r", encoding="utf-8") as f:
    raw = json.load(f)

for ex in raw:
    ex["SQL"] = ex["query"]
    del ex["query_toks"], ex["query_toks_no_value"], ex["question_toks"], ex["sql"]

ds = Dataset.from_list(raw)
ds = ds.map(lambda e: common.attach_schema_json(e, Path(args.data).parent))

if args.limit > 0:
    ds = ds.select(range(args.limit))

##############################
# Generation #################
##############################

@torch.no_grad()
def generate(model, prompt, max_new_tokens=512):
    toks = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**toks, max_new_tokens=max_new_tokens)
    dec = tok.decode(out[0], skip_special_tokens=True)
    return dec[len(prompt):].strip()

@torch.no_grad()
def run_query_pipeline(ex, example):
    schema = common._fmt_schema(ex["schema"])
    question = ex["question"]

    # Agent 1 → generate plan
    plan_prompt = common.build_query_plan_prompt(ex)
    query_plan = generate(plan_model, plan_prompt).split("\n")[0].strip()

    # Agent 2 → generate SQL
    sql_prompt = common.build_sql_from_plan_prompt(ex, query_plan)
    raw_sql = generate(sql_model, sql_prompt, max_new_tokens=128)

    # Extract clean SELECT
    m = re.search(r"(?is)select\b.*?(?=\n|\/\*|;|\Z)", raw_sql)
    sql = m.group(0).strip() if m else ""
    print(f"\n{example}\n\nPROMPT =====\n{sql_prompt}\n\nQUERY_PLAN ===== \n{query_plan}\n\nGEN =====\n{sql}\n")
    return query_plan, sql

def is_valid(sql):
    try:
        sqlglot.parse_one(sql, dialect="sqlite")
        return True
    except:
        return False

def execute_sql(db_id, sql):
    db_file = Path(args.db_root) / db_id / f"{db_id}.sqlite"
    if not db_file.exists():
        return None
    try:
        conn = sqlite3.connect(db_file)
        cur = conn.cursor()
        cur.execute(sql)
        return cur.fetchall()
    except:
        return None
    finally:
        conn.close()

##############################
# Evaluation #################
##############################

results = []

# print(f"⚙️ Running eval on {lrgs.limit} examples...")
iter = 0

for ex in ds.select(range(args.limit)):
    plan, pred_sql = run_query_pipeline(ex, iter)
    gold_sql = ex["SQL"].strip()

    exact = pred_sql.strip(";").strip().lower() == gold_sql.lower()
    valid = is_valid(pred_sql)

    iter += 1
    exec_match = None
    if args.run_exec:
        pred_rows = execute_sql(ex["db_id"], pred_sql) if valid else None
        gold_rows = execute_sql(ex["db_id"], gold_sql)
        exec_match = (pred_rows == gold_rows) if pred_rows is not None and gold_rows is not None else False

    results.append({
        "question": ex["question"],
        "query_plan": plan,
        "pred": pred_sql,
        "gold": gold_sql,
        "exact": exact,
        "valid": valid,
        "exec_match": exec_match
    })

# Aggregates
n = len(results)
agg = {
    "exact_match": sum(r["exact"] for r in results) / n,
    "valid_sql": sum(r["valid"] for r in results) / n,
    # "execution_match": sum(r["exec_match"] for r in results) / n
}

if args.run_exec:
    exec_valid = [r["exec_match"] for r in results if r["exec_match"] is not None]
    agg["exec_match"] = sum(exec_valid) / len(exec_valid) if exec_valid else 0.0

print(json.dumps(agg, indent=2))

# Save
Path(args.save).parent.mkdir(exist_ok=True, parents=True)
with open(args.save, "w") as f:
    json.dump({"aggregates": agg, "examples": results}, f, indent=2)
print(f"✅ Saved to {args.save}")
