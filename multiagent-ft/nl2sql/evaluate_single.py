"""Evaluate single-agent LoRA model trained with single_finetune.py on Bird-SQL dev set."""
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sqlglot
import argparse, json, torch, re
import common
import sqlite3

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="../../spider/dev.json")
# parser.add_argument("--data", default="../../Bird-SQL/dev_20240627/dev.json")
parser.add_argument("--db_root", default="../../spider/database",
                    help="Folder that holds <db_id>/<db_id>.sqlite files")
parser.add_argument("--ckpt", default="./out_single_lora/final", help="Path to saved LoRA adapter")
parser.add_argument("--base", default="defog/sqlcoder-7b-2", help="Base model used during training")
parser.add_argument("--limit", type=int, default=100)
args = parser.parse_args()

##############################
# Load model #################
##############################

print(f"⏳ Loading base model: {args.base}")
base = AutoModelForCausalLM.from_pretrained(args.base, load_in_4bit=True, device_map="auto")
print(f"⏳ Loading PEFT adapter: {args.ckpt}")
model = PeftModel.from_pretrained(base, "schaturv/sqlcoder_spider_simple_prompt", adapter_name="single_agent").eval()
model.set_adapter("single_agent")
print(f"Active adapters: {model.active_adapters}")

tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
tok.pad_token = tok.eos_token  # in case it's missing

##############################
# Dataset ####################
##############################

DEV_ROOT = Path(args.data).parent

# removing Arrow usage for processing nested json in Spider
with open(args.data, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

for ex in raw_data:
    ex["SQL"] = ex["query"]
    del ex["query_toks"]
    del ex["query_toks_no_value"]
    del ex["question_toks"]
    del ex["sql"]

ds = Dataset.from_list(raw_data)

# ds = load_dataset("json", data_files=args.data)["train"]
ds = ds.map(lambda ex: common.attach_schema_json(ex, DEV_ROOT))

##############################
# Generation #################
##############################

@torch.no_grad()
def gen_sql(example):
    prompt = common.build_single_code_repr_prompt(example)
    toks = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**toks, max_new_tokens=128)
    decoded = tok.decode(out[0], skip_special_tokens=True)
    sql_part = decoded[len(prompt):].strip()
    match = re.search(r"select\b.*?(?=\n|\/\*|;|\Z)", sql_part, re.S)
    sql = match.group(0).strip() if match else "" # sql = sql_part.split(";")[0].strip() + ";"
    print(f"\nPROMPT =====\n{prompt}\n\nGEN =====\n{sql}\n")
    # formatted = decoded.split("### SQL:")[-1].strip()
    # print(f"Response for {prompt}: ", formatted)
    return sql

def is_valid(sql):
    try:
        sqlglot.parse_one(sql, dialect="sqlite")
        return True
    except:
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

##############################
# Run evaluation #############
##############################

results = []
print(f"⚙️  Evaluating on {args.limit} examples...")

for ex in ds.select(range(args.limit)):
    pred = gen_sql(ex)
    gold = ex["SQL"].strip()
    exact = pred.lower().strip(";").strip() == gold.lower()
    valid = is_valid(pred)
    pred_rows = execute_sql(ex["db_id"], pred) if valid else None
    gold_rows = execute_sql(ex["db_id"], gold)
    exec_match = (pred_rows is not None) and (pred_rows == gold_rows)

    results.append({
        "question": ex["question"],
        "pred": pred,
        "gold": gold,
        "exact": exact,
        "valid": valid,
        "exec_match": exec_match,
    })

# Aggregates
agg = {
    "exact_match": sum(r["exact"] for r in results) / len(results),
    "valid_sql": sum(r["valid"] for r in results) / len(results),
    "execution_match": sum(r["exec_match"] for r in results) / len(results)
}

print(json.dumps(agg, indent=2))

with open("result_jsons/evaluation_results_with_EA.json", "w") as f:
    json.dump({"aggregates": agg, "examples": results}, f, indent=2)

print("Saved to evaluation_results_with_EA.json")
