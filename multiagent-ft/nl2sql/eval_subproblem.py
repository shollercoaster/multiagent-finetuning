import json, sqlite3
import torch, argparse, logging
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from peft import PeftModel
from common import attach_schema_json, build_sql_prompt_by_category
# from multiagent_subproblem import data_loading, categorize, normalize_query_structure

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#parser = argparse.ArgumentParser()
#parser.add_argument("--schema", default="../../spider/tables.json")

# === Configuration ===
BASE_MODEL = "deepseek-ai/deepseek-coder-6.7b-instruct"
ADAPTER_ROOT = Path("./output")
DEV_FILE = Path("../../spider/dev.json")
SCHEMA_PATH = Path("../../spider/tables.json")
DEVICE = "cuda"
DB_DIR = Path("../../spider/database")  # Path to folder containing SQLite DBs

SUBTYPES = ["normal", "groupby", "orderby", "complex"]
ADAPTER_MAP = {cat: f"schaturv/lora_{cat}" for cat in SUBTYPES}  # Or load local path

# === Load tokenizer ===
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
tok.pad_token = tok.eos_token

# ds = data_loading(DEV_FILE)
# Categorize SQL subtype
def categorize(sample):
    example = sample["sql"]
    query = sample.get("query", "").lower()
    if any([example["groupBy"], example["having"], example["intersect"],
            example["union"], example["except"], example["limit"] is not None]):
        if example["groupBy"] and example["orderBy"]:
            sample["category"] = "complex"
        elif example["groupBy"] or "groupby" in query:
            sample["category"] = "groupby"
        elif example["orderBy"] or "orderby" in query:
            sample["category"] = "orderby"
        else:
            sample["category"] = "complex"
    else:
        sample["category"] = "normal"
    return sample

def normalize_query_structure(entry):
    # Ensure these fields are lists or set to empty list
    for key in ["groupBy", "having", "orderBy", "intersect", "union", "except"]:
        value = entry.get(key)
        if not isinstance(value, list):
            entry[key] = []
    if entry.get("limit") is None:
        entry["limit"] = 0  # or keep it None if needed, but must be consistent
    return entry

# Load and categorize data
logger.info("🔍 Loading and categorizing data...")

def data_loading(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for ex in data:
        del ex["query_toks"]
        del ex["query_toks_no_value"]
        del ex["question_toks"]
        ex.update(categorize(ex))
        del ex["sql"]

    data = [normalize_query_structure(ex) for ex in data]
    # Attach schema and convert to dataset
    logger.info("📦 Attaching schema...")
    # ds = DatasetDict({"train": Dataset.from_list(data)})
    ds = Dataset.from_list(data)
    ds = ds.map(lambda ex: attach_schema_json(ex, SCHEMA_PATH))
    ds = ds.map(lambda ex: {"text": build_sql_prompt_by_category(ex)})
    return ds

ds = data_loading(DEV_FILE)

# === Load models per subtype ===
agents = {}
for cat in SUBTYPES:
    print(f"🧠 Loading adapter for: {cat}")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto", torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, ADAPTER_MAP[cat]).to(DEVICE)
    print(model.active_adapters)
    model.eval()
    pipe = TextGenerationPipeline(model=model, tokenizer=tok, device=0)
    agents[cat] = pipe

# === Inference & Evaluation ===
predictions = []
exact_match = 0
exec_match = 0

def exec_sql(query: str, db_id: str):
    """Execute SQL query against Spider database and return rows"""
    db_path = DB_DIR / db_id / f"{db_id}.sqlite"
    if not db_path.exists():
        return None
    try:
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
        return sorted(rows)
    except Exception as e:
        return None

iter = 0
for sample in ds:
    category = sample["category"]
    prompt = sample["text"]
    gold = sample["SQL"].strip()
    db_id = sample["db_id"]

    gen = agents[category](
        prompt,
        max_new_tokens=512,
        do_sample=False,
        return_full_text=False,
        pad_token_id=tok.pad_token_id
    )[0]["generated_text"].strip()

    print(f"\n{iter}\n\nPROMPT =====\n{prompt}\n\n\nGEN =====\n{gen}\n")
    iter += 1

    predictions.append({
        "question": sample["question"],
        "db_id": sample["db_id"],
        "gold": gold,
        "pred": gen,
        "category": category,
        "gold_exec": gold_exec,
        "pred_exec": pred_exec

    })

    if gen.lower().strip(";") == gold.lower().strip(";"):
        exact_match += 1

# === Save & Report ===
Path("eval_outputs").mkdir(exist_ok=True)
with open("result_jsons/eval_multiagent_subproblems.json", "w") as f:
    json.dump(predictions, f, indent=2)

print(f"\n✅ Evaluation complete.")
print(f"Total examples: {len(ds)}")
print(f"Exact match accuracy: {exact_match / len(ds):.3f}")
print(f"Execution accuracy:   {exec_match / len(ds):.3f}")
