"""Fine-tune multiple SQL agents on Spider data categorized by SQL subtype (normal, groupBy, orderBy, complex)."""
import os, json, torch, argparse, logging
from pathlib import Path
from datasets import load_dataset, DatasetDict, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from common import attach_schema_json, build_sql_prompt_by_category

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="../../spider/train_spider.json")
parser.add_argument("--schema", default="../../spider/tables.json")
parser.add_argument("--base_model", default="deepseek-ai/deepseek-coder-6.7b-instruct") # "Qwen/Qwen2.5-Coder-3B-Instruct")
parser.add_argument("--output_dir", default="./output")
parser.add_argument("--num_train_epochs", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()

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
with open(args.data, "r", encoding="utf-8") as f:
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
ds = ds.map(lambda ex: attach_schema_json(ex, Path(args.schema).parent))
ds = ds.map(lambda ex: {"text": build_sql_prompt_by_category(ex)})

# Split by category
subtypes = ["normal", "groupby", "orderby", "complex"]
category_data = {t: ds.filter(lambda x: x["category"] == t) for t in subtypes}

# Load model and tokenizer
logger.info(f"🧠 Loading base model: {args.base_model}")
tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True, use_fast=True, padding="max_length")
model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
model = prepare_model_for_kbit_training(model).to("cuda")
collator = DataCollatorForLanguageModeling(tok, mlm=False, return_tensors="pt")

def tokenize_sql_generation(ex):
    prompt = ex["text"]
    packed = tok(prompt, truncation=True, padding="max_length", max_length=512)
    
    prompt_len = len(tok(prompt, truncation=True, add_special_tokens=False, padding=False, max_length=512)["input_ids"])
    labels = packed["input_ids"].copy()
    labels[:prompt_len] = [-100] * prompt_len
    packed["labels"] = labels
    return packed

# LoRA config
lora_cfg = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn", "q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
print("Single sample: ", category_data["groupby"][0])
# Train separate adapter per category
for category in subtypes:
    logger.info(f"🚀 Training agent for category: {category}")
    category_data[category] = category_data[category].map(tokenize_sql_generation, remove_columns=category_data[category].column_names)
    adapter_name = f"lora_{category}"
    agent_model = get_peft_model(model, lora_cfg, adapter_name=adapter_name).to("cuda")
    agent_model.set_adapter(adapter_name)
    agent_model.print_trainable_parameters()
    output_path = Path(args.output_dir) / f"lora_{category}"

    args_out = TrainingArguments(
        output_dir=str(output_path),
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        save_strategy="epoch",
        logging_steps=5,
        learning_rate=5e-5,
        fp16=True,
        report_to="none",
        remove_unused_columns=False
    )

    trainer = Trainer(
        model=agent_model,
        train_dataset=category_data[category],
        tokenizer=tok,
        args=args_out,
        data_collator=collator
    )
    torch.cuda.empty_cache()
    trainer.train()
    agent_model.save_pretrained(output_path)
    adapter_save_path=f"schaturv/{adapter_name}"
    agent_model.push_to_hub(adapter_save_path)
    print(f"lora_{category} model saved to {adapter_save_path}")

logger.info("✅ Finetuning completed.")
