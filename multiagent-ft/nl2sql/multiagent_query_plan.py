from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, TaskType
import json, torch, common

BASE_MODEL = "deepseek-ai/deepseek-coder-6.7b-instruct"
TRAIN_ROOT = Path("../../spider")
DATA_FILE = TRAIN_ROOT / "train_spider.json"
PLAN_OUT = "./out_plan_agent"
SQL_OUT = "./out_sql_agent"

# Tokenizer
tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, padding="max_length")
tok.pad_token = tok.eos_token

# Load & preprocess dataset
with open(DATA_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

for ex in raw_data:
    ex["SQL"] = ex["query"]
    del ex["query_toks"], ex["query_toks_no_value"], ex["question_toks"], ex["sql"]

ds = Dataset.from_list(raw_data)
ds = ds.map(lambda ex: common.attach_schema_json(ex, TRAIN_ROOT))

def mask_after_prompt(prompt_len, ids):
    labels = ids.copy()
    labels[:prompt_len] = [-100] * prompt_len
    return labels

data_collator = DataCollatorForLanguageModeling(
# data_collator = DataCollatorForSeq2Seq(
    tokenizer=tok,
    mlm=False,
    return_tensors="pt",
)


##################################
# Agent 1: Query Plan Generator ##
##################################

plan_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "v_proj"]
)
plan_model = get_peft_model(
    AutoModelForCausalLM.from_pretrained(BASE_MODEL, load_in_4bit=True),
    plan_cfg
).to("cuda")

def tok_plan(ex):
    gold_plan = ex.get("query_plan", "")  # must be in data
    prompt = common.build_query_plan_prompt(ex)
    packed = tok(prompt + gold_plan, truncation=True, max_length=512, padding="max_length")
    prompt_len= len(tok(prompt, truncation=True, max_length=512, add_special_tokens=False)["input_ids"])
    labels = packed["input_ids"].copy()
    labels[:prompt_len] = [-100] * prompt_len
    packed["labels"] = labels
    return packed

    # enc["labels"] = mask_after_prompt(len(tok(prompt)["input_ids"]), enc["input_ids"])
    # return enc

torch.cuda.empty_cache()
plan_ds = ds.map(tok_plan, remove_columns=ds.column_names)
Trainer(
    model=plan_model,
    args=TrainingArguments(output_dir=PLAN_OUT, per_device_train_batch_size=4,
                           num_train_epochs=1, learning_rate=1e-4, fp16=True,
                           logging_steps=50, save_steps=10000, report_to="none"),
    train_dataset=plan_ds, data_collator=data_collator
).train()
plan_model.save_pretrained(f"{PLAN_OUT}/final")
plan_model.push_to_hub("schaturv/spider_plan_agent")

##################################
# Agent 2: SQL Generator #########
##################################

sql_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    task_type=TaskType.CAUSAL_LM, target_modules=["q_proj", "v_proj"]
)
sql_model = get_peft_model(
    AutoModelForCausalLM.from_pretrained(BASE_MODEL, load_in_4bit=True),
    sql_cfg
).to("cuda")

def tok_sql(ex):
    query_plan = ex.get("query_plan", "")  # gold plan
    prompt = common.build_sql_from_plan_prompt(ex, query_plan)
    packed = tok(prompt + ex["SQL"], truncation=True, max_length=512, padding="max_length")
    prompt_len= len(tok(prompt, truncation=True, max_length=512, add_special_tokens=False)["input_ids"])
    labels = packed["input_ids"].copy()
    labels[:prompt_len] = [-100] * prompt_len
    packed["labels"] = labels
    return packed

    # enc["labels"] = mask_after_prompt(len(tok(prompt)["input_ids"]), enc["input_ids"])
    # return enc
'''
torch.cuda.empty_cache()
sql_ds = ds.map(tok_sql, remove_columns=ds.column_names)
Trainer(
    model=sql_model,
    args=TrainingArguments(output_dir=SQL_OUT, per_device_train_batch_size=4,
                           num_train_epochs=1, learning_rate=1e-4, fp16=True,
                           logging_steps=50, save_steps=10000, report_to="none"),
    train_dataset=sql_ds, data_collator=data_collator
).train()
sql_model.save_pretrained(f"{SQL_OUT}/final")
sql_model.push_to_hub("schaturv/spider_sql_from_plan_agent")
'''
