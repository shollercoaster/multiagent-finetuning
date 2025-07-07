"""Train *two* LoRA adapters: one picks needed columns, the other writes SQL."""
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
import torch, common
import json

BASE_MODEL = "defog/sqlcoder-7b-2"
TRAIN_ROOT = Path("../../spider")
DATA_FILE = TRAIN_ROOT / "train_spider.json"
SCHEMA_OUT = "./out_schema_agent"
SQL_OUT = "./out_sql_agent"

tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, padding="max_length")
tok.pad_token = tok.eos_token

# removing Arrow usage for processing nested json in Spider
with open(DATA_FILE, "r", encoding="utf-8") as f:
    raw_data = json.load(f)

for ex in raw_data:
    ex["SQL"] = ex["query"]
    del ex["query_toks"]
    del ex["query_toks_no_value"]
    del ex["question_toks"]
    del ex["sql"]

ds = Dataset.from_list(raw_data)
# ds = load_dataset("json", data_files=str(DATA_FILE))["train"]
ds = ds.map(lambda ex: common.attach_schema_json(ex, TRAIN_ROOT))

##############################
# Helpers ####################
##############################

def mask_after_prompt(example_prompt_len, merged_ids):
    labels = merged_ids.copy(); labels[:example_prompt_len] = [-100]*example_prompt_len; return labels

##############################
# 1. Schema agent ############
##############################

schema_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
                        target_modules=["q_proj", "v_proj"])
schema_model = get_peft_model(AutoModelForCausalLM.from_pretrained(BASE_MODEL, load_in_4bit=True),
                              schema_cfg).to("cuda")

def schema_tok(ex):
    gold = ex.get("evidence", "")
    prompt = common.build_schema_prompt(ex)
    packed = tok(prompt + gold, truncation=True, max_length=1024, padding="max_length")
    packed["labels"] = mask_after_prompt(len(tok(prompt)["input_ids"]), packed["input_ids"])
    return packed

torch.cuda.empty_cache()
schema_ds = ds.map(schema_tok, remove_columns=ds.column_names)
Trainer(model=schema_model,
        args=TrainingArguments(output_dir=SCHEMA_OUT, per_device_train_batch_size=8, num_train_epochs=1,
                               learning_rate=1e-4, fp16=True, save_steps=10000, logging_steps=100, report_to="none", gradient_checkpointing=False, gradient_accumulation_steps=1),
        train_dataset=schema_ds).train()
schema_model.save_pretrained(f"{SCHEMA_OUT}/final")
schema_model.push_to_hub("schaturv/spider_schema_agent")

##############################
# 2. SQL agent ###############
##############################

sql_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
                     target_modules=["q_proj", "v_proj"])
sql_model = get_peft_model(AutoModelForCausalLM.from_pretrained(BASE_MODEL, load_in_8bit=True),
                           sql_cfg).to("cuda")

def sql_tok(ex):
    needed = ex.get("evidence", "")  # gold during training
    prompt = common.build_sql_prompt(ex, needed)
    packed = tok(prompt + ex["SQL"], truncation=True, max_length=1024, padding="max_length")
    packed["labels"] = mask_after_prompt(len(tok(prompt)["input_ids"]), packed["input_ids"])
    return packed

torch.cuda.empty_cache()
sql_ds = ds.map(sql_tok, remove_columns=ds.column_names)
Trainer(model=sql_model,
        args=TrainingArguments(output_dir=SQL_OUT, per_device_train_batch_size=8, gradient_accumulation_steps=1,
                               num_train_epochs=1, learning_rate=1e-4, fp16=True, save_steps=10000, logging_steps=100, gradient_checkpointing=False,
                               report_to="none"),
        train_dataset=sql_ds).train()
sql_model.save_pretrained(f"{SQL_OUT}/final")
sql_model.push_to_hub("schaturv/spider_sql_agent")
