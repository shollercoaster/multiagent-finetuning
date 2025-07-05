"""Finetune a *single* LoRA adapter to map (schema + question) → SQL."""
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling, DataCollatorForSeq2Seq
from peft import LoraConfig, get_peft_model, TaskType
import torch, common
import json

BASE_MODEL = "defog/sqlcoder-7b-2"

# Paths for BirdSQL dataset
# TRAIN_ROOT = Path("../../Bird-SQL/train")
# DATA_FILE = TRAIN_ROOT / "train.json" 

# Paths for Spider dataset
TRAIN_ROOT = Path("../../spider")
DATA_FILE = TRAIN_ROOT / "train_spider.json"

OUTPUT_DIR = "./out_single_lora"

##############################
# Tokeniser ##################
##############################

tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True, padding='max_length')
tok.pad_token = tok.eos_token

##############################
# Dataset ####################
##############################

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
# add schema from train_tables.json

ds = ds.map(lambda ex: common.attach_schema_json(ex, TRAIN_ROOT))

##############################
# Tokenise & mask ############
##############################

def tok_fn(ex):
    prompt = common.build_single_prompt(ex)
    tok_out = tok(prompt + ex["SQL"], truncation=True, max_length=512, padding="max_length")
    prompt_len= len(tok(prompt, truncation=True, max_length=512, add_special_tokens=False)["input_ids"])
    labels = tok_out["input_ids"].copy() # list of ints, not a Tensor
    labels[:prompt_len] = [-100] * prompt_len # -100 masks question tokens so model only learns sql answers
    tok_out["labels"] = labels
    return tok_out

train_ds = ds.map(tok_fn, remove_columns=ds.column_names)

##############################
# Model + training ###########
##############################

base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, load_in_4bit=True, attn_implementation="flash_attention_2" )
lo_cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
                    target_modules=["q_proj", "v_proj"])
model = get_peft_model(base, lo_cfg).to("cuda")

model.gradient_checkpointing_enable()
model.enable_input_require_grads()

data_collator = DataCollatorForLanguageModeling(
# data_collator = DataCollatorForSeq2Seq(
    tokenizer=tok,
    mlm=False,
    return_tensors="pt",
    # model=model, # used for DataCollatorForSeq2Seq (used for encoder-decoder models)
    # label_pad_token_id=-100,
    # pad_to_multiple_of=8,
)

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=16,
    num_train_epochs=1,
    learning_rate=1e-4,
    fp16=True,
    save_steps=10000,
    logging_steps=100,
    report_to="none",
    label_names=["labels"],
    gradient_checkpointing=False,
    gradient_accumulation_steps=1,
)

torch.cuda.empty_cache()
Trainer(model=model, args=args, train_dataset=train_ds, data_collator=data_collator).train()
model.save_pretrained(f"{OUTPUT_DIR}/final")
tok.save_pretrained(f"{OUTPUT_DIR}/final")
