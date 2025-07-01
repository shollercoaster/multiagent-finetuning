import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import load_dataset

# 1. Configuration
MODEL_NAME = "defog/sqlcoder-7b-2"
DATASET_PATH = "../../Bird-SQL/train/train.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2. Initialize Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# 3. Dataset Preparation
def format_multiagent_example(example):
    return {
        "input": (
            f"Schema Context: Database {example['db_id']}\n"
            f"Question: {example['question']}\n"
            f"Evidence: {example.get('evidence', '')}\n"
            f"Generate SQL:"
        ),
        "labels": example["SQL"]  # Key change: unified labels
    }

dataset = load_dataset("json", data_files=DATASET_PATH).map(format_multiagent_example)

def tokenize_function(examples):
    # Tokenize inputs
    tokenized_inputs = tokenizer(
        examples["input"],
        max_length=1024,
        truncation=True,
        padding="max_length"
    )
    
    # Tokenize labels separately
    tokenized_labels = tokenizer(
        examples["labels"],
        max_length=1024,
        truncation=True,
        padding="max_length"
    )
    
    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels": tokenized_labels["input_ids"]  # Key for seq2seq
    }

# Process dataset
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names  # Remove original columns
)

# 4. Data Collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=None,
    pad_to_multiple_of=8
)

# 5. Model Initialization
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16 if DEVICE == "cuda" else torch.float32
).to(DEVICE)

# 6. Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=3,
    logging_steps=50,
    save_strategy="steps",
    save_steps=500,
    eval_strategy="no",  # Disabled for simplicity
    fp16=DEVICE == "cuda",
    remove_unused_columns=False,  # Critical fix
    report_to="none"
)

# 7. Trainer Setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
    tokenizer=tokenizer
)

# 8. Start Training
trainer.train()
