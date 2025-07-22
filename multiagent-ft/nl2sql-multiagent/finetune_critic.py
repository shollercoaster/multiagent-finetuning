# train_qwen_critic.py
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

model_name = "Qwen/Qwen2.5-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True  # needed for Qwen tokenizer
)

# Load data
ds = load_dataset("json", data_files="critic_data_incomplete_2.jsonl", split="train")

# Format prompt + target
def format_example(ex):
    prompt = (
        f"System: You are a critic agent in an NL2SQL framework.\n"
        f"User: Here is a question and its incorrect SQL alongwith the error, generate the correct SQL:\n"
        f"Question: {ex['question']}\n"
        f"Incorrect SQL: {ex['incorrect_sql']}\n"
        f"Error Code: {ex['error_code']}\n"
        f"Explanation: {ex['explanation']}\n"
        f"Correct SQL:"
    )
    return {"text": prompt + " " + ex["gold_sql"]}

ds = ds.map(format_example)

# Tokenize for chat-style input
def tokenize_fn(examples):
    texts = examples["text"]
    msgs = [
        [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": txt}
        ]
        for txt in texts
    ]
    # Use apply_chat_template on the structured messages
    batched = tokenizer.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
    )
    # Now batch tokenize
    tokens = tokenizer(batched, truncation=True, max_length=1024)
    return {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"]}
    '''
    tokens = tokenizer(
        tokenizer.apply_chat_template(
            [{"role":"system","content":""}, {"role":"user","content": x["text"]}],
            tokenize=False,
            add_generation_prompt=True
        ),
        truncation=True,
        max_length=512,
    )
    return {"input_ids": tokens.input_ids, "attention_mask": tokens.attention_mask}
    '''
ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])

# Trainer setup
config = SFTConfig(
    max_length=512,
    learning_rate=3e-5,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    output_dir="qwen2_critic_sft",
)

trainer = SFTTrainer(args=config, model=model, train_dataset=ds)
trainer.train()
model.push_to_hub("schaturv/critic_qwencoder2.5_3b_2")
