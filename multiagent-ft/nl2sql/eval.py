import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline
)
from datasets import load_dataset
import json
from sqlglot import parse_one, Dialects

# 1. Load Model and Tokenizer
model_path = "./results/checkpoint-500"  # Your trained model
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_path).to("cuda")

# 2. Load Test Data
test_data = load_dataset("json", data_files="../../Bird-SQL/dev_20240627/dev.json")["train"]

# 3. Evaluation Metrics
def evaluate_sql(pred_sql, gold_sql, db_schema):
    """Compute multiple SQL metrics"""
    metrics = {
        "exact_match": False,
        "valid_sql": False,
        "execution_match": None  # Requires DB connection
    }
    
    # Exact string match (case-insensitive)
    metrics["exact_match"] = pred_sql.strip().lower() == gold_sql.strip().lower()
    
    # Valid SQL check
    try:
        parse_one(pred_sql, dialect=Dialects.SQLITE)
        metrics["valid_sql"] = True
    except:
        pass
    
    return metrics

# 4. Generate Predictions
def generate_sql(question, db_id):
    prompt = f"Database: {db_id}\nQuestion: {question}\nGenerate SQL:"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=1024,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 5. Run Evaluation
results = []
for example in test_data.select(range(100)):  # Evaluate first 100 examples
    pred_sql = generate_sql(example["question"], example["db_id"])
    metrics = evaluate_sql(pred_sql, example["SQL"], example.get("schema", ""))
    
    results.append({
        "question": example["question"],
        "pred_sql": pred_sql,
        "gold_sql": example["SQL"],
        **metrics
    })

# 6. Calculate Aggregate Metrics
total = len(results)
aggregates = {
    "exact_match": sum(r["exact_match"] for r in results) / total,
    "valid_sql": sum(r["valid_sql"] for r in results) / total,
}

# 7. Save Results
with open("evaluation_results.json", "w") as f:
    json.dump({"examples": results, "aggregates": aggregates}, f, indent=2)

print(f"Evaluation Complete. Exact Match: {aggregates['exact_match']:.2%}")
