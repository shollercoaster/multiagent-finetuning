torchrun --nproc_per_node=4 multiagent_bird.py \
  --model_name_or_path defog/sqlcoder-7b-2 \
  --dataset_path ../../Bird-SQL/train/train.json \
  --output_dir ./models \
  --fp16 True \
  --gradient_accumulation_steps 8
