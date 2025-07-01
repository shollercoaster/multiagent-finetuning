# Generating datasets
`python original_gen.py --agents 3 --rounds 2 --model gpt-4o-mini --save_str logs --summarize`

# Finetuning the generator 
`python ft_generator.py --file_path logs_3_2.json --save_path gen_ft_data.jsonl --gpt --iteration 0`

# Finetuning the critic
`python ft_critic.py --file_path logs_3_2.json --save_path ft_critic_data --gpt --iteration 0 --model_ids gpt-4o-mini-2024-07-18 gpt-4o-mini-2024-07-18 gpt-4o-mini-2024-07-18`

# Run finetuned models
`python ft_gen.py --generators ft_gen_data --critics ft_critic_data --model gpt3.5 --save_str algebra_finetuned`

# Evaluate 
`python eval_math.py`
