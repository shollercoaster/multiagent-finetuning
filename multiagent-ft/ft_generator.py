import json
from openai import OpenAI
import os
import numpy as np
from tqdm import tqdm
import random
from grader import grade_answer
import argparse

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def last_boxed_only(sample):
    """
    Given a (q,a) sample, filter the answers so that they only contain 
    the last \boxed{...} or \fbox{...} element
    """
    q, a = sample
    a = last_boxed_only_string(a)
    if a == None:
        return None
    return (q, a)

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def parse_answer(input_str):
	return remove_boxed(last_boxed_only_string(input_str))

def most_frequent(answers):
    answer_set = []
    counts = []

    for answer in answers:
        is_match = False
        for i, candidate_answer in enumerate(answer_set):
            if grade_answer(candidate_answer, answer):
                is_match = True
                counts[i] = counts[i] + 1
                break

        if not is_match:
            answer_set.append(answer)
            counts.append(1)

    responses = sorted(zip(counts, answer_set))
    return responses[-1][1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", action = "store", type = str, required = True, dest = "file_path")
    parser.add_argument("--save_path", action = "store", type = str, required = True, dest = "save_path")
    parser.add_argument("--gpt", action = "store_true", dest = "gpt")
    parser.add_argument("--iteration", action = "store", type = int, dest = "iteration", default = 1, help = "Iteration of finetuning")
    parser.add_argument("--model_ids", nargs = "*", dest = "model_ids", help = "GPT-4o-mini Model IDS after finetuning for future iterations")
    parser.set_defaults(gpt = False)
    args = parser.parse_args()
    data = json.load(open(args.file_path, "r"))

    iteration = args.iteration
    nagent = 3
    answers_dicts = [{} for i in range(nagent)]
    counters = [0 for i in range(nagent)]

    for k, v in tqdm(data.items()):
        agent_answers, gt_answer = v

        answers = []
        for agent_answer in agent_answers:
            answer = parse_answer(agent_answer[-1]['content'])

            if answer is not None:
                answers.append(answer)

        if len(answers) == 0:
            continue

        consensus_anwer = most_frequent(answers)

        for i, agent_answer in enumerate(agent_answers):

            answers_dict = answers_dicts[i]
            counter = counters[i]

            gen_answer = agent_answer[1]['content']
            answer = parse_answer(gen_answer)

            if grade_answer(answer, consensus_anwer):
                answers_dict[counter] = agent_answer[:2]
                counter = counter + 1
                counters[i] = counter

    if not args.gpt:
        ft_json = "{}_{}".format(args.save_path, iteration) + "training_consensus_{}.json"
        for i in range(nagent):
            answer_json = []
            with open(ft_json.format(i), "w") as f:
                answers_dict = answers_dicts[i]
                items = list(answers_dict.items())
                random.shuffle(items)
                for i, (k,v) in enumerate(items):
                    example_dict = {"id": f"identity_{i}"}
                    example = v[:1] + v[-1:]
                    conversations = []
                    for e in example:
                        new_e = {}
                        if e["role"] == "user":
                            new_e["from"] = "human"
                        else:
                            new_e["from"] = "gpt"
                        
                        new_e["value"] = e["content"]
                        conversations.append(new_e)
                    example_dict["conversations"] = conversations
                    answer_json.append(example_dict)
                json.dump(answer_json, f)
    else:
        ft_jsonl = "{}_{}".format(args.save_path, iteration) + "training_consensus_{}.jsonl"
        if args.iteration == 1 or not args.model_ids:
            model_ids = ['gpt-4o-mini-2024-07-18'] * nagent
        else:
            model_ids = args.model_ids
        for i in range(nagent):
            with open(ft_jsonl.format(i), "w") as f:
                items = list(answers_dict.items())
                random.shuffle(items)
                for k, v in items:
                    example = v[:1] + v[-1:]
                    json.dump({'messages': example}, f)
                    f.write("\n")

        # client.api_key = os.getenv("OPENAI_API_KEY")

        file_ids = []
        for i in range(nagent):
            file_id = client.files.create(
                    file=open(ft_jsonl.format(i), "rb"),
                        purpose='fine-tune'
            )
            file_ids.append(file_id)

        print("File id: ", file_ids)
        file_id = file_ids[0].id
        finetuned_model_ids = []
        for i in range(nagent):
            job_id = client.fine_tuning.jobs.create(training_file=file_ids[i].id, model=model_ids[i], hyperparameters={'n_epochs': 2, 'batch_size': 1, 'learning_rate_multiplier': 1})
            finetuned_model_ids.append(job_id.fine_tuned_model)
            print(i, job_id)
        print(answers_dict)
        print(job_id)
        with open("finetuned_generator_model_ids.json", "w") as f:
            json.dump(finetuned_model_ids, f)

