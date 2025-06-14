import os
import json
from openai import OpenAI
import numpy as np
import time
from tqdm import tqdm
from grader import grade_answer
import random
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
    parser.add_argument("--model_ids", nargs = "*", dest = "model_ids", help = "GPT-3.5 Model IDS after finetuning for future iterations")
    parser.set_defaults(gpt = False)
    args = parser.parse_args()
    data = json.load(open(args.file_path, "r"))
    data_list = [data]

    iteration = args.iteration
    nagent = 3
    counters = [0 for i in range(nagent)]
    answers_dicts = [{} for i in range(nagent)]
    correct_counters = [0 for i in range(nagent)]
    correct_answers_dicts = [{} for i in range(nagent)]

    for k, v in tqdm(data.items()):
        agent_answers, gt_answer = v

        answers = []
        for agent_answer in agent_answers:
            answer = parse_answer(agent_answer[-1]['content'])

            if answer is not None:
                answers.append(answer)

        if len(answers) == 0:
            continue

        consensus_answer = most_frequent(answers)

        for data_i in data_list:
            agent_answers, gt_answer = data_i[k]

            for i, agent_answer in enumerate(agent_answers):
                gen_answer = agent_answer[1]['content']
                answer = parse_answer(gen_answer)

                other_gen_answer = agent_answer[-1]['content']
                other_answer = parse_answer(other_gen_answer)

                if grade_answer(other_answer, consensus_answer):
                    if not grade_answer(answer, other_answer):
                        answers_dict = answers_dicts[i]
                        counter = counters[i]

                        answers_dict[counter] = agent_answer
                        counter = counter + 1

                        counters[i] = counter
                    else:
                        correct_answers_dict = correct_answers_dicts[i]
                        correct_counter = correct_counters[i]

                        correct_answers_dict[correct_counter] = agent_answer
                        correct_counter = correct_counter + 1

                        correct_counters[i] = correct_counter

    if not args.gpt:
        ft_json = "{}_{}_".format(args.save_path, iteration) + "training_consensus_critic_{}.jsonl"
        for i in range(nagent):
            answer_json = []
            data = answers_dicts[i]
            correct_data = correct_answers_dicts[i]
            correct_data = list(correct_data.items())
            random.shuffle(correct_data)
            with open(ft_json.format(i), "w") as f:
                for j, (k, v) in enumerate(data.items()):
                    example_dict = {"id": f"identity_incorrect_{j}"}
                    conversations = []
                    for e in v:
                        new_e = {}
                        if e["role"] == "user":
                            new_e["from"] = "human"
                        else:
                            new_e["from"] = "gpt"
                        new_e["value"] = e["content"]
                        conversations.append(new_e)
                    example_dict["conversations"] = conversations
                    json.dump(example_dict, f)
                    f.write("\n")

                for j, (k, v) in enumerate(correct_data):
                    example_dict = {"id": f"identity_correct_{j}"}
                    conversations = []
                    for e in v:
                        new_e = {}
                        if e["role"] == "user":
                            new_e["from"] = "human"
                        else:
                            new_e["from"] = "gpt"
                        new_e["value"] = e["content"]
                        conversations.append(new_e)
                    example_dict["conversations"] = conversations
                    json.dump(example_dict, f)
                    f.write("\n")

#                    if random.choice([0, 1]):
#                        example = correct_data[i][1]
#                        new_example_dict = {"id": f"identity_{i}"}
#                        conversations = []
#                        for e in example:
#                            new_e = {}
#                            if e["role"] == "user":
#                                new_e["from"] = "human"
#                            else:
#                                new_e["from"] = "gpt"
                            
#                            new_e["value"] = e["content"]
#                            conversations.append(new_e)
#                        new_example_dict["conversastions"] = conversations
#                        answer_json.append(new_example_dict)
#                json.dump(answer_json, f)
		
    else:
        if iteration == 1:
            model_ids = ["gpt-4o-mini-2024-07-18" for _ in range(nagent)]
        else:
            model_ids = args.model_ids
            assert len(model_ids) == nagent
        print("original counters per agent: ", counters)
        print("correct counters per agent: ", correct_counters)
        ft_jsonl = "{}_{}_".format(args.save_path, iteration) + "training_consensus_critic_{}.jsonl"

        for i in range(nagent):
            data = answers_dicts[i]
            correct_data = correct_answers_dicts[i]

            correct_data = list(correct_data.items())
            random.shuffle(correct_data)

            print("data elements: ", len(data))
            with open(ft_jsonl.format(i), "w") as f:
                for i, (k, v) in enumerate(data.items()):
                    messages = []
                    for e in v:
                        message = {
                            "role": "user" if e["role"] == "user" else "assistant",
                            "content": e["content"]
                        }
                        messages.append(message)

                    json.dump({"messages": messages}, f)
                    f.write("\n")

                # Optionally write correct examples (if available)
                for j, (k, v) in enumerate(correct_data):
                    messages = []
                    for e in v:
                        message = {
                            "role": "user" if e["role"] == "user" else "assistant",
                            "content": e["content"]
                        }
                        messages.append(message)

                    json.dump({"messages": messages}, f)
                    f.write("\n")

#                    print(i)
#                    example = v
#                    print(example)
#                    json.dump({'messages': example}, f)
#                    f.write("\n")

#                    if random.choice([0, 1]):
#                        example = correct_data[i][1]
#                        json.dump({'messages': example}, f)
#                        f.write("\n")
        file_ids = []
        for i in range(nagent):
            file_id = client.files.create(file=open(ft_jsonl.format(i), "rb"),
            purpose='fine-tune')
            file_ids.append(file_id)

        print("File id: ", file_ids)
        file_id = file_ids[0].id

        for i in range(nagent):
            job_id = client.fine_tuning.jobs.create(training_file=file_ids[i].id, model=model_ids[i], hyperparameters={'n_epochs': 2, 'batch_size': 1, 'learning_rate_multiplier': 1})

            print(i, "Job id: ", job_id)
