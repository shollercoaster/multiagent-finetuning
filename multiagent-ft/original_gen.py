from glob import glob
from openai import OpenAI
import os
import torch
import json
import numpy as np
import re
import time
import random
import transformers
import argparse

client = OpenAI()

def generate_answer(answer_context, model = "gpt-4o-mini", hf_model = None, tokenizer = None, device = None, temperature = 1, top_p = 0.9):
    if model not in ["mistral", "llama3", "phi3"]:
        if model == "gpt3.5":
            model_str = "gpt-3.5-turbo-0125"
        elif model == "gpt-4o-mini":
            model_str = "gpt-4o-mini"
        else:
            model_str = "gpt-4o"
        try:
            completion = client.chat.completions.create(
                    model=model_str,
                    messages=answer_context,
                    temperature = temperature,
                    n=1,
                    max_tokens=300)
            completion = {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": completion.choices[0].message.content
                        }
                    }
                ]
            }


        except Exception as e:
            print("retrying due to an error......", e)
            time.sleep(20)
            return generate_answer(answer_context)
    else:
        input_text = tokenizer.apply_chat_template(answer_context, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
        output = hf_model.generate(input_ids, max_length=len(input_ids[0]) + 2048, 
                                return_dict_in_generate=True, output_scores=True, do_sample = True, top_p = top_p, temperature = temperature)
        generated_ids = output[0][:, len(input_ids[0]):].squeeze().to("cpu")
        completion = tokenizer.decode(generated_ids, skip_special_tokens=True)
        completion = {"choices": [{"message": {"role": "assistant", "content": completion}}]}
    return completion

def construct_assistant_message(completion):
    content = completion["choices"][0]["message"]["content"]
    return {"role": "assistant", "content": content}

def summarize_message(agent_contexts, model = "gpt4o-mini", hf_model = None, tokenizer = None, device = None, temperature = 1, top_p = 0.9):
    prefix_string = "Here are a list of opinions from different agents: "

    for agent in agent_contexts:
        agent_response = agent[-1]["content"]
        response = "\n\n One agent response: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\n Write a summary of the different opinions from each of the individual agent and explain the reasoning in each solution."
    agent_context = [{"role": "user", "content": prefix_string}]
    completion = generate_answer(agent_context, model = model, hf_model = hf_model, tokenizer = tokenizer, device = device, temperature = temperature, top_p = top_p)
    content = completion["choices"][0]["message"]["content"]

    return content

def construct_message(agents, prefix, idx):
    if len(agents) == 0:
        return {"role": "user", "content": "Can you double check that your answer is correct? Please reiterate your answer, with your final answer a single answer of the form \\boxed{{answer}} at the end of your response.".format(prefix)}

    prefix_string = "Here is are solution from other agents: "

    for agent in agents:
        agent_response = agent[idx]["content"]
        response = "\n\n One agent response: {}".format(agent_response)

        prefix_string = prefix_string + response

    prefix_string = prefix_string + "\n\n Using each response as additional advice, can you give an updated bullet by bullet answer to {}? Your final answer should be be in the form \\boxed{{answer}} given at the end of your response.".format(prefix)
    return {"role": "user", "content": prefix_string}

def construct_message_summary(summary, prefix, idx):
    summary = summary[:300]
    prefix_string = "Here is a summary of solutions from several other agents: {}".format(summary)

    prefix_string = prefix_string + "\n\n Examine each these solutions as additional advice, can solve {} and give your updated answer? Explain your reasoning. \n Your final answer should be be in the form \\boxed{{answer}} given at the end of your response.".format(prefix)
    return {"role": "user", "content": prefix_string}


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents", action = "store", dest = "agents", type = int, required = True, help = "Number of agents for debate")
    parser.add_argument("--rounds", action = "store", dest = "rounds", type = int, required = True, help = "Number of rounds for debate")
    parser.add_argument("--save_str", action = "store", type = str, dest = "save_str", required = True)
    parser.add_argument("--model", action = "store", default = "gpt3.5", type = str, choices = ["gpt-4o", "gpt-4o-mini", "gpt3.5", "gpt4", "mistral", "llama3", "phi3"])
    parser.add_argument("--summarize", action = "store_true", dest = "summarize")
    parser.add_argument("--device", action = "store", type = int, dest = "device", default = 0)
    parser.add_argument("--temperature", action = "store", default = 1, type = float, dest = "temperature")
    parser.add_argument("--top_p", action = "store", default = 0.9, type = float, dest = "top_p")
    parser.set_defaults(summarize = False)
    args = parser.parse_args()
    jsons = sorted(glob("MATH/test/algebra/*.json"))[:50]
    random.seed(0)
    random.shuffle(jsons)
    hard_problems = []

    for json_file in jsons:
        data = json.load(open(json_file, "r"))
        if ('1' in data['level']) or ('2' in data['level']) or ('3' in data['level']):
            hard_problems.append(data)

    agents = args.agents
    rounds = args.rounds
    random.seed(0)
    random.shuffle(hard_problems)

    generated_description = {}
    hf_model, device, tokenizer = None, None, None
    if args.model == "mistral" or args.model == "mixtral" or args.model == "llama3" or args.model == "phi3":
        if args.model == "mistral":
            model_str = "mistralai/Mistral-7B-Instruct-v0.2"
        elif args.model == "llama3":
            model_str = "meta-llama/Meta-Llama-3-8B"
        elif args.model == "phi3":
            model_str = "microsoft/Phi-3-mini-128k-instruct" 
        else:
            raise NotImplementedError()
        device = torch.device(f"cuda:{args.device}")
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_str)
        hf_model = transformers.AutoModelForCausalLM.from_pretrained(model_str).to(device)

    for problem, data in enumerate(hard_problems[:5]):
        question = data["problem"]
        answer = data["solution"]

        print("problem: ", problem)

        answer_parse = parse_answer(answer)

        agent_contexts = [[{"role": "user", "content": """Can you solve the following math problem? {} Provide a bullet point summary of your reasoning. Your final answer should be a single answer, in the form \\boxed{{answer}}, at the end of your response.""".format(question)}] for agent in range(agents)]

        for round in range(rounds):
            for i, agent_context in enumerate(agent_contexts):

                if round != 0:
                    if args.summarize:
                        agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                        random.shuffle(agent_contexts_other)
                        summary = summarize_message(agent_contexts_other[:5], model = args.model, hf_model = hf_model, tokenizer = tokenizer, 
                                                        device = device, temperature= args.temperature, top_p = args.top_p)
                        print(summary)
                        message = construct_message_summary(summary, question, 2 * round - 1)
                    else:
                        agent_contexts_other = agent_contexts[:i] + agent_contexts[i+1:]
                        random.shuffle(agent_contexts_other)
                        message = construct_message(agent_contexts_other[:5], question, 2 * round - 1)
                    agent_context.append(message)

                completion = generate_answer(agent_context, model = args.model, hf_model = hf_model, tokenizer = tokenizer, 
                                                device = device, temperature= args.temperature, top_p = args.top_p)

                assistant_message = construct_assistant_message(completion)
                agent_context.append(assistant_message)
                print(completion)
                print(problem, "{} gt_answer: ".format(problem), answer_parse)

        generated_description[question] = (agent_contexts, answer)

    json.dump(generated_description, open("{}_{}_{}.json".format(args.save_str, agents, rounds), "w"))
    pass
