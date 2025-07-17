import json
from collections import defaultdict
from typing import List
from utils import (
    call_agent, load_spider, load_schema, BUGS_DB,
    schema_agent_prompt, subproblem_agent_prompt,
    query_plan_agent_prompt, sql_agent_prompt,
    critic_agent_prompt, exec_query, BUGS_DB, postprocess_sql, is_critic_valid
)
from utils import *
from prompts import *
import difflib

MAX_CRITIC_ATTEMPTS = 4

def evaluate():
    dev = load_spider(dev=True)
    taxonomy = json.load(open("error_taxonomy.json"))
    total, exact_match, valid_sql, exec_correct = 0, 0, 0, 0
    results = []

    for idx, item in enumerate(dev[100:105]):
        print("\n--- Sample", idx + 1, "---")
        question = item['question']
        gold_sql = item['query']
        db_id = item['db_id']

        schema = load_schema(db_id)

        entry = {
            "question": question,
            # "gold_sql": gold_sql,
            "db_id": db_id,
            "agents": {}
        }

        # 1. Schema Agent
        schema_prompt = schema_agent_prompt(question, schema)
        print("[Schema Agent Prompt]\n", schema_prompt)
        schema_info = call_agent(schema_agent_prompt(question, schema))
        print("[Schema Agent Output]\n", schema_info)
        entry["agents"]["schema"] = {"prompt": schema_prompt, "output": schema_info}

        # 2. Subproblem Agent
        subproblem_prompt = subproblem_agent_prompt(question, schema_info)
        print("[Subproblem Agent Prompt]\n", subproblem_prompt)
        sub_json = call_agent(subproblem_prompt)
        print("[Subproblem Agent Output]\n", sub_json)
        entry["agents"]["subproblem"] = {"prompt": subproblem_prompt, "output": sub_json}

        # 3. Query Plan Agent
        plan_prompt = query_plan_agent_prompt(question, schema_info, sub_json)
        print("[Query Plan Agent Prompt]\n", plan_prompt)
        plan = call_agent(query_plan_agent_prompt(question, schema_info, sub_json))
        print("[Query Plan Agent Output]\n", plan)
        entry["agents"]["plan"] = {"prompt": plan_prompt, "output": plan}

        # 4. SQL Generating Agent
        sql_prompt = sql_agent_prompt(plan, schema_info)
        print("[SQL Agent Prompt]\n", sql_prompt)
        sql = call_agent(sql_agent_prompt(plan, schema_info))
        sql = postprocess_sql(sql)
        print("[SQL Agent Output]\n", sql)

        gold_rows, gold_err = exec_query(f"../../spider/database/{db_id}/{db_id}.sqlite", gold_sql)
        gen_rows, gen_err = exec_query(f"../../spider/database/{db_id}/{db_id}.sqlite", sql)
        exec_failed = not(gen_err is None and gold_err is None and gen_rows == gold_rows)

        critic_history = []
        attempts = 0

        while exec_failed and attempts < MAX_CRITIC_ATTEMPTS:
            valid, errors = check_valid_critic_and_push_error(sql, question, db_id, taxonomy)
            critic_history.append(errors)
            if valid:
                break

            # regenerate plan & SQL with error types
            plan = call_agent(query_plan_agent_prompt(question, schema_info, sub_json, errors))
            sql = postprocess_sql(call_agent(sql_agent_prompt(plan, schema_info, errors)))
            print(f"\nISSUES MENTIONED: {errors}\n, SQL RE-GENERATED: {sql}\n\n")
            gen_rows, gen_err = exec_query(f"../../spider/database/{db_id}/{db_id}.sqlite", sql)
            print(f"\nVALID SQL?: {gen_err is None}\n")
            exec_failed = not(gen_err is None and gold_err is None and gen_rows == gold_rows)
            attempts += 1

        entry["agents"]["critic"] = [{
            "initial_sql": sql, "critic_history": critic_history,
            "exec_success": not exec_failed
        }]

        '''
        entry["agents"]["sql"] = {"prompt": sql_prompt, "output": sql}
        entry["agents"]["critic_issues"] = []
        for _ in range(2):
            valid_critic, issues = is_critic_valid(sql, question, db_id)
            print(f"[Critic Feedback] Issues found: {issues}")
            entry["agents"]["critic_valid"] = valid_critic
            entry["agents"]["critic_issues"].append(issues)
            if valid_critic:
                break
            else:
                # Regenerate SQL
                plan = call_agent(query_plan_agent_prompt(question, schema_info, sub_json, entry["agents"].get("critic_issues")))
                sql_prompt = sql_agent_prompt(plan, entry["agents"]["critic_issues"])
                sql = call_agent(sql_prompt)
                sql = postprocess_sql(sql)

        
        # 5. Critic Agent + bug fix loop
        bug_found = True
        while bug_found:
            critic_prompt = critic_agent_prompt(sql, json.dumps(BUGS))
            print("[Critic Agent Prompt]\n", critic_prompt)
            critic = json.loads(call_agent(critic_agent_prompt(sql, json.dumps(BUGS))))
            print("[Critic Agent Output]\n", critic_response)
            if critic.get("valid"):
                bug_found = False
            else:
                plan = call_agent(query_plan_agent_prompt(question, schema_info, sub_json))
                sql = call_agent(sql_agent_prompt(plan))
        '''
        # Metric 1: Exact Match
        gold_sql = postprocess_sql(gold_sql)
        entry["gold_sql"] = gold_sql
        print(f"Gold SQL: {gold_sql}\n Generated SQL: {sql}\n")
        if sql.strip().lower() == gold_sql.strip().lower():
            exact_match += 1
            entry["exact_match"] = True
        else:
            entry["exact_match"] = False

        # Metric 2: Valid SQL
        gen_rows, gen_err = exec_query(f"../../spider/database/{db_id}/{db_id}.sqlite", sql)
        entry["valid_sql"] = gen_err is None
        if gen_err is None:
            valid_sql += 1

        # Metric 3: Execution Accuracy
        gold_rows, gold_err = exec_query(f"../../spider/database/{db_id}/{db_id}.sqlite", gold_sql)
        entry["exec_match"] = (gen_err is None and gold_err is None and gen_rows == gold_rows)
        if gen_err is None and gold_err is None and gen_rows == gold_rows:
            exec_correct += 1

        total += 1
        results.append(entry)

    summary = {
        "total": total,
        "exact_match": exact_match,
        "valid_sql": valid_sql,
        "execution_accuracy": exec_correct,
        "exact_match_rate": round(exact_match / total, 4),
        "valid_sql_rate": round(valid_sql / total, 4),
        "execution_accuracy_rate": round(exec_correct / total, 4)
    }

    output = {
        "summary": summary,
        "results": results
    }

    with open("results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n======= Evaluation Summary =======")
    print(json.dumps(summary, indent=2))

    print(f"Total: {total}")
    print(f"Exact Match: {exact_match}/{total} = {exact_match/total:.2%}")
    print(f"Valid SQL: {valid_sql}/{total} = {valid_sql/total:.2%}")
    print(f"Execution Accuracy: {exec_correct}/{total} = {exec_correct/total:.2%}")


if __name__ == '__main__':
    evaluate()

