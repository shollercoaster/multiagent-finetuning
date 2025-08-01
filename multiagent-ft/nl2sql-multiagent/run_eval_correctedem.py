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
from analyze_by_subproblems import *
import difflib

MAX_CRITIC_ATTEMPTS = 4
MAX_REPAIR_ATTEMPTS = 2

def evaluate():
    dev = load_spider(dev=True)
    taxonomy = json.load(open("error_taxonomy.json"))
    total, exact_match, valid_sql, exec_correct = 0, 0, 0, 0
    results = []
    seen = set() # seen samples in creating dataset for finetuning critic

    for idx, item in enumerate(dev):
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

        # 1a. Schema Linking Agent
        schema_linking_prompt = alt_schema_linking_agent_prompt(question, schema, schema_info)
        # print("[Schema Linking Agent Prompt]\n", schema_prompt)
        corrected_schema = call_agent(schema_linking_prompt)
        print("[Schema Linking Agent Output]\n", corrected_schema)
        entry["agents"]["schema_linking"] = {"prompt": schema_linking_prompt, "output": corrected_schema}

        # 2. Subproblem Agent
        subproblem_prompt = subproblem_agent_prompt(question, corrected_schema)
        # print("[Subproblem Agent Prompt]\n", subproblem_prompt)
        sub_json = clean_json(call_agent(subproblem_prompt))
        print("[Subproblem Agent Output]\n", sub_json)
        entry["agents"]["subproblem"] = {"prompt": subproblem_prompt, "output": sub_json}

        subproblem_specific_clauses = list(set(parse_subproblems(sub_json)))
        subprob_plan, subprob_sql = clause_specific_prompts(subproblem_specific_clauses)

        # 3. Query Plan Agent
        plan_prompt = query_plan_agent_prompt(question, corrected_schema, sub_json)
        # plan_prompt = query_plan_agent_prompt(question, schema, sub_json, subprob_plan)
        # print("[Query Plan Agent Prompt]\n", plan_prompt)
        plan = call_agent(plan_prompt)
        print("[Query Plan Agent Output]\n", plan)
        entry["agents"]["plan"] = {"prompt": plan_prompt, "output": plan}

        # 4. SQL Generating Agent
        sql_prompt = sql_agent_prompt(plan) # corrected_schema
        # sql_prompt = sql_agent_prompt(plan, schema, subprob_sql)
        # print("[SQL Agent Prompt]\n", sql_prompt)
        sql = call_agent(sql_prompt)
        sql = postprocess_sql(sql)
        print("[SQL Agent Output]\n", sql)

        gold_rows, gold_err = exec_query(f"../../spider/database/{db_id}/{db_id}.sqlite", gold_sql)
        gen_rows, gen_err = exec_query(f"../../spider/database/{db_id}/{db_id}.sqlite", sql)
        exec_failed = not(gen_err is None and gold_err is None and gen_rows == gold_rows)

        critic_history = []
        attempts = 0

        while exec_failed and attempts < MAX_CRITIC_ATTEMPTS:
            valid, errors = check_valid_critic_and_push_error(sql, question, db_id, corrected_schema, taxonomy)
            critic_history.append(errors)

            if not valid:
                for error_code in errors:
                    key = (question, sql, error_code)
                    if key in seen: continue
                    seen.add(key)
                    out = {
                        "question": question,
                        "incorrect_sql": sql,
                        "error_code": error_code,
                        "explanation": taxonomy.get(error_code, ""),
                        "gold_sql": gold_sql
                    }
                    with open("critic_data.jsonl", "a", encoding="utf-8") as fout:
                        fout.write(json.dumps(out, ensure_ascii=False) + "\n")

            if valid:
                break
            # repaired_sqls = { error: postprocess_sql(call_agent(repair_agent_prompt(error, sql, question, schema_info))) for error in errors}
            # sql = postprocess_sql(call_agent(repair_combined_agent_prompt(repaired_sqls, question)))

            # add subproblem related instructions
            # subprob_plan, subprob_sql = clause_specific_prompts(subproblem_specific_clauses)
            # regenerate plan & SQL with error types
            plan = call_agent(query_plan_agent_prompt(question, corrected_schema, sub_json, errors)) # subprob_plan, errors))
            plan_checked = call_agent(plan_sanity_agent_prompt(question, corrected_schema, sub_json, plan))
            sql = postprocess_sql(call_agent(sql_agent_prompt(plan_checked, corrected_schema, critic_issues=errors), temperature=0.3)) #, corrected_schema, subprob_sql, errors)))
            print(f"\nISSUES MENTIONED: {errors}\n, SQL RE-GENERATED: {sql}\n\n")
            gen_rows, gen_err = exec_query(f"../../spider/database/{db_id}/{db_id}.sqlite", sql)
            print(f"\nVALID SQL?: {gen_err is None}\n")
            exec_failed = not(gen_err is None and gold_err is None and gen_rows == gold_rows)
            attempts += 1

        entry["agents"]["critic"] = [{
            "initial_sql": sql, "critic_history": critic_history,
            "exec_success": not exec_failed
        }]
        
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
        if gen_err is None and gold_err is None:
            gen_norm = normalize_rows(gen_rows)
            gold_norm = normalize_rows(gold_rows)
            if (gen_norm == gold_norm):
                entry["exec_match"] = True
                exec_correct += 1
            else:
                entry["exec_match"] = False
        else:
            entry["exec_match"] = False
        ea = entry["exec_match"]
        print(f"\n Execution Match: {ea}\n")
        total += 1
        results.append(entry)

        print(f"Total: {total}")
        print(f"Exact Match: {exact_match}/{total} = {exact_match/total:.2%}")
        print(f"Valid SQL: {valid_sql}/{total} = {valid_sql/total:.2%}")
        print(f"Execution Accuracy: {exec_correct}/{total} = {exec_correct/total:.2%}")
    
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

    with open("ablations/full_clausespecific_unorderedeval_results.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n======= Evaluation Summary =======")
    print(json.dumps(summary, indent=2))

    print(f"Total: {total}")
    print(f"Exact Match: {exact_match}/{total} = {exact_match/total:.2%}")
    print(f"Valid SQL: {valid_sql}/{total} = {valid_sql/total:.2%}")
    print(f"Execution Accuracy: {exec_correct}/{total} = {exec_correct/total:.2%}")


if __name__ == '__main__':
    evaluate()

