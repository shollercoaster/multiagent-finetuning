import json
from collections import defaultdict
from typing import List
from utils import *
from prompts import *
from analyze_by_subproblems import *
import difflib
from scratchpad import Scratchpad

MAX_ATTEMPTS = 3

def evaluate():
    dev = load_spider(dev=True)
    taxonomy = json.load(open("error_taxonomy.json"))
    total, exact_match, valid_sql, exec_correct = 0, 0, 0, 0
    results = []
    seen = set() # seen samples in creating dataset for finetuning critic

    for idx, item in enumerate(dev[100:200]):
        print("\n--- Sample", idx + 1, "---")
        question = item['question']
        gold_sql = item['query']
        db_id = item['db_id']

        schema = load_schema(db_id)

        scratchpad = Scratchpad(
            question=item['question'],
            db_id=item['db_id'],
            schema=load_schema(item['db_id'])
        )

        # 1. Schema Agent
        schema_prompt = alt_schema_linking_agent_prompt(question, schema)
        # print("[Schema Agent Prompt]\n", schema_prompt)
        corrected_schema = call_agent(schema_prompt)
        print("[Schema Agent Output]\n", corrected_schema)

        # 2. Subproblem Agent
        subproblem_prompt = subproblem_agent_prompt(question, corrected_schema)
        print("[Subproblem Agent Prompt]\n", subproblem_prompt)
        sub_json = clean_json(call_agent(subproblem_prompt))
        print("[Subproblem Agent Output]\n", sub_json)

        subproblem_specific_clauses = list(set(parse_subproblems(sub_json)))
        subprob_plan, subprob_sql = clause_specific_prompts(subproblem_specific_clauses)

        # 3. Query Plan Agent
        plan_prompt = query_plan_agent_prompt(question, corrected_schema, sub_json)
        scratchpad.plan = call_plan_agent(scratchpad.question, scratchpad.schema)
        print("[Query Plan Agent Output]\n", plan)

        # 4. SQL Generating Agent
        sql_prompt = sql_agent_prompt(question, plan, corrected_schema)
        initial_sql = call_agent(sql_prompt)
        
        sql = postprocess_sql(initial_sql)
        print("[SQL Agent Output]\n", sql)

        current_sql = sql

        for attempt_num in range(MAX_ATTEMPTS):
            
            # Execute the current SQL
            exec_match, error = query_execution(item, current_sql)
            exec_failed = not(exec_match)

            # If execution succeeds, we are done with the loop
            if exec_match:
                print("Execution successful!")
                break
            
            # --- Execution FAILED, update scratchpad and start correction ---
            print(f"Attempt {attempt_num + 1} failed. Error: {error}")
            
            # Add the failed attempt to the scratchpad's history
            # Note: The correction_plan is still empty for this entry
            scratchpad.add_attempt(sql=current_sql, error=error)

            # --- Call Correction Plan Agent ---
            # Get the full history from the scratchpad
            correction_context = scratchpad.get_correction_context()
            plan_prompt = correction_plan_agent_prompt_with_scratchpad(correction_context)
            plan_response = call_agent(plan_prompt)
            
            # IMPORTANT: Update the last attempt in the scratchpad with the plan that was just generated
            scratchpad.attempts[-1]['correction_plan'] = plan_response

            # --- Call Correction SQL Agent ---
            # Get the history again, which now includes the latest correction plan
            full_context_for_sql_agent = scratchpad.get_correction_context()
            sql_prompt = sql_correction_prompt_with_scratchpad(full_context_for_sql_agent, plan_response)
            
            # Generate the new SQL for the next iteration
            current_sql = postprocess_sql(call_agent(sql_prompt))

        
        sql = current_sql
        exec_match, error = query_execution(item, sql)
        exec_failed = not(exec_match)
        # Metric 1: Exact Match
        gold_sql = postprocess_sql(gold_sql)
        entry["gold_sql"] = gold_sql
        entry["gen_sql"] = sql
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

        entry["exec_match"] = exec_match
        if exec_match: 
            exec_correct += 1
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

    with open("ablations/200_1schema_correctionloop_promptchanges.json", "w") as f:
        json.dump(output, f, indent=2)

    print("\n======= Evaluation Summary =======")
    print(json.dumps(summary, indent=2))

    print(f"Total: {total}")
    print(f"Exact Match: {exact_match}/{total} = {exact_match/total:.2%}")
    print(f"Valid SQL: {valid_sql}/{total} = {valid_sql/total:.2%}")
    print(f"Execution Accuracy: {exec_correct}/{total} = {exec_correct/total:.2%}")


if __name__ == '__main__':
    evaluate()

