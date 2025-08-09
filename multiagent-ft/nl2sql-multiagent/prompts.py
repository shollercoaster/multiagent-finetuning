import os, json
from string import Template
from typing import List

# Load your API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
taxonomy = json.load(open("error_taxonomy.json"))

def plan_cot_agent_prompt(question: str,
                             schema: str,
                             critic_issues="",
                             clause_prompts="",
                             subproblem_output="") -> str:
    base = Template(
        """
You are a Chain-of-Thought Query Plan agent in expert Text-to-SQL system, specializing in creating logical query plans.

Your task is to decompose a question into a series of clear, natural language steps that will guide a separate SQL generation agent.

Follow this process:

1. **Analyze the Question:** Deeply understand the user's intent. What information are they asking for? What conditions and constraints are mentioned?
2. **Map to Schema:** Examine the database schema to identify all necessary tables, columns, and the foreign relationships (JOINs) required to answer the question. Explicitly state the tables needed.
3. **Formulate the Plan:** Create a step-by-step plan. Each step should correspond to a part of the final SQL query (e.g., selecting columns, filtering data, grouping results, ordering results). The plan should be easy for another AI agent to follow to write the SQL code.

**Crucial Instructions:**
- The plan must be in natural language.
- Explicitly mention which table and column to use for each step.
- Do NOT write the final SQL query.
- All necessary tables appear in `FROM` and `JOIN` steps with correct join keys.
- GROUP BY, HAVING, ORDER BY, LIMIT clauses appear in the proper order and ONLY when they're necessary.

Inputs:
Question:
$question

Table Schema:
$schema

$feedback
ONLY return a concise, bullet-pointed summary of suggested query plan.
""")
    if critic_issues != "":
        feedback = "\nA CRITIC AGENT HIGHLIGHTED THESE POTENTIAL ERRORS, MAKE SURE YOUR QUERY PLAN DOES NOT MAKE THESE MISTAKES:\n" # "\nIMPORTANT: Avoid the following errors detected in the earlier S>        feedback += "\n".join([f"- {issue}" for issue in critic_issues])
        feedback += "\n Generate a query plan that FIXES all listed errors like missing column name, etc.\n"
    else:
        feedback = ""
    return base.substitute(
        question=question, schema=schema, feedback=feedback)

# 1. Schema Agent prompt
def schema_agent_prompt(question: str, schema_list: str) -> str:
    return Template(
        """
You are a Schema Agent in an NL2SQL framework. Given a natural language question and table schemas (with columns, PKs, and FKs), identify the relevant tables and columns needed, including intermediate tables for joins.

Question: $question
Schemas:
$schema_list

Return a list of lines in the format:
Table: primary_key_col, foreign_key_col, col1, col2, ...

ALWAYS list Foreign Key and Primary Key information.
ONLY list relevant tables and columns in given format and no other extra characters.
"""
    ).substitute(question=question, schema_list=schema_list)

def alt_schema_linking_agent_prompt(question: str, table_schema: str, schema_agent_output="") -> str:
    return Template(
        """
You are a Schema Linking Agent in an NL2SQL framework. Find the schema_links for generating SQL query for the question based on the database schema and Foreign keys.\n"

Question: $question
Table Schema:
$table_schema

Return the schema links in given format:
Table1: Col1, Col2..
Foreign Keys = (Table1.ColXYZ, Tabl2.ColXYZ).. 
ONLY list relevant tables and columns and Foreign Keys in given format and no other extra characters.
"""
    ).substitute(
        question=question,
        table_schema=table_schema)

# 1.a Schema Linking Agent Prompt
def schema_linking_agent_prompt(question: str, table_schema: str, schema_agent_output: str) -> str:
    return Template(
        """
You are a Schema Linking Agent in an NL2SQL framework. Your job is to verify the schema agent's output and ensure all necessary schema relationships are included.

Given:
- A natural language question
- Database schemas with columns, primary keys (PK), and foreign keys (FK)
- The output of a Schema Agent listing relevant tables and columns

Check for:
- Missing or incorrect FK-PK relationships and add them
- Incomplete column selections (especially join keys)
- Table alias mismatches
- Linkage errors that would lead to incorrect joins or groupBy clauses

If issues exist, correct them by editing ONLY the schema agent's output.

Inputs:
Question: $question

Schemas:
$schema_list

Schema Agent Output:
$schema_agent_output

Correct and return the fixed output in the SAME format:
Table: primary_key_col, foreign_key_col, col1, col2, ...
"""
    ).substitute(
        question=question,
        schema_list=table_schema,
        schema_agent_output=schema_agent_output,
    )

# 2. Subproblem Agent prompt
def subproblem_agent_prompt(question: str, schema_info: str) -> str:
    return Template(
        """
You are a Subproblem Agent in an NL2SQL framework. Your task is to decompose a natural language question into SQL subproblems.

You will be provided:
- A natural language question
- A textual schema summary that lists relevant tables and columns (generated by a Schema Agent)

Use this information to infer which SQL clauses are likely needed (e.g., WHERE, GROUPBY, JOIN, DISTINCT, ORDER BY, HAVING, EXCEPT, LIMIT, UNION).

Question:
$question

Schema:
$schema

Output a JSON object containing a list of subproblems:
{
  "subproblems": [
    { "clause": "SELECT", "expression": "..." },
    { "clause": "JOIN", "expression": "..." },
    ...
  ]
}

Only output valid JSON — no markdown, no extra commentary.

"""
    ).substitute(question=question.strip(), schema=schema_info.strip())

# 3. Query Plan Agent prompt
def query_plan_agent_prompt(question: str, schema_info: str, subproblem_json, subprob_plan="", critic_issues: list = None) -> str:
    base_prompt = Template(
        """
You are a Query Plan Agent in an NL2SQL Framework. Using the question, schema info, and subproblems, generate a step-by-step SQL query plan.

Question: $question
Schema Info:
$schema_info
Subproblems:
$subproblem_json

$critic_feedback

$subprob_plan

Return plan steps with specific table, column names like:
1. FROM tableA
2. JOIN tableB ON tableA.colX = tableB.colY
3. JOIN tableC ON tableB.colZ = tableC.colW

Return only the plan (no SQL or extra text).
"""
    )
    if critic_issues != "":
        feedback = "\nPREVIOUS ERRORS TO AVOID:\n" # "\nIMPORTANT: Avoid the following errors detected in the earlier SQL attempt:\n"
        feedback += "\n".join([f"- {issue}" for issue in critic_issues])
        feedback += "\n Generate a query plan that FIXES all listed errors.\n"
    else:
        feedback = ""

    return base_prompt.substitute(question=question, schema_info=schema_info, subproblem_json=subproblem_json, critic_feedback=feedback, subprob_plan=subprob_plan)

# 4. SQL Generating Agent prompt
def sql_agent_prompt(question, plan: str, schema=None, subprob_sql="", critic_issues : list = None) -> str:
    base_prompt = Template(
        """
You are a world-class SQL writer AI in an NL2SQL multiagent framework. Your task is to write a single, syntactically correct SQL query that perfectly implements the provided query plan.
Pay close attention to the table and column names in the schema.

$question

Plan:
$plan

$schema

$subprob_sql

$critic_feedback

Write ONLY the final valid SQL query. Do NOT include commentary or unnecessary characters in the query.
"""
    )
    if critic_issues:
        # feedback = "\nIMPORTANT: Avoid the following errors found in the earlier SQL attempt:\n"
        feedback = "\nENSURE THAT YOU ADDRESS THESE ERRORS:\n" + "\n".join(f"- {e}" for e in critic_issues)
        # feedback += "\n".join([f"- {issue}" for issue in critic_issues])
    else:
        feedback = ""
    if schema:
        schema_info = "\n Relevant Table Schema: \n" + schema
    else:
        schema_info = ""
    return base_prompt.substitute(plan=plan, schema=schema_info, critic_feedback=feedback, subprob_sql=subprob_sql, question=question)

def correction_sql_agent_prompt(question: str, schema, correction_plan, wrong_sql) -> str:
    return """
You are an expert SQL debugger AI in NL2SQL multiagent framework. Your previous attempt to write a query failed.
Your new task is to analyze the feedback and your incorrect query, then generate a new, corrected query after reading the question and analyzing the relevant schema.

Question:
$question 

Incorrect SQL:
$wrong_sql

Correction query plan- You MUST follow these steps to fix the query:
$correction_plan

$schema

Write ONLY the final valid SQL query. Do NOT include commentary or unnecessary characters in the query.
""").substitute(question=question, wrong_sql=wrong_sql, schema=schema, correction_plan=correction_plan)
  
def correction_plan_agent_prompt(question: str, wrong_sql: str, schema, database_error=None) -> str:
    base = Template(
    """
You are a Senior SQL Debugger. Your sole task is to analyze a failed SQL query and its database error (if any) to create a clear, step-by-step correction plan. Do NOT write the corrected SQL yourself.

You are an expert in a comprehensive error taxonomy, including categories like:
- `schema.incorrect_column/table`: Mismatch between query and schema.
- `syntax.invalid_keyword`: SQL syntax errors.
- `join.missing_or_incorrect`: Errors in JOIN logic.
- `aggregation.incorrect_grouping`: Errors with GROUP BY or aggregate functions.
- `ambiguity.unclear_intent`: When the query doesn't match the question's intent.

**Your Reasoning Process:**
1.  **Analyze the Ground Truth Error:** First, examine the `Execution Error from Database`. This is the most important clue.
2.  **Pinpoint the Mismatch:** Compare the error message to the `Failed SQL Query` and the `Pruned Schema` to find the exact source of the error.
3.  **Formulate a Hypothesis:** State the root cause of the error in a single sentence.
4.  **Create the Plan:** Write a concise, step-by-step natural language plan that a junior SQL developer can follow to fix the query.

**Input for Analysis:**

**1. Original Question:**
"$question"

**2. Pruned Schema (Only relevant tables are shown):**
$schema

**3. Failed SQL Query:**
$wrong_sql

$database_error

Your goal is to provide a clear, step-by-step explanation of why the query is wrong and exactly how to fix it. Reference the specific tables and columns that need to be changed. Output this as a natural language correction plan.
""")
    if database_error:
        prompt = "**4. Query Execution Error:** \n" + database_error
    return base.substitute(question=question.strip(), wrong_sql=wrong_sql.strip(), schema=schema, database_error=database_error)
       
def critic_agent_prompt(question: str, sql: str) -> str:
    return Template(
    """
You are a Critic Agent. Your job is to inspect a SQL query and determine whether it is logically and structurally valid.

Check for:
- Logical fallacies (e.g. comparing two incompatible types)
- Incorrect/missing joins
- Missing clauses required to answer the question
- Aggregations used incorrectly
- Hardcoded values that should come from a column
- Unused subqueries or tables

Question:
{question}

SQL Query:
{sql}

Return **only** one of the following:

1. If the SQL is valid:

{
   "valid": true
}

2. If the SQL has issues:

{
  "valid": false,
  "error_types": [
    {"error_type": "explanation"},
    ...
  ]
}

DO NOT add any explanation, markdown, or text outside this JSON. Only valid JSON.
Only return errors if you find them. Don't complicate things.
"""
    ).substitute(question=question.strip(), sql=sql.strip())

def critic_finetuned_prompt(question: str, sql: str, error_codes):
    base = Template("""
You are a Critic Agent in an NL2SQL framework. Given a text question and the incorrect SQL alongwith the errors in that>
Question:
$question

Incorrect SQL:
$sql

Errors:
$error

Only return the valid SQL.
""")
    if error_codes:
        feedback = "\n".join([f"{error_code}: {taxonomy.get(error_code, '')}" for error_code in error_codes])
        feedback += "\n Generate a query that FIXES all listed errors.\n"
    else:
        feedback = ""
    return base.substitute(question=question, sql=sql, error=feedback).strip()

def taxonomy_critic_agent_prompt(question: str, sql: str, taxonomy: dict, schema) -> str:
    taxonomy_str = json.dumps(taxonomy, indent=2)
    return f"""
You are a Critic Agent. Given an NL question and generated SQL, identify structural or semantic errors based on the following taxonomy (9 categories, 36 subtypes):

{taxonomy}

Input:
Question: {question}
SQL: {sql}
Schema: {schema}

Return **only** JSON following this schema:

If no errors:
{{
  "valid": true
}}

If there are errors:
{{
  "valid": false,
  "error_types": [ "category.subtype", ... ]
}}

Check also for, and mention the name of column or table that's missing or extra:
- join_missing_table
- join_wrong_column
- table_forbidden

Example: ["join.missing_table - (table_name)", "join.extra_col - (col_name)", "aggregation.agg_no_groupby"]

Return only JSON, no other commentary."""

def plan_sanity_agent_prompt(question: str,
                             schema_output: str,
                             plan: str,
                             clause_prompts="",
                             subproblem_output="") -> str:
    base = Template(
        """
You are a **Plan Sanity Checker Agent** in an NL2SQL framework. Your task is to read the question, review the SQL plan and ensure it **structurally matches** the intended intent from the schema and subproblem breakdown.

Inputs:
Question:
$question

Selected Schema Lines:
$schema_output
$subproblems

Original Plan to Validate:
$plan

Check that:
- All necessary tables appear in `FROM` and `JOIN` steps with correct join keys.
- GROUP BY, HAVING, ORDER BY, LIMIT clauses appear in the proper order and ONLY when they're necessary.
- Filter operations (WHERE/HAVING) match the subproblem intent.
- There are no extra tables or columns beyond intent.

$clause_prompts
Remove the issues from the original plan, and summarize the revised plan + error highlights into a few sentences. Pass that summary instead of the raw history.
Example: “Plan summary: joined tables A–B–C via columns X.Y and Y.Z; GROUP BY on col Z. Found missing join to table D on A.K = D.K.”
""")
    """
Return either:
1. `VALID PLAN` if Original plan is correct
2. If you spot issues, return a revised plan, step-by-step, with minimal edits, numbered lines, and nothing else.
    """
    if subproblem_output != "":
        subproblems = "\nSubproblems Identified: \n" + subproblem_output
    else:
        subproblems = ""
    return base.substitute(
        question=question,
        schema_output=schema_output,
        subproblems=subproblems, clause_prompts=clause_prompts,
        plan=plan,
    )


def repair_agent_prompt(issue: str, sql: str, question: str, schema):
    return f"""
You are a Repair Agent in an NL2SQL framework. There’s a {issue} error. Given the question, relevant table schema, the generated SQL, fix the error.

Question:
{question}
Schema:
{schema}
Generated SQL:
{sql}

Fix only the {issue}, keeping other logic intact.
"""

def repair_combined_agent_prompt(repaired_sqls, question):
    lines = []
    for issue, sql in repaired_sqls.items():
        lines.append(f"Issue: {issue}\nRepaired SQL:\n{sql}")
    body = "\n\n".join(lines)
    return f"""
You are a Repair Combining Agent in an NL2SQL framework. Different repair agents each fixed a specific issue in the SQL query. Combine their outputs into one coherent final SQL. 
Make sure to KEEP ALL FIXES while avoiding conflicts. 

Original Question:
{question}

Available fixed versions:
{body}

Produce a single SQL query that integrates all corrections. Return only the final SQL.
"""
