import os
from string import Template

# Load your API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 1. Schema Agent prompt
def schema_agent_prompt(question: str, schema_list: str) -> str:
    return Template(
        """
You are a Schema Agent. Given a natural language question and table schemas, identify the relevant tables and columns.

Question: $question
Schemas:
$schema_list

Return a list of lines in the format:
Table: col1, col2, ...
Only list relevant tables and columns needed to answer the question.
"""
    ).substitute(question=question, schema_list=schema_list)

# 2. Subproblem Agent prompt
def subproblem_agent_prompt(sql: str) -> str:
    return Template(
        """
You are a Subproblem Agent. Given a SQL query, decompose it into subproblems or expressions.

SQL: $sql

Identify key clauses (e.g., GROUP BY, ORDER BY, UNION, JOIN) and list subproblems in JSON form:
{"subproblems": [{"clause": "GROUP BY", "expression": "..." }, ...]}
Only output valid JSON.
"""
    ).substitute(sql=sql)

# 3. Query Plan Agent prompt
def query_plan_agent_prompt(question: str, schema_info: str, subproblem_json: str) -> str:
    return Template(
        """
You are a Query Plan Agent. Using the question, schema info, and subproblems, generate a step-by-step SQL query plan.

Question: $question
Schema Info:
$schema_info
Subproblems:
$subproblem_json

Provide a concise and to-the-point plan, each step describing how to build parts of the SQL.
"""
    ).substitute(question=question, schema_info=schema_info, subproblem_json=subproblem_json)

# 4. SQL Generating Agent prompt
def sql_agent_prompt(plan: str) -> str:
    return Template(
        """
You are an SQL Generating Agent. Given the plan, write ONLY the final SQL query with no extra text or formatting.

Plan:
$plan

Return exactly one valid SQL statement.
"""
    ).substitute(plan=plan)

# 5. Critic Agent prompt
def critic_agent_prompt(sql: str, bug_list: str) -> str:
    return Template(
        """
You are a Critic Agent. Validate the SQL query against known NL2SQL bug patterns.

Original SQL: $sql
Known bugs database: $bug_list

If you find errors, output JSON:{"error": "<description>"}
Otherwise output:{"valid": true}
"""
    ).substitute(sql=sql, bug_list=bug_list)
