import json, os, re
# from openai import OpenAI
from typing import List, Dict
from prompts import (
    schema_agent_prompt, subproblem_agent_prompt,
    query_plan_agent_prompt, sql_agent_prompt,
    critic_agent_prompt, OPENAI_API_KEY
)
from prompts import *
import sqlite3
from subprocess import Popen, PIPE
from datetime import datetime

# client = OpenAI(api_key=OPENAI_API_KEY)

def call_agent(prompt: str, temperature: float = 0.0) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return resp.choices[0].message.content.strip()

def postprocess_sql(sql: str) -> str:
    sql = sql.strip().lower()
    sql = re.sub(r"```sql|```", "", sql)
    sql = re.sub(r"^sql[:\s]*", "", sql)
    sql = re.sub(r"```sql|```", "", sql)
    sql = sql.replace("`", "") # .replace("'", "").replace("\"", "") # issue in edgecases like properName = 'Saumya' but it will remove quotes
    sql = re.sub(r"\s+", " ", sql).strip()
    sql = re.sub(r"\s+,", ",", sql)
    if sql.endswith(";"):
        sql = sql[:-1].strip()
    return sql

def check_valid_critic_and_push_error(sql: str, question: str, db_id: str,
                    taxonomy: dict,
                    error_db_path="error_db.jsonl") -> (bool, list):
    # Call critic agent with taxonomy context
    prompt = taxonomy_critic_agent_prompt(question, sql, taxonomy)
    response = call_agent(prompt).strip()
    # print(f"\n CRITIC RESPONSE: {response}\n")
    try:
        response = clean_json_prefix(response).strip()
        print(f"\n CRITIC RESPONSE: {response}\n")
        parsed = json.loads(response)
    except json.JSONDecodeError:
        # In case of malformed JSON, treat as error
        return False, ["critic.json_decode_error"]

    valid = parsed.get("valid", False)
    error_types = parsed.get("error_types", [])

    # If invalid, append entry to NDJSON error DB
    if not valid:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "db_id": db_id,
            "question": question,
            "sql": sql,
            "error_types": error_types
        }
        with open(error_db_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, separators=(",",":")) + "\n")

    return valid, error_types

def is_critic_valid(sql: str, question: str, db_id: str, error_db_path="error_db.json") -> (bool, list):
    try:
        response = call_agent(critic_agent_prompt(sql, question))
        parsed = json.loads(response)
        valid = parsed.get("valid", False)
        error_types = parsed.get("error_types", [])

        # If invalid, write to error_db.json
        if not valid:
            entry = {
                "timestamp": str(datetime.now()),
                "db_id": db_id,
                "question": question,
                "sql": sql,
                "label": False,
                "error_types": error_types
            }
            if os.path.exists(error_db_path):
                db = json.load(open(error_db_path))
            else:
                db = []
            db.append(entry)
            # with open(error_db_path, "w") as f:
                # json.dump(db, f, indent=2)

        return valid, error_types

    except Exception as e:
        return False, [{"error_type": f"Exception during critic evaluation: {str(e)}"}]


def load_spider(dev: bool = True, testing=False) -> List[Dict]:
    path = "../../spider/dev.json" if dev else "../../spider/train_spider.json"
    if testing == True:
        path = "testing_limit.json"
    with open(path) as f:
        return json.load(f)


def load_schema_without_PKFK(db_id: str) -> str:
    db_dir = os.path.join("../../spider/database", db_id)
    # sql_path = os.path.join(f"../../spider/database/{db_id}/schema.sql")
    print("Directory listing:", os.listdir(f"../../spider/database/{db_id}"))
    sql_files = [f for f in os.listdir(db_dir) if f.endswith(".sql")]
    # if not os.path.exists(sql_path):
        # return ""
    if not sql_files:
        return ""

    sql_path = os.path.join(db_dir, sql_files[0])
    with open(sql_path, "r") as f:
        lines = f.readlines()

    schema_lines = []
    current_table = None
    inside_table = False

    for line in lines:
        line = line.strip()
        if line.upper().startswith("CREATE TABLE"):
            parts = line.split()
            if len(parts) >= 3:
                current_table = parts[2].strip('"`[]')
                schema_lines.append(f"{current_table}:")  # New table
                inside_table = True
        elif inside_table:
            if line.startswith(")"):
                inside_table = False
                current_table = None
            elif line and not line.upper().startswith("PRIMARY KEY") and not line.upper().startswith("FOREIGN KEY"):
                # Parse column name
                col_name = line.split()[0].strip('"`[]').rstrip(",")
                if col_name:
                    schema_lines.append(f"  {col_name}")
    
    return "\n".join(schema_lines)


# NL2SQL bugs file
# BUGS_DB = open("../../nl2sql_bugs.json").read()

def exec_query(db_file: str, sql: str):
    conn = sqlite3.connect(db_file)
    try:
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return rows, None
    except Exception as e:
        return None, str(e)
    finally:
        conn.close()

def clean_json_prefix(text: str) -> str:
    """
    Strips all characters before the first '{', including backticks or the word 'json'.
    """
    # Regex: find the first '{' and return everything from there
    match = re.search(r'\{', text)
    if not match:
        raise ValueError("No JSON object found in critic response")
    trimmed = text[match.start():]

    # Optional: remove wrapping backticks ``` or single quotes
    trimmed = re.sub(r'^```+', '', trimmed)
    trimmed = re.sub(r"^['\"]+", '', trimmed)
    end = text.rfind('}')
    json_str = trimmed[:end+1]
    json_str = json_str.strip("`\n\r ")
    return json_str

def load_schema(db_id: str) -> str:
    db_path = f"../../spider/database/{db_id}/{db_id}.sqlite"
    if not os.path.exists(db_path):
        print("[load_schema] DB not found:", db_path)
        return ""

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys=ON;")
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = [row[0] for row in cur.fetchall()]

    schema_lines = []
    for tbl in tables:
        schema_lines.append(f"{tbl}:")
        # Get columns & PK
        cur.execute(f"PRAGMA table_info({tbl});")
        cols = cur.fetchall()
        col_names = [c[1] for c in cols]
        pks = [c[1] for c in cols if c[5] > 0]
        schema_lines.append(f"  Columns: {', '.join(col_names)}")
        if pks:
            schema_lines.append(f"  Primary Key: {', '.join(pks)}")
        # Get FKs
        cur.execute(f"PRAGMA foreign_key_list({tbl});")
        fks = cur.fetchall()
        if fks:
            schema_lines.append(f"  Foreign Keys:")
            for fk in fks:
                _, _, ref_tbl, from_col, to_col, *_ = fk
                schema_lines.append(f"    - {from_col} → {ref_tbl}.{to_col}")

    conn.close()
    joined = "\n".join(schema_lines)
    print("[load_schema] Schema:\n" + joined)
    return joined

def clause_specific_prompts(clauses):
    plan, sql = "", ""
    for clause in clauses:
        clause = clause.upper()
        if clause == "HAVING" or clause == "GROUPBY" or clause == "GROUP BY":
            plan += """
    GROUP BY detected:
    - All non-aggregated SELECT columns must be in GROUP BY.
    - GROUP BY should appear after WHERE but before HAVING/ORDER BY.

    If HAVING is present:
    - Use HAVING to filter on aggregates, not WHERE.
    """
            sql += """
    Ensure:
    - All non-aggregated SELECT columns are in GROUP BY.
    - HAVING filters only aggregated expressions.
    - HAVING appears after GROUP BY.
    - Use WHERE for pre-aggregation filters only.
    """


        if clause == "ORDERBY" or clause == "ORDER BY":
            plan += """
    ORDER BY detected:
    - Specify column(s) to sort on with direction (ASC/DESC).
    - ORDER BY should be planned after WHERE/ GROUP BY / HAVING steps.
    - If LIMIT or OFFSET is present, ORDER BY must come before them.
    """
            sql+= """
    Ensure:
    - ORDER BY references valid columns (or aliases defined in SELECT/grouping).
    - ORDER BY is placed after GROUP BY or HAVING if those exist.
    - If LIMIT is used, ORDER BY must guarantee deterministic results.
                    """


        if clause == "LIMIT":
            plan += """
    LIMIT detected:
    - Decide which rows are returned: use ORDER BY to define which subset is used.
    - Plan ORDER BY step before LIMIT to ensure consistent results.
                    """
            sql+= """
    Ensure:
    - Use ORDER BY before LIMIT for deterministic row selection.
    - LIMIT appears as the final clause after ORDER BY.
    """

        if clause == "JOIN":
            plan += """
    OIN detected:
    - Plan all necessary JOINs between tables, listing each table and ON condition.
    - Each JOIN must reference valid foreign key paths from schema.
    - Avoid Cartesian products: every JOIN must include a precise ON clause.
    """
            sql+= """
    Ensure:
    - Include all tables referenced in the plan via JOINS.
    - Each JOIN uses correct foreign key column(s) in ON clause.
    - Do not introduce unintended full joins or missing JOIN conditions.
    """

        if clause == "UNION":
            plan += """
    UNION detected:
    - Both subqueries must select the same number of columns with compatible types.
    - Specify UNION vs UNION ALL depending on whether duplicates should be removed.
    - Plan ORDER BY / LIMIT after the entire UNION block.
    """
            sql+= """
    Ensure:
    - Each UNION branch has identical column count and data types.
    - Use DISTINCT (default UNION) or ALL explicitly.
    - If ORDER BY or LIMIT is applied, apply it only at the end of the UNION output.
    """

        if clause == "INTERSECT":
            plan += """
    📌 INTERSECT detected:
    - Both queries must select the same number and type of columns.
    - Plan for duplicates: INTERSECT removes duplicates unless INTERSECT ALL is specified.
    - ORDER BY / LIMIT clauses apply after the intersect.
    """
            sql+= """
    Ensure:
    - Each INTERSECT branch selects same number and types of columns.
    - Use INTERSECT or INTERSECT ALL as needed.
    - Place ORDER BY and LIMIT after the intersect expression.
    """

        if clause == "EXCEPT":
            plan += """
    📌 EXCEPT detected:
    - Both queries must select the same number and type of columns.
    - Plan which side to apply EXCEPT (left - right rows).
    - ORDER BY / LIMIT should be planned after the EXCEPT block.
    """
            sql += """
    Ensure:
    - EXCEPT branches share identical column count/types.
    - Use EXCEPT or EXCEPT ALL appropriately.
    - Apply ORDER BY and LIMIT only to the final output of the EXCEPT.
    """

    return plan, sql