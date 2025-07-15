import json, os, re
from openai import OpenAI
from typing import List, Dict
from prompts import (
    schema_agent_prompt, subproblem_agent_prompt,
    query_plan_agent_prompt, sql_agent_prompt,
    critic_agent_prompt, OPENAI_API_KEY
)
import sqlite3
from subprocess import Popen, PIPE

client = OpenAI(api_key=OPENAI_API_KEY)

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

def load_spider(dev: bool = True) -> List[Dict]:
    path = "../../spider/dev.json" if dev else "../../spider/train_spider.json"
    with open(path) as f:
        return json.load(f)


def load_schema(db_id: str) -> str:
    sql_path = os.path.join(f"../../spider/database/{db_id}/schema.sql")
    if not os.path.exists(sql_path):
        return ""
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
BUGS_DB = open("../../nl2sql_bugs.json").read()

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

print(postprocess_sql("sql\nSELECT COUNT(Singer_ID) AS NumberOfSingers FROM singer;\n"))
