"""Common utilities for NL2SQL single‑ and multi‑agent LoRA experiments."""
from pathlib import Path
import json, functools
from typing import Dict, List, Any

##############################
# Prompt templates ###########
##############################

PROMPT_TMPL_SINGLE = (
    "### Instruction:\n"
    "You are an NL2SQL agent. Generate a valid SQLite query that answers the question.\n\n"
    "### Schema:\n{schema}\n\n"
    "### Question:\n{question}\n\n"
    "### SQL:\n"
)

PROMPT_TMPL_SCHEMA = (
    "### Instruction:\n"
    "You are a schema‑selection agent. List the table and column names needed to answer the question.\n\n"
    "### Schema:\n{schema}\n\n"
    "### Question:\n{question}\n\n"
    "### NeededColumns (one per line):\n"
)

PROMPT_TMPL_SQL = (
    "### Instruction:\n"
    "You are an SQL generation agent. Given the question AND the list of needed tables/columns, write a valid SQLite query.\n\n"
    "### Schema:\n{schema}\n\n"
    "### Question:\n{question}\n\n"
    "### NeededColumns:\n{cols}\n\n"
    "### SQL:\n"
)

########################################
# Schema loader from *_tables.json #####
########################################

USING_SPIDER_DS = True # set to False when using Bird-SQL

@functools.lru_cache()
def _schema_map(split_root: Path):
    if USING_SPIDER_DS:
        table_file = split_root / "tables.json"
    else:
        table_file = next(split_root.glob("*_tables.json"))

    with open(table_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    mapping = {}
    for db in raw:
        tbl_names = db["table_names"]
        cols_by_tbl = {i: [] for i in range(len(tbl_names))}
        for tbl_idx, col_name in db["column_names"]:
            if tbl_idx >= 0:
                cols_by_tbl[tbl_idx].append(col_name)
        mapping[db["db_id"]] = [
            {"table_name": tbl_names[i], "column_names": cols_by_tbl[i]} for i in range(len(tbl_names))
        ]
    return mapping

def attach_schema_json(ex: Dict[str, Any], split_root: Path) -> Dict[str, Any]:
    """Add list‑of‑dict 'schema' key to dataset example using schema JSON."""
    ex["schema"] = _schema_map(split_root)[ex["db_id"]]
    # For Spider: normalize SQL key name
    if USING_SPIDER_DS and "query" in ex:
        ex["SQL"] = ex["query"]
    return ex

########################################
# Prompt builders ######################
########################################

def _fmt_schema(schema: List[Dict[str, Any]]) -> str:
    return "\n".join(f"{t['table_name']}: {', '.join(t['column_names'])}" for t in schema)


def build_single_prompt(ex: Dict[str, Any]) -> str:
    return PROMPT_TMPL_SINGLE.format(schema=_fmt_schema(ex["schema"]), question=ex["question"])


def build_schema_prompt(ex: Dict[str, Any]) -> str:
    return PROMPT_TMPL_SCHEMA.format(schema=_fmt_schema(ex["schema"]), question=ex["question"])


def build_sql_prompt(ex: Dict[str, Any], needed_cols: str) -> str:
    return PROMPT_TMPL_SQL.format(schema=_fmt_schema(ex["schema"]), question=ex["question"], cols=needed_cols.strip())
