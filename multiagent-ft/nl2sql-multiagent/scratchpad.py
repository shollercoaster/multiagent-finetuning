# In utils.py

import json

class Scratchpad:
    """
    A shared scratchpad to maintain the state and history of a single NL2SQL problem.
    It's primarily designed to provide rich context to the correction loop agents.
    """
    def __init__(self, question: str, db_id: str, schema: str):
        self.question = question
        self.db_id = db_id
        self.schema = schema
        self.plan = ""
        self.attempts = []  # A list to store the history of (sql, error, plan) tuples

    def add_attempt(self, sql: str, error: str, correction_plan: str = ""):
        """Adds a new attempt to the history."""
        self.attempts.append({
            "sql": sql,
            # "execution_error": error,
            "correction_plan": correction_plan
        })

    def get_correction_context(self) -> str:
        """
        Formats the history of the correction loop into a string for the prompt.
        This provides the agent with the full context of what has already been tried.
        """
        if not self.attempts:
            return "No previous attempts have been made."

        context_str = "Here is the history of previous failed attempts. Learn from them to generate a correct query.\n\n"
        # We iterate through all previous attempts to give full context
        for i, attempt in enumerate(self.attempts):
            context_str += f"--- Attempt #{i + 1} ---\n"
            context_str += f"Failed SQL: \n```sql\n{attempt['sql']}\n```\n"
            # context_str += f"Database Execution Error: \n`{attempt['execution_error']}`\n"
            if attempt['correction_plan']:
                context_str += f"Correction Plan for this attempt was: \n{attempt['correction_plan']}\n"
            context_str += "---\n\n"
        
        return context_str
