"""
sql_tool.py
-----------
LangChain tools for SQL query generation and execution.
"""

import json
from langchain.tools import tool
from database.db_connector import DatabaseConnector


def get_sql_tools(connector: DatabaseConnector):
    """Returns SQL and schema tools for the LangChain agent."""

    @tool
    def sql_query_tool(query: str) -> str:
        """
        Executes a SQL query against the connected database.

        Use this tool to answer data-related questions by running
        a valid SELECT statement.

        Input: A complete, executable SQL SELECT statement
        Output: Query results as JSON string or an error message
        """
        # Only allow SELECT statements for safety
        if not query.strip().upper().startswith("SELECT"):
            return "Error: Only SELECT queries are allowed."

        try:
            results = connector.execute_query(query)

            if not results:
                return "The query returned no results."

            return json.dumps(results, default=str, ensure_ascii=False, indent=2)

        except ValueError as e:
            # Return error with hint so the agent can self-correct
            return (
                f"SQL Error: {e}\n"
                f"Please check table and column names and try again."
            )

    @tool
    def get_schema_tool(dummy: str = "") -> str:
        """
        Returns the full database schema (tables, columns, foreign keys).

        Always call this tool first before writing SQL queries
        to know which tables and columns are available.

        Input: Any string (ignored)
        Output: Full schema as formatted text
        """
        try:
            return connector.get_schema()
        except Exception as e:
            return f"Schema error: {e}"

    return sql_query_tool, get_schema_tool