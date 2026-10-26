"""
multi_db_sql_tool.py
--------------------
LangChain tools for querying multiple databases simultaneously.
The agent specifies which database to query by name.
"""

import json
from typing import Union
from langchain.tools import tool
from database.multi_db_connector import MultiDatabaseConnector


def get_multi_db_tools(multi_connector: MultiDatabaseConnector):
    """Returns SQL and schema tools that support multiple databases."""

    @tool
    def multi_db_schema_tool(dummy: str = "") -> str:
        """
        Returns the schemas of ALL connected databases.

        Always call this tool first to understand which databases
        are available and what tables/columns each contains.

        Input: Any string (ignored)
        Output: Combined schema of all connected databases
        """
        return multi_connector.get_all_schemas()

    @tool
    def multi_db_query_tool(input_json: Union[str, dict]) -> str:
        """
        Executes a SQL query on a specific database.

        Use this tool to query any of the connected databases.
        You must specify the database name and the SQL query.

        Input: JSON string with:
            - db_name: name of the database to query
            - query: SQL SELECT statement to execute

        Example input:
        {
            "db_name": "sales",
            "query": "SELECT COUNT(*) FROM orders"
        }

        Output: Query results as JSON or error message
        """
        # GPT-4o (and some other models) pass a native dict via function-calling
        # instead of a JSON string.  Accept both forms.
        if isinstance(input_json, dict):
            params = input_json
        else:
            try:
                params = json.loads(input_json)
            except json.JSONDecodeError as e:
                return f"Invalid JSON input: {e}"

        db_name = params.get("db_name")
        query = params.get("query")

        if not db_name or not query:
            return "Missing required fields: 'db_name' and 'query'"

        # Only allow SELECT statements
        if not query.strip().upper().startswith("SELECT"):
            return "Error: Only SELECT queries are allowed."

        try:
            results = multi_connector.execute_query(db_name, query)

            if not results:
                return f"Query on '{db_name}' returned no results."

            return json.dumps(results, default=str, ensure_ascii=False, indent=2)

        except ValueError as e:
            return str(e)
        except Exception as e:
            return (
                f"SQL Error on '{db_name}': {e}\n"
                f"Please check table/column names and try again."
            )

    @tool
    def list_databases_tool(dummy: str = "") -> str:
        """
        Returns a list of all currently connected databases.

        Use this tool to check which databases are available
        before querying.

        Input: Any string (ignored)
        Output: List of connected database names
        """
        names = multi_connector.get_connection_names()
        if not names:
            return "No databases connected."
        return f"Connected databases: {', '.join(names)}"

    return multi_db_schema_tool, multi_db_query_tool, list_databases_tool