"""
python_tool.py
--------------
LangChain tool for Python code generation and execution.
Handles data analysis and visualization tasks.
"""

import io
import sys
import traceback
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Streamlit
import matplotlib.pyplot as plt
import pandas as pd
from langchain.tools import tool


# Allowed modules for code execution
ALLOWED_MODULES = {"pandas", "numpy", "matplotlib", "seaborn", "json", "math"}


def get_python_tool():
    """Returns the Python execution tool for the LangChain agent."""

    @tool
    def python_code_tool(code: str) -> str:
        """
        Executes Python code for data analysis and visualization.

        Use this tool to perform statistical analysis, data transformations,
        or generate charts from data retrieved via SQL.

        Input: Valid Python code as a string
        Output: stdout output, or an error message if execution fails

        Notes:
        - Use 'data' variable if passing results from sql_query_tool
        - For plots, use plt.savefig('output.png') to save figures
        - pandas and numpy are available as pd and np
        """
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            # Execution context with pre-imported libraries
            exec_globals = {
                "pd": pd,
                "plt": plt,
                "__builtins__": __builtins__,
            }

            # numpy and seaborn imported conditionally
            try:
                import numpy as np
                import seaborn as sns
                exec_globals["np"] = np
                exec_globals["sns"] = sns
            except ImportError:
                pass

            exec(code, exec_globals)

            output = sys.stdout.getvalue()
            return output if output else "Code executed successfully (no output)."

        except Exception:
            error = traceback.format_exc()
            return f"Execution error:\n{error}"

        finally:
            sys.stdout = old_stdout
            plt.close('all')  # Clean up figures

    return python_code_tool