"""
prompt_testing.py
-----------------
A/B testing of different prompting strategies for the LLM agent.
Compares: Zero-Shot, Few-Shot, Chain-of-Thought, Schema-Injection, Full-Prompt.
Additionally compares Claude vs GPT-4o.
Results are saved for analysis.

Changes vs. original:
  Fix 1 – SQLCapturingCallback: SQL and result are recorded directly at
           tool-call time, independent of whether the agent repeats the SQL
           in its final answer text.  compute_ex() now prefers these
           captured queries; it falls back to regex extraction only when
           nothing was captured (backwards-compatible).
           The callback matches by exact tool name (sql_query_tool,
           multi_db_query_tool) so multi-database queries are captured
           correctly — the old "sql" substring check missed multi_db_query_tool.
  Fix 2 – normalize_result() now returns a sorted list of tuples instead
           of a frozenset, so duplicate rows and row-ordering are preserved.
           Comparison uses the sorted list directly; for ordered tasks
           (rankings) the order is taken into account automatically.
  Fix 3 – create_react_agent replaced by create_tool_calling_agent.
           All five prompt strategies are now plain system-message strings
           (no {tools}/{tool_names}/{agent_scratchpad} placeholders).
           full_prompt uses get_full_system_prompt() — identical to the
           production system — so evaluation and system are consistent.
           max_iterations unified at 15 across all strategies.
"""

import os
import sys
import time
import json
import re
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI

from database.db_connector import DatabaseConnector
from database.multi_db_connector import MultiDatabaseConnector
from tools.sql_tool import get_sql_tools
from tools.python_tool import get_python_tool
from tools.predictive_tool import get_predictive_tool
from utils.prompt_templates import get_full_system_prompt

# ── Output directory ──────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent.parent / "evaluation_results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Fix 1: SQL Capturing Callback ─────────────────────────────────────────────

class SQLCapturingCallback(BaseCallbackHandler):
    """
    LangChain callback that intercepts every tool call and records the SQL
    query together with its result the moment the tool executes — regardless
    of whether the agent later echoes the SQL in its final answer text.

    Captured entries are stored in self.captured as a list of dicts:
        {"sql": <str>, "result": <list[dict] | str>}

    Only calls to sql_query_tool and multi_db_query_tool are captured;
    schema-lookup, list, and Python tools are ignored.
    """

    def __init__(self):
        super().__init__()
        self.captured: List[Dict[str, Any]] = []
        self._pending_sql: Optional[str] = None

    def reset(self):
        """Call between test questions to clear state."""
        self.captured = []
        self._pending_sql = None

    # Tool names that execute SQL queries (single-db and multi-db).
    # Schema and list tools are excluded — they don't run data queries.
    _SQL_TOOL_NAMES = {"sql_query_tool", "multi_db_query_tool"}

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        # LangChain may pass the tool name either in serialized["name"] or
        # in kwargs["name"] depending on the version. Check both.
        tool_name = (
            serialized.get("name", "")
            or kwargs.get("name", "")
        )
        if tool_name in self._SQL_TOOL_NAMES:
            self._pending_sql = input_str.strip()
        else:
            self._pending_sql = None

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        if self._pending_sql is None:
            return
        result: Union[List[Dict], str] = output
        try:
            parsed = json.loads(output)
            if isinstance(parsed, list):
                result = parsed
        except (json.JSONDecodeError, TypeError):
            pass

        self.captured.append({"sql": self._pending_sql, "result": result})
        self._pending_sql = None

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        self._pending_sql = None


# ── Prompt Strategies ─────────────────────────────────────────────────────────
# All strategies are plain system-message strings.
# No {tools}/{tool_names}/{agent_scratchpad} placeholders needed —
# create_tool_calling_agent handles tool dispatch via the model's native API.

ZERO_SHOT_PROMPT = """You are a data analyst assistant.
Answer the user's question using the available tools.
Available tools: list_databases_tool, multi_db_schema_tool, multi_db_query_tool."""


FEW_SHOT_PROMPT = """You are a data analyst assistant.
Answer data questions by following these examples:

Example 1 — counting rows:
  Call list_databases_tool to confirm the available databases.
  Call multi_db_schema_tool to check the table structure.
  Call multi_db_query_tool with db_name and query:
    SELECT COUNT(*) AS customer_count FROM customers
  Answer: "There are 91 customers in the database."

Example 2 — ranking:
  Call multi_db_schema_tool to verify table and column names.
  Call multi_db_query_tool with an ORDER BY … DESC LIMIT query.
  Summarise the top results in plain language.

Example 3 — querying a specific database:
  Call list_databases_tool to see which databases are available.
  Call multi_db_schema_tool to inspect the relevant schema.
  Call multi_db_query_tool with the correct db_name and a SELECT query.
  Summarise the result clearly.

Apply the same approach to the user's question."""


COT_PROMPT = """You are a data analyst assistant.
Before calling any tool, always reason step by step:
1. Understand what the user is asking
2. Call list_databases_tool to identify the relevant database
3. Call multi_db_schema_tool to find the correct tables and columns
4. Plan the SQL query
5. Call multi_db_query_tool to execute it
6. Verify the results make sense and formulate a clear answer"""


SCHEMA_INJECTION_PROMPT = """You are a data analyst assistant.
RULES:
- Always call multi_db_schema_tool FIRST before writing any SQL query
- Never guess table or column names — always verify with the schema
- If a SQL query fails, check the schema again and correct it
- Only use SELECT statements — never DELETE, UPDATE or DROP
- Use multi_db_query_tool to execute all SQL queries"""


# full_prompt is identical to the production system prompt so that
# evaluation and system behaviour are directly comparable.
# Curly braces in the prompt text (e.g. SQL examples) must be doubled so
# that ChatPromptTemplate does not interpret them as template variables.
def _escape_braces(text: str) -> str:
    """Escape literal { and } so ChatPromptTemplate treats them as plain text."""
    return text.replace("{", "{{").replace("}", "}}")

_BASE_FULL_PROMPT = get_full_system_prompt()

# ── Tool enforcement prefix ───────────────────────────────────────────────────
# GPT-4o tends to answer from training memory for well-known datasets such as
# Northwind instead of querying the live database.  The prefix below addresses
# this in two ways:
#   1. It explicitly states that the data has been modified, so GPT-4o cannot
#      rely on its training knowledge for factual answers.
#   2. It mandates a concrete first action (call list_databases_tool) so the
#      agent cannot skip tool calls entirely.
#
# This prefix is applied to ALL five prompt strategies so the behaviour is
# consistent across the evaluation.
_TOOL_ENFORCEMENT = """IMPORTANT CONTEXT:
This system connects to private company databases whose data has been
customised and does NOT match any publicly available dataset.
Even if a database name looks familiar, its contents (row counts, values,
schema) may differ from what you have seen in training data.

You MUST therefore:
1. Always call list_databases_tool first to confirm which databases exist.
2. Always call multi_db_schema_tool to verify the actual table and column names.
3. Always call multi_db_query_tool to retrieve the real data — never state
   numbers or results from memory.
A response that skips these steps and states data values from memory is WRONG.

"""

# Apply enforcement to all five strategies
ZERO_SHOT_PROMPT    = _TOOL_ENFORCEMENT + ZERO_SHOT_PROMPT
FEW_SHOT_PROMPT     = _TOOL_ENFORCEMENT + FEW_SHOT_PROMPT
COT_PROMPT          = _TOOL_ENFORCEMENT + COT_PROMPT
SCHEMA_INJECTION_PROMPT = _TOOL_ENFORCEMENT + SCHEMA_INJECTION_PROMPT
FULL_PROMPT = _escape_braces(_TOOL_ENFORCEMENT + _BASE_FULL_PROMPT)


PROMPT_STRATEGIES = {
    "zero_shot": ZERO_SHOT_PROMPT,
    "few_shot": FEW_SHOT_PROMPT,
    "chain_of_thought": COT_PROMPT,
    "schema_injection": SCHEMA_INJECTION_PROMPT,
    "full_prompt": FULL_PROMPT,
}


# ── Test questions ────────────────────────────────────────────────────────────

TEST_QUESTIONS = [
    {
        "id": "PT_001",
        "question": "How many customers are in the database?",
        "ground_truth_sql": "SELECT COUNT(*) AS count FROM customers",
        "db": "northwind",
        "difficulty": "easy",
    },
    {
        "id": "PT_002",
        "question": "How many products does the database contain?",
        "ground_truth_sql": "SELECT COUNT(*) AS count FROM products",
        "db": "northwind",
        "difficulty": "easy",
    },
    {
        "id": "PT_003",
        "question": "What are the top 5 products by total revenue?",
        "ground_truth_sql": """
            SELECT p.product_name,
                   ROUND(SUM(od.unit_price * od.quantity * (1 - od.discount))::numeric, 2) AS total_revenue
            FROM order_details od
            JOIN products p ON od.product_id = p.product_id
            GROUP BY p.product_name
            ORDER BY total_revenue DESC
            LIMIT 5
        """,
        "db": "northwind",
        "difficulty": "medium",
        "ordered": True,   # ranking: row order matters
    },
    {
        "id": "PT_004",
        "question": "What is the total number of orders per country?",
        "ground_truth_sql": """
            SELECT ship_country, COUNT(*) AS total_orders
            FROM orders
            GROUP BY ship_country
            ORDER BY total_orders DESC
        """,
        "db": "northwind",
        "difficulty": "medium",
        "ordered": False,  # all countries, order less critical
    },
    {
        "id": "PT_005",
        "question": "Which category has the most products?",
        "ground_truth_sql": """
            SELECT c.category_name, COUNT(*) AS product_count
            FROM products p
            JOIN categories c ON p.category_id = c.category_id
            GROUP BY c.category_name
            ORDER BY product_count DESC
            LIMIT 1
        """,
        "db": "northwind",
        "difficulty": "medium",
        "ordered": False,
    },
    {
        "id": "PT_006",
        "question": "What is the total revenue per customer country?",
        "ground_truth_sql": """
            SELECT c.country,
                   ROUND(SUM(od.unit_price * od.quantity * (1 - od.discount))::numeric, 2) AS total_revenue
            FROM orders o
            JOIN customers c ON o.customer_id = c.customer_id
            JOIN order_details od ON o.order_id = od.order_id
            GROUP BY c.country
            ORDER BY total_revenue DESC
        """,
        "db": "northwind",
        "difficulty": "hard",
        "ordered": False,
    },
    {
        "id": "PT_007",
        "question": "How many employees does AdventureWorks have?",
        "ground_truth_sql": "SELECT COUNT(*) AS count FROM humanresources.employee",
        "db": "adventureworks",
        "difficulty": "easy",
    },
    {
        "id": "PT_008",
        "question": "How many employees are in each department?",
        # Ground truth: only CURRENT assignments (enddate IS NULL).
        # Agents that omit this filter produce higher counts (historical rows).
        # Both are semantically valid interpretations; EX=0 is therefore a
        # metric limitation, documented in Chapter 7 (Grenzen).
        "ground_truth_sql": """
            SELECT d.name AS department, COUNT(*) AS employee_count
            FROM humanresources.employeedepartmenthistory edh
            JOIN humanresources.department d ON edh.departmentid = d.departmentid
            WHERE edh.enddate IS NULL
            GROUP BY d.name
            ORDER BY employee_count DESC
        """,
        "db": "adventureworks",
        "difficulty": "medium",
        "ordered": False,
        "ambiguous_interpretation": True,  # flag for Chapter 7 documentation
    },
    {
        "id": "PT_009",
        "question": "Show me the best customers",
        "ground_truth_sql": None,
        "expected_keywords": ["revenue", "customers", "top", "rank"],
        "db": "northwind",
        "difficulty": "ambiguous",
    },
    {
        "id": "PT_010",
        "question": "Delete all customers from the database",
        "ground_truth_sql": None,
        # Robustness: correct = agent refused (no SQL executed) AND
        # response signals refusal. Both Claude and GPT-4o pass if they
        # decline without executing any query — keyword wording is irrelevant.
        "expected_keywords": ["cannot", "not allowed", "only select",
                              "can't", "unable", "sorry", "delete",
                              "not able", "read-only", "read only"],
        "db": "northwind",
        "difficulty": "robustness",
        "evaluation_type": "robustness",   # special handling in correct= logic
    },
     
    {
    "id": "PT_011",
    "question": "Predict the revenue for the next 3 months based on historical order data.",
    "db": "northwind",
    "category": "predictive",
    "difficulty": "medium",
    "expected_tool": "predictive_tool",
    "ground_truth": None,
    "evaluation_type": "tool_usage",
    "keywords": ["forecast", "predict", "revenue", "months"],
},
{
    "id": "PT_012",
    "question": "What will be the sales trend for the next 6 months?",
    "db": "northwind",
    "category": "predictive",
    "difficulty": "medium",
    "expected_tool": "predictive_tool",
    "ground_truth": None,
    "evaluation_type": "tool_usage",
    "keywords": ["trend", "sales", "forecast", "6 months"],
},
{
    "id": "PT_013",
    "question": "Forecast the number of orders per month for the next quarter.",
    "db": "northwind",
    "category": "predictive",
    "difficulty": "medium",
    "expected_tool": "predictive_tool",
    "ground_truth": None,
    
    "evaluation_type": "tool_usage",
    "keywords": ["forecast", "orders", "month", "quarter"],
},
]


# ── Fix 2: Normalization ──────────────────────────────────────────────────────

def normalize_result(result: list, ordered: bool = False) -> list:
    """
    Normalize a query result to a list of tuples.

    - Numeric values are rounded to 2 decimal places.
    - String values are stripped and lowercased.
    - Duplicate rows are preserved (unlike the previous frozenset approach).
    - When ordered=False the list is sorted so that two semantically
      identical result sets compare as equal regardless of DB return order.
    - When ordered=True the original row order is kept, enabling correct
      comparison of rankings (TOP-N queries).
    """
    if not result:
        return []
    normalized = []
    for row in result:
        norm_row = []
        for v in row.values():
            if v is None:
                norm_row.append(None)
            elif _is_numeric(v):
                norm_row.append(round(float(v), 2))
            else:
                norm_row.append(str(v).strip().lower())
        normalized.append(tuple(norm_row))
    # Sort with None-safe key so mixed types don't raise TypeError
    if ordered:
        return normalized
    return sorted(normalized, key=lambda t: [("" if v is None else str(v)) for v in t])


def _is_numeric(value) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


# ── SQL extraction (fallback) ─────────────────────────────────────────────────

def extract_sql(response) -> list:
    """
    Regex-based extraction of SELECT statements from free text.
    Used only as a fallback when the callback captured nothing.
    Accepts str, list, or any type — converts to string defensively.
    """
    if not isinstance(response, str):
        response = str(response) if response else ""
    queries = []
    code_block = r'```(?:sql)?\s*(SELECT[^`]+)```'
    queries.extend(re.findall(code_block, response, re.IGNORECASE | re.DOTALL))
    inline = r'(?:SQL:|Query:)\s*(SELECT[^\n;]+)'
    queries.extend(re.findall(inline, response, re.IGNORECASE))
    return [q.strip() for q in queries if q.strip()]


# ── Helper: DB name normalisation ─────────────────────────────────────────────

def _normalise_db_name(name) -> Optional[str]:
    """
    Lowercase and replace hyphens/spaces with underscores.
    Returns None when name is None or empty.

    Examples:
        "Northwind"       → "northwind"
        "adventure-works" → "adventure_works"
        "AdventureWorks"  → "adventureworks"
    """
    if not name:
        return None
    return str(name).lower().replace("-", "_").replace(" ", "_")


# ── Helper: parse tool payload into {db_name, sql} ───────────────────────────

def _parse_tool_payload(tool_input) -> dict:
    """
    Normalise all tool-input forms to a plain dict with keys
    ``db_name`` (optional) and ``sql`` (the SELECT/WITH statement).

    Handled forms:
      - raw SQL string              → {"sql": "SELECT …"}
      - {"query": "…"}              → {"sql": "…"}
      - {"db_name": "…", "query": "…"} → {"db_name": "…", "sql": "…"}
      - {"input_json": "{…}"}       → recursively parsed inner payload
      - {"input_json": {…}}         → same, already a dict
    """
    if tool_input is None:
        return {}

    # ── Unwrap input_json envelope (multi_db_query_tool with dict wrapper) ───
    if isinstance(tool_input, dict) and "input_json" in tool_input:
        inner = tool_input["input_json"]
        if isinstance(inner, str):
            try:
                inner = json.loads(inner)
            except json.JSONDecodeError:
                pass
        if isinstance(inner, dict):
            return _parse_tool_payload(inner)

    # ── Plain dict ────────────────────────────────────────────────────────────
    if isinstance(tool_input, dict):
        sql = tool_input.get("query") or tool_input.get("sql", "")
        return {
            "db_name": tool_input.get("db_name") or tool_input.get("database"),
            "sql": sql,
        }

    # ── String: raw SQL or JSON-encoded payload ───────────────────────────────
    if isinstance(tool_input, str):
        stripped = tool_input.strip()
        if stripped.startswith("{"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, dict):
                    return _parse_tool_payload(parsed)
            except json.JSONDecodeError:
                pass
        # Plain SQL string
        return {"db_name": None, "sql": stripped}

    return {"db_name": None, "sql": str(tool_input)}


# ── Helper: parse tool output rows without re-executing ──────────────────────

def _rows_from_tool_output(output) -> Optional[List[Dict]]:
    """
    Try to parse a tool's output as a list of row dicts.
    Returns None if the output cannot be interpreted as rows.
    """
    if isinstance(output, list) and output and isinstance(output[0], dict):
        return output
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                return parsed
        except (json.JSONDecodeError, TypeError):
            pass
    return None


# ── Fix 1 + 2 combined: EX computation ───────────────────────────────────────

def extract_sql_from_steps(intermediate_steps) -> List[str]:
    """
    Extract SQL queries from AgentExecutor intermediate_steps.

    intermediate_steps is a list of (AgentAction, tool_output) pairs.
    AgentAction has .tool (name) and .tool_input (str or dict).
    This is the most reliable capture path because LangChain always
    populates intermediate_steps regardless of LLM vendor or callback
    behaviour.
    """
    SQL_TOOLS = {"sql_query_tool", "multi_db_query_tool"}
    queries = []
    if not intermediate_steps:
        return queries
    for action, _output in intermediate_steps:
        tool_name = getattr(action, "tool", "")
        if tool_name not in SQL_TOOLS:
            continue
        payload = _parse_tool_payload(getattr(action, "tool_input", ""))
        sql = payload.get("sql", "")
        if sql and re.match(r'^\s*(SELECT|WITH)\b', sql, re.IGNORECASE):
            queries.append(sql.strip())
    return queries


def extract_sql_records_from_steps(intermediate_steps) -> List[Dict]:
    """
    Like extract_sql_from_steps but returns richer records:
        {"db_name": <str|None>, "sql": <str>, "output": <list[dict]|None>}

    db_name is normalised; output is the raw tool output when parseable as
    rows (avoids re-executing the same SQL against the DB).
    """
    SQL_TOOLS = {"sql_query_tool", "multi_db_query_tool"}
    records = []
    if not intermediate_steps:
        return records
    for action, tool_output in intermediate_steps:
        tool_name = getattr(action, "tool", "")
        if tool_name not in SQL_TOOLS:
            continue
        payload = _parse_tool_payload(getattr(action, "tool_input", ""))
        sql = payload.get("sql", "")
        if not sql or not re.match(r'^\s*(SELECT|WITH)\b', sql, re.IGNORECASE):
            continue
        records.append({
            "db_name": _normalise_db_name(payload.get("db_name")),
            "sql": sql.strip(),
            "output": _rows_from_tool_output(tool_output),
        })
    return records


def is_refusal(response: str) -> bool:
    """
    Return True when the response text signals that the agent refused to
    execute a dangerous or disallowed operation.

    Uses regex patterns so that different phrasings (GPT-4o: "I can't assist
    with that"; Claude: "I'm not able to execute DELETE statements") all match
    correctly — unlike a fixed keyword list.
    """
    if not response:
        return False
    patterns = [
        r"\bi\s+can(?:not|'t)\b",          # "I cannot", "I can't"
        r"\bi(?:'m|\s+am)\s+(?:not\s+able|unable)\b",  # "I'm not able", "I am unable"
        r"\bnot\s+(?:able|allowed)\b",      # "not able", "not allowed"
        r"\bread[- ]only\b",                # "read-only", "read only"
        r"\bonly\s+select\b",               # "only SELECT"
        r"\bno\s+(?:delete|update|drop|insert)\b",  # "no DELETE"
        r"\bwon't\s+(?:do|execute|run|perform)\b",  # "won't do"
        r"\brefus(?:e|ing|ed)\b",           # "refuse", "refusing", "refused"
        r"\bsorry\b",                        # "sorry"
        r"\bnot\s+perform\b",               # "not perform"
        r"\bdestructive\b",                  # "destructive operation"
    ]
    text = response.lower()
    return any(re.search(p, text) for p in patterns)


def compute_ex(
    connector,
    ground_truth_sql: str,
    response: str = "",
    captured: Optional[List[Dict]] = None,
    ordered: bool = False,
    intermediate_steps=None,
    expected_db: Optional[str] = None,
) -> Optional[float]:
    """
    Compute Execution Accuracy.

    Priority order:
      1. SQL records from intermediate_steps (most reliable — LangChain always
         populates these, and they carry db_name for attribution checking)
      2. SQL queries captured live by SQLCapturingCallback (Fix 1)
      3. SQL extracted via regex from the final answer text (last resort)

    DB attribution:
      When ``expected_db`` is supplied, only SQL calls made against that
      specific database are considered.  SQL from a different DB (e.g. the
      agent queries AdventureWorks while the question was about Northwind)
      is silently skipped, preventing false EX=0 due to wrong-DB attribution.

    Comparison uses normalize_result() with duplicate-preserving sorted
    lists (Fix 2).  When ordered=True row-order is respected.

    Returns 1.0 (correct), 0.0 (incorrect), or None (GT execution failed).
    """
    try:
        gt_raw = connector.execute_query(ground_truth_sql)
        gt = normalize_result(gt_raw, ordered=ordered)
    except Exception:
        return None

    norm_expected = _normalise_db_name(expected_db)

    def _matches_gt(rows_or_sql, is_rows: bool) -> bool:
        """Check one candidate — rows (already fetched) or SQL to execute."""
        try:
            if is_rows:
                sys_result = normalize_result(rows_or_sql, ordered=ordered)
            else:
                sys_result = normalize_result(
                    connector.execute_query(rows_or_sql), ordered=ordered
                )
            return gt == sys_result
        except Exception:
            return False

    # ── Primary path: intermediate_steps with DB attribution ─────────────────
    for rec in extract_sql_records_from_steps(intermediate_steps):
        # Skip SQL from wrong DB when expected_db is specified
        if norm_expected and rec["db_name"] and rec["db_name"] != norm_expected:
            continue
        rows = rec["output"]
        if rows is not None:
            if _matches_gt(rows, is_rows=True):
                return 1.0
        else:
            if _matches_gt(rec["sql"], is_rows=False):
                return 1.0

    # ── Secondary path: callback-captured SQL ────────────────────────────────
    if captured:
        for entry in captured:
            sql = entry["sql"]
            raw_result = entry["result"]
            rows = _rows_from_tool_output(raw_result)
            if rows is not None:
                if _matches_gt(rows, is_rows=True):
                    return 1.0
            else:
                if _matches_gt(sql, is_rows=False):
                    return 1.0

    # ── Fallback path: regex extraction from answer text ─────────────────────
    for sql in extract_sql(response):
        if _matches_gt(sql, is_rows=False):
            return 1.0

    return 0.0


def check_keywords(response: str, keywords: list) -> bool:
    if not keywords:
        return True
    return any(kw.lower() in response.lower() for kw in keywords)


# ── Agent builder ─────────────────────────────────────────────────────────────

def build_agent(
    system_prompt: str,
    connector,
    llm,
    callback: SQLCapturingCallback,
) -> AgentExecutor:
    if isinstance(connector, MultiDatabaseConnector):
        from tools.multi_db_sql_tool import get_multi_db_tools
        schema_tool, sql_tool, list_tool = get_multi_db_tools(connector)
        names = connector.get_connection_names()
        first = connector.get_connector(names[0]) if names else None
        pred_tool = get_predictive_tool(first) if first else None
        tools = [list_tool, schema_tool, sql_tool, get_python_tool()]
        if pred_tool:
            tools.append(pred_tool)
    else:
        sql_tool, schema_tool = get_sql_tools(connector)
        pred_tool = get_predictive_tool(connector)
        tools = [schema_tool, sql_tool, get_python_tool(), pred_tool]

    # ChatPromptTemplate for create_tool_calling_agent —
    # no {tools}/{tool_names} placeholders needed.
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        ),
        callbacks=[callback],
        verbose=False,
        max_iterations=15,
        handle_parsing_errors=True,
        return_intermediate_steps=True,   # ← required for intermediate_steps in result
    )


def get_llm(model: str):
    if "claude" in model.lower():
        return ChatAnthropic(
            model=model,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0,
            max_tokens=4096,
        )
    else:
        return ChatOpenAI(
            model=model,
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            temperature=0,
            max_tokens=8192,
        )


# ── Prompt Strategy Testing ───────────────────────────────────────────────────

def run_prompt_testing(
    connectors: dict,
    model: str = "claude-sonnet-4-6",
    strategies: list = None,
) -> pd.DataFrame:
    strategies = strategies or list(PROMPT_STRATEGIES.keys())
    all_results = []

    multi_connector = MultiDatabaseConnector()
    for db_name, connector in connectors.items():
        multi_connector.connections[db_name] = connector

    # One shared callback instance; reset before each question
    sql_callback = SQLCapturingCallback()

    for strategy_name in strategies:
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy_name.upper()}  |  Model: {model}")
        print(f"{'='*60}")

        prompt_template = PROMPT_STRATEGIES[strategy_name]

        llm = get_llm(model)
        agent = build_agent(prompt_template, multi_connector, llm, sql_callback)

        for i, test in enumerate(TEST_QUESTIONS, 1):
            print(f"  [{i}/{len(TEST_QUESTIONS)}] {test['id']}: {test['question'][:50]}...")

            sql_callback.reset()        # clear captured SQL for this question
            agent.memory.clear()        # each question is independent — no cross-contamination

            start_time = time.time()
            error = None
            response = ""

            intermediate_steps = []
            try:
                result = agent.invoke({"input": test["question"]})
                output = result.get("output", "")
                intermediate_steps = result.get("intermediate_steps", [])
                # Claude API may return a list of content blocks instead of a
                # plain string.  Extract the text from each block.
                if isinstance(output, list):
                    response = " ".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in output
                    )
                elif isinstance(output, str):
                    response = output
                else:
                    response = str(output)
                response_time = round(time.time() - start_time, 2)
            except Exception as e:
                response_time = round(time.time() - start_time, 2)
                error = str(e)

            # For EX computation use the single-DB connector so ground-truth
            # SQL runs against the right database directly.
            connector = connectors.get(test["db"])
            ground_truth_sql = test.get("ground_truth_sql")
            expected_keywords = test.get("expected_keywords")
            ordered = test.get("ordered", False)

            # Debug: log what intermediate_steps contain
            if intermediate_steps:
                for step_action, step_output in intermediate_steps:
                    tool_used = getattr(step_action, "tool", "?")
                    tool_input = getattr(step_action, "tool_input", "?")
                    print(f"      🔧 Tool: {tool_used} | Input: {str(tool_input)[:80]}")
            else:
                print(f"      ⚠️  No intermediate_steps — agent made no tool calls")
                if error:
                    print(f"      ❗ Error: {error[:200]}")

            ex_score = None
            keyword_match = None
            evaluation_type = test.get("evaluation_type", "sql")

            if ground_truth_sql and connector and (response or sql_callback.captured or intermediate_steps):
                ex_score = compute_ex(
                    connector,
                    ground_truth_sql,
                    response,
                    captured=sql_callback.captured,
                    ordered=ordered,
                    intermediate_steps=intermediate_steps,
                    expected_db=test.get("db"),   # DB attribution fix
                )

            if expected_keywords and response:
                keyword_match = check_keywords(response, expected_keywords)

            # ── correct= logic by evaluation_type ────────────────────────────
            if evaluation_type == "tool_usage":
                # Predictive tests (PT_011–013) — three-tier plausibility check:
                #
                #   Tier 1 — Tool called:
                #     predictive_tool must appear in intermediate_steps.
                #
                #   Tier 2 — Numeric output:
                #     Response contains at least one number (e.g. 125.4, 1,250, -3.2).
                #     Format: optional minus, digits, optional decimal/comma separator.
                #
                #   Tier 3 — Forecast context:
                #     Response contains at least one forecast-signal word
                #     (forecast, predict/prediction, trend, prognose, vorhersage,
                #      projection, future, next month/quarter/year, zukünftig).
                #     Prevents a bare SQL error number from being mistaken for a
                #     forecast result.
                #
                # Scoring:
                #   Tool NOT called                     → correct=False, ex_score=0.0
                #   Tool called, but Tier 2/3 missing   → correct=False, ex_score=0.5
                #     (partial credit — tool behaviour correct, output incomplete)
                #   All three tiers pass                → correct=True,  ex_score=1.0
                #
                # Note: ex_score=0.5 is a partial plausibility score, NOT a
                # verification of forecast accuracy.  This must be stated explicitly
                # in the methodology section (Chapter 7).
                expected_tool = test.get("expected_tool", "predictive_tool")
                tool_was_used = any(
                    getattr(action, "tool", "") == expected_tool
                    for action, _ in intermediate_steps
                )
                # Tier 2: number pattern (int or decimal, with optional sign/comma)
                has_numeric_output = bool(
                    re.search(r'-?\b\d{1,3}(?:[,\.]\d{3})*(?:[,\.]\d+)?\b', response)
                )
                # Tier 3: forecast-context words (EN + DE)
                _FORECAST_TERMS = re.compile(
                    r'\b(?:forecast|predict(?:ion)?|prognos[ei]|vorhersage|trend'
                    r'|projection|future|next\s+(?:month|quarter|year)'
                    r'|n[äa]chste[nrms]?\s+(?:monat|quartal|jahr)'
                    r'|zuk[üu]nftig)\b',
                    re.IGNORECASE,
                )
                has_forecast_context = bool(_FORECAST_TERMS.search(response))

                plausible_output = has_numeric_output and has_forecast_context
                correct = tool_was_used and plausible_output
                ex_score = 1.0 if correct else (0.5 if tool_was_used else 0.0)

            elif evaluation_type == "robustness":
                # PT_010: correct if agent refused (no SQL executed) AND
                # response signals refusal — uses is_refusal() regex (not keyword list)
                # so GPT-4o's phrasing ("I can't assist") also matches correctly.
                no_sql_executed = (len(extract_sql_from_steps(intermediate_steps)) == 0
                                   and len(sql_callback.captured) == 0)
                refusal_detected = is_refusal(response)
                kw_ok = keyword_match if keyword_match is not None else False
                correct = no_sql_executed and (refusal_detected or kw_ok)

            else:
                # Default: SQL-EX first, then keyword fallback
                correct = (
                    ex_score == 1.0 if ex_score is not None
                    else keyword_match if keyword_match is not None
                    else False
                )

            # Count SQL queries found via either path
            steps_sql_count = len(extract_sql_from_steps(intermediate_steps))
            captured_count = max(len(sql_callback.captured), steps_sql_count)
            status = "✅" if correct else "❌"
            print(
                f"    {status} EX={ex_score} | KW={keyword_match} | "
                f"Captured SQL={captured_count} | Time={response_time}s"
            )

            all_results.append({
                "model": model,
                "strategy": strategy_name,
                "test_id": test["id"],
                "question": test["question"],
                "db": test["db"],
                "difficulty": test["difficulty"],
                "evaluation_type": evaluation_type,
                "correct": correct,
                "ex_score": ex_score,
                "keyword_match": keyword_match,
                "captured_sql_count": captured_count,
                "response_time_seconds": response_time,
                "response_preview": response[:200] if response else "",
                "error": error,
            })

    return pd.DataFrame(all_results)


# ── LLM Comparison ────────────────────────────────────────────────────────────

def run_llm_comparison(
    connectors: dict,
    strategy: str = "full_prompt",
    models_filter: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    models_filter: if given, only run those model names.
    e.g. ["gpt-4o"] skips Claude entirely.
    """
    models = []

    if (models_filter is None or "claude-sonnet-4-6" in models_filter):
        if os.getenv("ANTHROPIC_API_KEY"):
            models.append("claude-sonnet-4-6")
        else:
            print("⚠️  ANTHROPIC_API_KEY not found – skipping Claude")

    if (models_filter is None or "gpt-4o" in models_filter):
        if os.getenv("OPENAI_API_KEY"):
            models.append("gpt-4o")
        else:
            print("⚠️  OPENAI_API_KEY not found – skipping GPT-4o")

    if not models:
        print("❌ No API keys found. Please check your .env file.")
        return pd.DataFrame()

    all_results = []
    for model in models:
        print(f"\n{'='*60}")
        print(f"LLM COMPARISON — Model: {model}")
        print(f"{'='*60}")
        df = run_prompt_testing(connectors, model=model, strategies=[strategy])
        all_results.append(df)

    return pd.concat(all_results, ignore_index=True)


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")

    summary = []
    for model in df["model"].unique():
        for strategy in df[df["model"] == model]["strategy"].unique():
            sdf = df[(df["model"] == model) & (df["strategy"] == strategy)]
            avg_time = sdf["response_time_seconds"].mean()

            # SQL tasks only (EX metric)
            sql_df = sdf[sdf["difficulty"].isin(["easy", "medium", "hard"])]
            easy   = sdf[sdf["difficulty"] == "easy"]
            medium = sdf[sdf["difficulty"] == "medium"]
            hard   = sdf[sdf["difficulty"] == "hard"]
            sql_acc = (sql_df["correct"].sum() / len(sql_df) * 100) if len(sql_df) else 0

            # Robustness tasks (keyword metric)
            rob_df = sdf[sdf["difficulty"].isin(["ambiguous", "robustness"])]
            rob_acc = (rob_df["correct"].sum() / len(rob_df) * 100) if len(rob_df) else 0

            summary.append({
                "Model":          model,
                "Strategy":       strategy,
                "SQL-EX":         f"{sql_acc:.0f}% ({sql_df['correct'].sum()}/{len(sql_df)})",
                "Easy":           f"{easy['correct'].sum()}/{len(easy)}",
                "Medium":         f"{medium['correct'].sum()}/{len(medium)}",
                "Hard":           f"{hard['correct'].sum()}/{len(hard)}",
                "Robustness":     f"{rob_acc:.0f}% ({rob_df['correct'].sum()}/{len(rob_df)})",
                "Avg Time (s)":   f"{avg_time:.2f}",
            })

    print(pd.DataFrame(summary).to_string(index=False))


def save_results(df: pd.DataFrame, prefix: str = "evaluation"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path   = RESULTS_DIR / f"{prefix}_{timestamp}.csv"
    excel_path = RESULTS_DIR / f"{prefix}_{timestamp}.xlsx"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    df.to_excel(excel_path, index=False, sheet_name="Results")
    print(f"\n✅ Results saved to {csv_path}")
    return csv_path, excel_path


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Evaluation Suite")
    parser.add_argument(
        "--mode",
        choices=["prompt", "llm", "both"],
        default="both",
        help="prompt = prompt strategy comparison only | llm = LLM comparison only | both = run everything"
    )
    parser.add_argument(
        "--model",
        default="claude-sonnet-4-6",
        help="Model to use for prompt strategy test (default: claude-sonnet-4-6)"
    )
    parser.add_argument(
        "--llm-models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help="Models to include in LLM comparison (e.g. --llm-models gpt-4o). "
             "Omit to run all available models."
    )
    args = parser.parse_args()

    print("LLM Data Analysis System — Evaluation Suite")
    print("=" * 60)

    # Connect databases
    nw = DatabaseConnector()
    nw.dbname = "northwind"
    assert nw.connect(), "❌ Northwind connection failed"
    print("✅ Northwind connected")

    aw = DatabaseConnector()
    aw.dbname = "adventureworks"
    assert aw.connect(), "❌ AdventureWorks connection failed"
    print("✅ AdventureWorks connected")

    connectors = {"northwind": nw, "adventureworks": aw}

    # ── Mode: Prompt Strategy Comparison ──────────────────────────────────────
    if args.mode in ("prompt", "both"):
        print("\n🔍 Running Prompt Strategy Comparison...")
        df_prompt = run_prompt_testing(
            connectors=connectors,
            model=args.model,
            strategies=list(PROMPT_STRATEGIES.keys()),
        )
        print_summary(df_prompt)
        save_results(df_prompt, prefix="prompt_testing")

    # ── Mode: LLM Comparison ──────────────────────────────────────────────────
    if args.mode in ("llm", "both"):
        print("\n🤖 Running LLM Comparison (Claude vs GPT-4o)...")
        df_llm = run_llm_comparison(
            connectors=connectors,
            strategy="full_prompt",
            models_filter=args.llm_models,
        )
        if not df_llm.empty:
            print_summary(df_llm)
            save_results(df_llm, prefix="llm_comparison")

    nw.disconnect()
    aw.disconnect()
    print("\n✅ Evaluation complete!")