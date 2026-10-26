"""
evaluation.py
-------------
Systematic evaluation of the LLM-based data analysis system.

Metrics:
- Execution Accuracy (EX): ground truth SQL vs system SQL results match
- Hallucination Rate: numeric claims vs actual DB values
- Response Time: seconds per query
- Robustness: system behavior on ambiguous/invalid queries
- LLM Comparison: Claude vs GPT-4
"""

import sys
import os
import time
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / '.env')

from database.db_connector import DatabaseConnector
from database.multi_db_connector import MultiDatabaseConnector
from agent.analyst_agent import DataAnalystAgent
from utils.hallucination_detector import HallucinationDetector
from utils.logger import EvaluationLogger

# ── Output directory ──────────────────────────────────────────────────────────
RESULTS_DIR = Path(__file__).parent.parent / "evaluation_results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Execution Accuracy helpers ────────────────────────────────────────────────

def normalize_result(result: list[dict]) -> set:
    """
    Normalizes query results for comparison.
    Converts to a frozenset of tuples for order-independent comparison.
    """
    if not result:
        return frozenset()
    normalized = []
    for row in result:
        normalized.append(
            tuple(
                round(float(v), 2) if _is_numeric(v) else str(v).strip().lower()
                for v in row.values()
            )
        )
    return frozenset(normalized)


def _is_numeric(value) -> bool:
    """Checks if a value can be converted to float."""
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def extract_sql_from_response(response: str) -> list[str]:
    """Extracts SQL SELECT statements from agent response."""
    import re
    sql_queries = []

    # Code blocks
    code_block = r'```(?:sql)?\s*(SELECT[^`]+)```'
    matches = re.findall(code_block, response, re.IGNORECASE | re.DOTALL)
    sql_queries.extend(matches)

    # Inline after SQL: marker
    inline = r'(?:SQL:|Query:)\s*(SELECT[^\n;]+)'
    matches = re.findall(inline, response, re.IGNORECASE)
    sql_queries.extend(matches)

    return [q.strip() for q in sql_queries if q.strip()]


def compute_execution_accuracy(
    connector: DatabaseConnector,
    ground_truth_sql: str,
    system_response: str,
) -> dict:
    """
    Computes Execution Accuracy (EX) by comparing ground truth SQL results
    with the results of SQL extracted from the system response.

    Returns:
        dict with:
        - ex_score: 1.0 (match) or 0.0 (no match)
        - ground_truth_result: actual DB result
        - system_sql: SQL extracted from response
        - system_result: system's DB result
        - error: error message if execution failed
    """
    result = {
        "ex_score": 0.0,
        "ground_truth_result": None,
        "system_sql": None,
        "system_result": None,
        "error": None,
    }

    # Execute ground truth SQL
    try:
        gt_result = connector.execute_query(ground_truth_sql)
        result["ground_truth_result"] = gt_result
        gt_normalized = normalize_result(gt_result)
    except Exception as e:
        result["error"] = f"Ground truth SQL error: {e}"
        return result

    # Extract and execute system SQL
    system_sqls = extract_sql_from_response(system_response)
    if not system_sqls:
        result["error"] = "No SQL found in system response"
        return result

    # Try each extracted SQL
    for sql in system_sqls:
        try:
            sys_result = connector.execute_query(sql)
            result["system_sql"] = sql
            result["system_result"] = sys_result
            sys_normalized = normalize_result(sys_result)

            if gt_normalized == sys_normalized:
                result["ex_score"] = 1.0
                return result
        except Exception as e:
            result["error"] = str(e)
            continue

    return result


def check_robustness(response: str, expected_keywords: list[str]) -> bool:
    """
    For robustness tests (UC5, UC6) where no ground truth SQL exists.
    Checks if system responds appropriately to ambiguous/invalid queries.
    """
    if not expected_keywords:
        return True
    response_lower = response.lower()
    return any(kw.lower() in response_lower for kw in expected_keywords)


# ── Test cases ────────────────────────────────────────────────────────────────
TEST_CASES = [
    # ── UC1: Simple descriptive queries ──────────────────────────────────────
    {
        "id": "UC1_NW_001",
        "category": "UC1_simple_sql",
        "db": "northwind",
        "question": "How many customers are in the database?",
        "ground_truth_sql": "SELECT COUNT(*) AS count FROM customers",
        "expected_keywords": None,
        "difficulty": "easy",
    },
    {
        "id": "UC1_NW_002",
        "category": "UC1_simple_sql",
        "db": "northwind",
        "question": "How many products does the database contain?",
        "ground_truth_sql": "SELECT COUNT(*) AS count FROM products",
        "expected_keywords": None,
        "difficulty": "easy",
    },
    {
        "id": "UC1_NW_003",
        "category": "UC1_simple_sql",
        "db": "northwind",
        "question": "What is the average unit price of all products?",
        "ground_truth_sql": "SELECT ROUND(AVG(unit_price)::numeric, 2) AS avg_price FROM products",
        "expected_keywords": None,
        "difficulty": "easy",
    },
    {
        "id": "UC1_NW_004",
        "category": "UC1_simple_sql",
        "db": "northwind",
        "question": "How many orders were placed in total?",
        "ground_truth_sql": "SELECT COUNT(*) AS count FROM orders",
        "expected_keywords": None,
        "difficulty": "easy",
    },
    {
        "id": "UC1_NW_005",
        "category": "UC1_simple_sql",
        "db": "northwind",
        "question": "How many suppliers are in the database?",
        "ground_truth_sql": "SELECT COUNT(*) AS count FROM suppliers",
        "expected_keywords": None,
        "difficulty": "easy",
    },
    {
        "id": "UC1_AW_001",
        "category": "UC1_simple_sql",
        "db": "adventureworks",
        "question": "How many employees does AdventureWorks have?",
        "ground_truth_sql": "SELECT COUNT(*) AS count FROM humanresources.employee",
        "expected_keywords": None,
        "difficulty": "easy",
    },
    {
        "id": "UC1_AW_002",
        "category": "UC1_simple_sql",
        "db": "adventureworks",
        "question": "How many products are in the AdventureWorks database?",
        "ground_truth_sql": "SELECT COUNT(*) AS count FROM production.product",
        "expected_keywords": None,
        "difficulty": "easy",
    },
    {
        "id": "UC1_AW_003",
        "category": "UC1_simple_sql",
        "db": "adventureworks",
        "question": "How many customers does AdventureWorks have?",
        "ground_truth_sql": "SELECT COUNT(*) AS count FROM sales.customer",
        "expected_keywords": None,
        "difficulty": "easy",
    },
    {
        "id": "UC1_AW_004",
        "category": "UC1_simple_sql",
        "db": "adventureworks",
        "question": "How many sales orders are in the database?",
        "ground_truth_sql": "SELECT COUNT(*) AS count FROM sales.salesorderheader",
        "expected_keywords": None,
        "difficulty": "easy",
    },
    {
        "id": "UC1_AW_005",
        "category": "UC1_simple_sql",
        "db": "adventureworks",
        "question": "How many product categories exist?",
        "ground_truth_sql": "SELECT COUNT(*) AS count FROM production.productcategory",
        "expected_keywords": None,
        "difficulty": "easy",
    },

    # ── UC2: Complex queries ──────────────────────────────────────────────────
    {
        "id": "UC2_NW_001",
        "category": "UC2_complex_sql",
        "db": "northwind",
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
        "expected_keywords": None,
        "difficulty": "medium",
    },
    {
        "id": "UC2_NW_002",
        "category": "UC2_complex_sql",
        "db": "northwind",
        "question": "What is the total number of orders per country?",
        "ground_truth_sql": """
            SELECT ship_country, COUNT(*) AS total_orders
            FROM orders
            GROUP BY ship_country
            ORDER BY total_orders DESC
        """,
        "expected_keywords": None,
        "difficulty": "medium",
    },
    {
        "id": "UC2_NW_003",
        "category": "UC2_complex_sql",
        "db": "northwind",
        "question": "Which category has the most products?",
        "ground_truth_sql": """
            SELECT c.category_name, COUNT(*) AS product_count
            FROM products p
            JOIN categories c ON p.category_id = c.category_id
            GROUP BY c.category_name
            ORDER BY product_count DESC
            LIMIT 1
        """,
        "expected_keywords": None,
        "difficulty": "medium",
    },
    {
        "id": "UC2_NW_004",
        "category": "UC2_complex_sql",
        "db": "northwind",
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
        "expected_keywords": None,
        "difficulty": "hard",
    },
    {
        "id": "UC2_AW_001",
        "category": "UC2_complex_sql",
        "db": "adventureworks",
        "question": "What are the top 5 product subcategories by number of products?",
        "ground_truth_sql": """
            SELECT ps.name AS subcategory, COUNT(*) AS product_count
            FROM production.product p
            JOIN production.productsubcategory ps ON p.productsubcategoryid = ps.productsubcategoryid
            GROUP BY ps.name
            ORDER BY product_count DESC
            LIMIT 5
        """,
        "expected_keywords": None,
        "difficulty": "medium",
    },
    {
        "id": "UC2_AW_002",
        "category": "UC2_complex_sql",
        "db": "adventureworks",
        "question": "How many employees are in each department?",
        "ground_truth_sql": """
            SELECT d.name AS department, COUNT(*) AS employee_count
            FROM humanresources.employeedepartmenthistory edh
            JOIN humanresources.department d ON edh.departmentid = d.departmentid
            WHERE edh.enddate IS NULL
            GROUP BY d.name
            ORDER BY employee_count DESC
        """,
        "expected_keywords": None,
        "difficulty": "medium",
    },
    {
        "id": "UC2_AW_003",
        "category": "UC2_complex_sql",
        "db": "adventureworks",
        "question": "What is the average list price per product category?",
        "ground_truth_sql": """
            SELECT pc.name AS category,
                   ROUND(AVG(p.listprice)::numeric, 2) AS avg_price
            FROM production.product p
            JOIN production.productsubcategory ps ON p.productsubcategoryid = ps.productsubcategoryid
            JOIN production.productcategory pc ON ps.productcategoryid = pc.productcategoryid
            WHERE p.listprice > 0
            GROUP BY pc.name
            ORDER BY avg_price DESC
        """,
        "expected_keywords": None,
        "difficulty": "hard",
    },

    # ── UC3: Visualization (keyword-based, no ground truth SQL) ───────────────
    {
        "id": "UC3_NW_001",
        "category": "UC3_visualization",
        "db": "northwind",
        "question": "Show me the number of orders per country as a chart",
        "ground_truth_sql": None,
        "expected_keywords": ["chart", "country", "orders", "germany", "usa"],
        "difficulty": "medium",
    },
    {
        "id": "UC3_NW_002",
        "category": "UC3_visualization",
        "db": "northwind",
        "question": "Show the revenue distribution by product category",
        "ground_truth_sql": None,
        "expected_keywords": ["category", "revenue", "beverages", "seafood"],
        "difficulty": "medium",
    },
    {
        "id": "UC3_AW_001",
        "category": "UC3_visualization",
        "db": "adventureworks",
        "question": "Show the number of employees per department as a bar chart",
        "ground_truth_sql": None,
        "expected_keywords": ["department", "employee", "chart"],
        "difficulty": "medium",
    },

    # ── UC4: Predictive analysis (keyword-based) ──────────────────────────────
    {
        "id": "UC4_NW_001",
        "category": "UC4_predictive",
        "db": "northwind",
        "question": "Predict the freight costs for the next 6 months",
        "ground_truth_sql": None,
        "expected_keywords": ["forecast", "model", "months", "predict"],
        "difficulty": "hard",
    },
    {
        "id": "UC4_NW_002",
        "category": "UC4_predictive",
        "db": "northwind",
        "question": "What will be the monthly order trend for the next 3 months?",
        "ground_truth_sql": None,
        "expected_keywords": ["forecast", "trend", "months", "model"],
        "difficulty": "hard",
    },
    {
        "id": "UC4_AW_001",
        "category": "UC4_predictive",
        "db": "adventureworks",
        "question": "Predict total sales for the next 6 months based on historical data",
        "ground_truth_sql": None,
        "expected_keywords": ["forecast", "sales", "model", "months"],
        "difficulty": "hard",
    },

    # ── UC5: Ambiguous queries ────────────────────────────────────────────────
    {
        "id": "UC5_NW_001",
        "category": "UC5_ambiguous",
        "db": "northwind",
        "question": "Show me the best customers",
        "ground_truth_sql": None,
        "expected_keywords": ["revenue", "customers", "top", "rank"],
        "difficulty": "medium",
    },
    {
        "id": "UC5_NW_002",
        "category": "UC5_ambiguous",
        "db": "northwind",
        "question": "Show me the numbers",
        "ground_truth_sql": None,
        "expected_keywords": ["clarif", "specify", "unclear", "which", "more"],
        "difficulty": "easy",
    },
    {
        "id": "UC5_AW_001",
        "category": "UC5_ambiguous",
        "db": "adventureworks",
        "question": "Who are the best employees?",
        "ground_truth_sql": None,
        "expected_keywords": ["sales", "performance", "employee", "top"],
        "difficulty": "medium",
    },

    # ── UC6: Robustness / invalid queries ─────────────────────────────────────
    {
        "id": "UC6_NW_001",
        "category": "UC6_robustness",
        "db": "northwind",
        "question": "Show me data from the xyz table",
        "ground_truth_sql": None,
        "expected_keywords": ["not found", "does not exist", "available", "no table"],
        "difficulty": "easy",
    },
    {
        "id": "UC6_NW_002",
        "category": "UC6_robustness",
        "db": "northwind",
        "question": "Delete all customers from the database",
        "ground_truth_sql": None,
        "expected_keywords": ["cannot", "not allowed", "only select", "read-only"],
        "difficulty": "easy",
    },
    {
        "id": "UC6_NW_003",
        "category": "UC6_robustness",
        "db": "northwind",
        "question": "Vergleich die Zahlen",
        "ground_truth_sql": None,
        "expected_keywords": ["unklar", "specify", "clarif", "welche", "which"],
        "difficulty": "easy",
    },
    {
        "id": "UC6_AW_001",
        "category": "UC6_robustness",
        "db": "adventureworks",
        "question": "Show me the salary of employee with id 99999",
        "ground_truth_sql": None,
        "expected_keywords": ["not found", "no result", "does not exist", "0"],
        "difficulty": "medium",
    },
]


# ── Core evaluation functions ─────────────────────────────────────────────────

def run_single_test(
    agent: DataAnalystAgent,
    detector: HallucinationDetector,
    connectors: dict,
    test_case: dict,
    model: str,
) -> dict:
    """Runs a single test case and returns the result dict."""
    print(f"  [{test_case['id']}] {test_case['question'][:60]}...")

    start_time = time.time()
    error = None
    response = ""

    try:
        response = agent.run(test_case["question"])
        response_time = round(time.time() - start_time, 2)
    except Exception as e:
        response_time = round(time.time() - start_time, 2)
        error = str(e)

    connector = connectors.get(test_case["db"])
    ground_truth_sql = test_case.get("ground_truth_sql")
    expected_keywords = test_case.get("expected_keywords")

    # ── Execution Accuracy (for SQL-based test cases) ─────────────────────────
    ex_score = None
    system_sql = None
    ex_error = None

    if ground_truth_sql and connector and response:
        ex_result = compute_execution_accuracy(
            connector, ground_truth_sql, response
        )
        ex_score = ex_result["ex_score"]
        system_sql = ex_result["system_sql"]
        ex_error = ex_result["error"]

    # ── Keyword-based check (for UC3, UC4, UC5, UC6) ─────────────────────────
    keyword_match = None
    if expected_keywords and response:
        keyword_match = check_robustness(response, expected_keywords)

    # ── Overall correctness ───────────────────────────────────────────────────
    if ex_score is not None:
        correct = ex_score == 1.0
    elif keyword_match is not None:
        correct = keyword_match
    else:
        correct = False

    # ── Hallucination detection ───────────────────────────────────────────────
    hallucination_detected = False
    if response and connector:
        hall_report = detector.verify_response(
            question=test_case["question"],
            response=response,
            db_name=test_case["db"],
        )
        hallucination_detected = hall_report.get("verified") is False

    status = "✅" if correct else "❌"
    print(f"    {status} EX={ex_score} | KW={keyword_match} | "
          f"Halluc={hallucination_detected} | Time={response_time}s")

    return {
        "id": test_case["id"],
        "category": test_case["category"],
        "db": test_case["db"],
        "difficulty": test_case["difficulty"],
        "model": model,
        "question": test_case["question"],
        "response_preview": response[:300] if response else "",
        "correct": correct,
        "ex_score": ex_score,
        "keyword_match": keyword_match,
        "system_sql": system_sql,
        "hallucination_detected": hallucination_detected,
        "response_time_seconds": response_time,
        "error": error or ex_error,
    }


def run_evaluation(connectors: dict, models: list[str]) -> pd.DataFrame:
    """Runs the full evaluation across all test cases and models."""
    all_results = []
    eval_logger = EvaluationLogger()

    for model in models:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")

        # Set model
        os.environ["LLM_MODEL"] = model
        os.environ["LLM_PROVIDER"] = "anthropic" if "claude" in model.lower() else "openai"

        # Build multi-connector
        multi_connector = MultiDatabaseConnector()
        for db_name, connector in connectors.items():
            multi_connector.connections[db_name] = connector

        agent = DataAnalystAgent(multi_connector)
        detector = HallucinationDetector(multi_connector)

        for i, test_case in enumerate(TEST_CASES, 1):
            print(f"\n[{i}/{len(TEST_CASES)}] {test_case['category']}")
            result = run_single_test(
                agent, detector, connectors, test_case, model
            )
            all_results.append(result)

            eval_logger.log_result(
                question=test_case["question"],
                expected=str(test_case.get("expected_keywords") or
                             test_case.get("ground_truth_sql", "")[:100]),
                actual=result["response_preview"],
                correct=result["correct"],
                response_time=result["response_time_seconds"],
                model=model,
                error=result["error"],
            )

        agent.reset_memory()

    df = pd.DataFrame(all_results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"evaluation_{timestamp}.csv"
    excel_path = RESULTS_DIR / f"evaluation_{timestamp}.xlsx"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    df.to_excel(excel_path, index=False)
    print(f"\n✅ Results saved to {csv_path}")

    return df


def print_summary(df: pd.DataFrame):
    """Prints evaluation summary."""
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")

    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        total = len(mdf)
        correct = mdf["correct"].sum()
        hallucinations = mdf["hallucination_detected"].sum()
        avg_time = mdf["response_time_seconds"].mean()

        # Execution Accuracy (only SQL test cases)
        sql_df = mdf[mdf["ex_score"].notna()]
        ex_accuracy = sql_df["ex_score"].mean() if len(sql_df) > 0 else None

        print(f"\nModel: {model}")
        print(f"  Total tests:           {total}")
        print(f"  Overall correctness:   {correct}/{total} ({correct/total*100:.1f}%)")
        if ex_accuracy is not None:
            print(f"  Execution Accuracy:    {ex_accuracy*100:.1f}% (SQL tests only)")
        print(f"  Hallucination rate:    {hallucinations}/{total} ({hallucinations/total*100:.1f}%)")
        print(f"  Avg response time:     {avg_time:.2f}s")

        print(f"\n  By category:")
        for cat in sorted(mdf["category"].unique()):
            cat_df = mdf[mdf["category"] == cat]
            cat_correct = cat_df["correct"].sum()
            cat_total = len(cat_df)
            print(f"    {cat}: {cat_correct}/{cat_total} ({cat_correct/cat_total*100:.0f}%)")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("LLM Data Analysis System — Evaluation")
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

    models = ["claude-sonnet-4-6"]

    df = run_evaluation(connectors, models)
    print_summary(df)

    nw.disconnect()
    aw.disconnect()
    print("\n✅ Evaluation complete!")