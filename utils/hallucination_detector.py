"""
hallucination_detector.py
-------------------------
Detects potential hallucinations in agent responses by comparing
numeric claims against actual database values.
"""

import re
import json
from database.db_connector import DatabaseConnector
from database.multi_db_connector import MultiDatabaseConnector


# Threshold for flagging potential hallucinations (5% deviation)
HALLUCINATION_THRESHOLD = 0.05


def extract_numbers_from_text(text: str) -> list[float]:
    """
    Extracts all numeric values from a text string.
    Handles formats like: 91, $1,525.05, 19,820, 3.14
    """
    # Remove currency symbols and extract numbers
    cleaned = text.replace("$", "").replace(",", "")
    pattern = r'\b\d+\.?\d*\b'
    matches = re.findall(pattern, cleaned)
    return [float(m) for m in matches if float(m) > 0]


def extract_sql_from_response(response: str) -> list[str]:
    """
    Extracts SQL queries from agent response.
    Looks for queries in code blocks or after 'SQL:' markers.
    """
    sql_queries = []

    # Look for SQL in code blocks
    code_block_pattern = r'```(?:sql)?\s*(SELECT[^`]+)```'
    matches = re.findall(code_block_pattern, response, re.IGNORECASE | re.DOTALL)
    sql_queries.extend(matches)

    # Look for inline SQL after markers
    inline_pattern = r'(?:SQL:|Query:)\s*(SELECT[^\n]+)'
    matches = re.findall(inline_pattern, response, re.IGNORECASE)
    sql_queries.extend(matches)

    return [q.strip() for q in sql_queries]


def verify_numeric_claim(
    claimed_value: float,
    actual_value: float,
    threshold: float = HALLUCINATION_THRESHOLD,
) -> dict:
    """
    Compares a claimed value against an actual database value.
    Returns a dict with verification result.
    """
    if actual_value == 0:
        deviation = 1.0 if claimed_value != 0 else 0.0
    else:
        deviation = abs(claimed_value - actual_value) / abs(actual_value)

    return {
        "claimed": claimed_value,
        "actual": actual_value,
        "deviation_pct": round(deviation * 100, 2),
        "is_hallucination": deviation > threshold,
    }


class HallucinationDetector:
    """
    Detects potential hallucinations in agent responses by:
    1. Extracting SQL queries from the response
    2. Re-executing them to get ground truth values
    3. Comparing claimed numbers against actual results
    """

    def __init__(self, connector: DatabaseConnector | MultiDatabaseConnector):
        self.connector = connector
        self.detections = []

    def _get_connector_for_db(self, db_name: str = None) -> DatabaseConnector:
        """Returns the appropriate connector based on type."""
        if isinstance(self.connector, MultiDatabaseConnector):
            if db_name:
                return self.connector.get_connector(db_name)
            # Use first available connector
            names = self.connector.get_connection_names()
            if names:
                return self.connector.get_connector(names[0])
            return None
        return self.connector

    def verify_response(
        self,
        question: str,
        response: str,
        db_name: str = None,
    ) -> dict:
        """
        Verifies an agent response for potential hallucinations.

        Returns a verification report with:
        - verified: True if no hallucinations detected
        - warnings: list of detected hallucinations
        - sql_verified: number of SQL queries re-executed
        """
        report = {
            "question": question,
            "verified": True,
            "warnings": [],
            "sql_verified": 0,
            "details": [],
        }

        # Extract SQL queries from response
        sql_queries = extract_sql_from_response(response)

        if not sql_queries:
            report["verified"] = None  # Cannot verify without SQL
            report["note"] = "No SQL queries found in response to verify."
            return report

        connector = self._get_connector_for_db(db_name)
        if not connector:
            report["verified"] = None
            report["note"] = "No database connector available."
            return report

        # Re-execute each SQL query and compare results
        for sql in sql_queries:
            try:
                actual_results = connector.execute_query(sql)
                report["sql_verified"] += 1

                if not actual_results:
                    continue

                # Extract numeric values from actual results
                actual_numbers = []
                for row in actual_results:
                    for val in row.values():
                        try:
                            actual_numbers.append(float(val))
                        except (TypeError, ValueError):
                            pass

                # Extract numbers claimed in response
                claimed_numbers = extract_numbers_from_text(response)

                # Compare each actual number against claimed numbers
                for actual_val in actual_numbers:
                    for claimed_val in claimed_numbers:
                        result = verify_numeric_claim(claimed_val, actual_val)
                        if result["is_hallucination"]:
                            warning = (
                                f"Claimed value {claimed_val} differs from "
                                f"actual DB value {actual_val} "
                                f"by {result['deviation_pct']}%"
                            )
                            report["warnings"].append(warning)
                            report["verified"] = False
                            report["details"].append(result)

            except Exception as e:
                report["details"].append({
                    "sql": sql,
                    "error": str(e),
                })

        self.detections.append(report)
        return report

    def get_hallucination_rate(self) -> dict:
        """
        Calculates the hallucination rate across all verifications.
        Used for Kapitel 7 evaluation metrics.
        """
        if not self.detections:
            return {"error": "No verifications performed yet."}

        total = len(self.detections)
        verified = sum(1 for d in self.detections if d.get("verified") is True)
        hallucinated = sum(1 for d in self.detections if d.get("verified") is False)
        unverifiable = sum(1 for d in self.detections if d.get("verified") is None)

        return {
            "total_responses": total,
            "verified_correct": verified,
            "hallucinations_detected": hallucinated,
            "unverifiable": unverifiable,
            "hallucination_rate_pct": round(hallucinated / total * 100, 2),
        }

    def export_detections(self) -> str:
        """Exports all detections as JSON string."""
        return json.dumps(self.detections, ensure_ascii=False, indent=2)