"""
query_history.py
----------------
Tracks all user questions, generated SQL queries, agent responses,
response times, hallucination flags, and star ratings (1-5).
Supports export to CSV and Excel for evaluation in Chapter 7.
"""

import json
import csv
from datetime import datetime
from pathlib import Path

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# History file location
HISTORY_DIR = Path(__file__).parent.parent / "logs"
HISTORY_DIR.mkdir(exist_ok=True)
HISTORY_FILE = HISTORY_DIR / "query_history.jsonl"


class QueryHistory:
    """
    Records all interactions between the user and the agent.
    Each entry contains the question, SQL queries used, response,
    response time, hallucination flag, and star rating (1-5).
    """

    def __init__(self):
        self.history_file = HISTORY_FILE
        self._session_entries = []

    def log(
        self,
        question: str,
        response: str,
        sql_queries: list[str] = None,
        tools_used: list[str] = None,
        response_time: float = None,
        db_names: list[str] = None,
        hallucination_detected: bool = None,
        user_rating: int = None,  # 1-5 stars
    ) -> int:
        """
        Logs a single interaction to the history file.
        Returns the index of the entry in the session list.
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "response": response[:500] if response else "",
            "sql_queries": sql_queries or [],
            "tools_used": tools_used or [],
            "response_time_seconds": response_time,
            "databases": db_names or [],
            "hallucination_detected": hallucination_detected,
            "user_rating": user_rating,  # None until user rates
        }

        # Save to file
        with open(self.history_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        # Keep in session memory
        self._session_entries.append(entry)
        return len(self._session_entries) - 1

    def update_rating(self, msg_id: int, rating: int):
        """
        Updates the star rating (1-5) for a specific message.
        Called when user submits a rating via st.feedback().
        """
        if 0 <= msg_id < len(self._session_entries):
            self._session_entries[msg_id]["user_rating"] = rating

    def update_feedback(self, msg_id: int, feedback: str):
        """
        Alias for update_rating() for backward compatibility.
        Accepts string "1"-"5" or integer.
        """
        try:
            rating = int(feedback)
            self.update_rating(msg_id, rating)
        except (ValueError, TypeError):
            pass

    def load_all(self) -> list[dict]:
        """Loads all history entries from the log file."""
        if not self.history_file.exists():
            return []
        entries = []
        with open(self.history_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def get_session_history(self) -> list[dict]:
        """Returns only the entries from the current session."""
        return self._session_entries

    def export_to_csv(self, output_path: str = None) -> str:
        """Exports query history to CSV."""
        entries = self._session_entries or self.load_all()
        if not entries:
            return None

        output_path = output_path or str(HISTORY_DIR / "query_history.csv")

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "question", "sql_queries", "tools_used",
                "response_time_seconds", "databases",
                "hallucination_detected", "user_rating",
            ])
            writer.writeheader()
            for entry in entries:
                writer.writerow({
                    "timestamp": entry.get("timestamp"),
                    "question": entry.get("question"),
                    "sql_queries": "; ".join(entry.get("sql_queries", [])),
                    "tools_used": ", ".join(entry.get("tools_used", [])),
                    "response_time_seconds": entry.get("response_time_seconds"),
                    "databases": ", ".join(entry.get("databases", [])),
                    "hallucination_detected": entry.get("hallucination_detected"),
                    "user_rating": entry.get("user_rating"),
                })

        return output_path

    def export_to_excel(self, output_path: str = None) -> str:
        """Exports query history to Excel (.xlsx)."""
        if not PANDAS_AVAILABLE:
            return None

        entries = self._session_entries or self.load_all()
        if not entries:
            return None

        output_path = output_path or str(HISTORY_DIR / "query_history.xlsx")

        df = pd.DataFrame([{
            "Timestamp": e.get("timestamp"),
            "Question": e.get("question"),
            "SQL Queries": "; ".join(e.get("sql_queries", [])),
            "Tools Used": ", ".join(e.get("tools_used", [])),
            "Response Time (s)": e.get("response_time_seconds"),
            "Databases": ", ".join(e.get("databases", [])),
            "Hallucination Detected": e.get("hallucination_detected"),
            "User Rating (1-5)": e.get("user_rating"),
        } for e in entries])

        df.to_excel(output_path, index=False, sheet_name="Query History")
        return output_path

    def get_statistics(self) -> dict:
        """Returns statistics about the query history including ratings."""
        entries = self._session_entries or self.load_all()
        if not entries:
            return {"error": "No history found."}

        total = len(entries)
        hallucinations = sum(
            1 for e in entries if e.get("hallucination_detected") is True
        )
        avg_time = sum(
            e.get("response_time_seconds", 0) or 0 for e in entries
        ) / total

        # Rating statistics
        rated = [e for e in entries if e.get("user_rating") is not None]
        avg_rating = (
            sum(e["user_rating"] for e in rated) / len(rated)
            if rated else None
        )
        rating_distribution = {str(i): 0 for i in range(1, 6)}
        for e in rated:
            r = str(e.get("user_rating", ""))
            if r in rating_distribution:
                rating_distribution[r] += 1

        return {
            "total_queries": total,
            "hallucinations_detected": hallucinations,
            "hallucination_rate_pct": round(hallucinations / total * 100, 2),
            "avg_response_time_seconds": round(avg_time, 2),
            "total_rated": len(rated),
            "avg_rating": round(avg_rating, 2) if avg_rating else None,
            "rating_distribution": rating_distribution,
        }

    def clear_session(self):
        """Clears the current session entries (not the file)."""
        self._session_entries = []


# Singleton instance
query_history = QueryHistory()