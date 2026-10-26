"""
file_uploader.py
----------------
Handles CSV and Excel file uploads.
Automatically imports uploaded data into PostgreSQL
so the agent can analyze it immediately.
"""

import pandas as pd
from pathlib import Path
from sqlalchemy import text
from database.db_connector import DatabaseConnector


# Supported file types
SUPPORTED_EXTENSIONS = [".csv", ".xlsx", ".xls"]


def read_file(file) -> pd.DataFrame:
    """
    Reads a CSV or Excel file and returns a pandas DataFrame.
    Accepts a file-like object (e.g. from Streamlit file_uploader).
    """
    filename = file.name.lower()

    if filename.endswith(".csv"):
        # Try common separators
        try:
            df = pd.read_csv(file, sep=",")
            if df.shape[1] == 1:
                file.seek(0)
                df = pd.read_csv(file, sep=";")
        except Exception:
            file.seek(0)
            df = pd.read_csv(file, sep=";")

    elif filename.endswith((".xlsx", ".xls")):
        df = pd.read_excel(file)

    else:
        raise ValueError(
            f"Unsupported file type. Supported: {SUPPORTED_EXTENSIONS}"
        )

    return df


def clean_table_name(filename: str) -> str:
    """
    Converts a filename to a valid PostgreSQL table name.
    e.g. 'Sales Data 2024.csv' → 'sales_data_2024'
    """
    name = Path(filename).stem
    name = name.lower()
    name = "".join(c if c.isalnum() else "_" for c in name)
    name = name.strip("_")

    # Ensure it doesn't start with a number
    if name[0].isdigit():
        name = "upload_" + name

    return name


def infer_sql_type(dtype) -> str:
    """Maps pandas dtypes to PostgreSQL column types."""
    dtype_str = str(dtype)

    if "int" in dtype_str:
        return "BIGINT"
    elif "float" in dtype_str:
        return "DOUBLE PRECISION"
    elif "bool" in dtype_str:
        return "BOOLEAN"
    elif "datetime" in dtype_str:
        return "TIMESTAMP"
    else:
        return "TEXT"


def import_to_postgres(
    df: pd.DataFrame,
    table_name: str,
    connector: DatabaseConnector,
    if_exists: str = "replace",
) -> dict:
    """
    Imports a DataFrame into PostgreSQL as a new table.

    Args:
        df: DataFrame to import
        table_name: Name for the new table
        connector: Active DatabaseConnector
        if_exists: 'replace' to overwrite, 'append' to add rows

    Returns:
        dict with import results (rows, columns, table_name)
    """
    if not connector.engine:
        raise ConnectionError("No active database connection.")

    # Clean column names
    df.columns = [
        "".join(c if c.isalnum() else "_" for c in col).strip("_").lower()
        for col in df.columns
    ]

    # Remove duplicate column names
    seen = {}
    new_cols = []
    for col in df.columns:
        if col in seen:
            seen[col] += 1
            new_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            new_cols.append(col)
    df.columns = new_cols

    # Import to PostgreSQL using pandas to_sql
    df.to_sql(
        name=table_name,
        con=connector.engine,
        if_exists=if_exists,
        index=False,
        method="multi",
        chunksize=1000,
    )

    return {
        "table_name": table_name,
        "rows_imported": len(df),
        "columns": list(df.columns),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
    }


def get_uploaded_tables(connector: DatabaseConnector) -> list[str]:
    """
    Returns a list of tables that were created via file upload.
    These are stored in a metadata table '_uploaded_tables'.
    """
    try:
        results = connector.execute_query(
            "SELECT table_name FROM _uploaded_tables ORDER BY uploaded_at DESC"
        )
        return [r["table_name"] for r in results]
    except Exception:
        return []


def register_uploaded_table(
    connector: DatabaseConnector,
    table_name: str,
    filename: str,
    rows: int,
):
    """
    Registers an uploaded table in a metadata table for tracking.
    Creates the metadata table if it doesn't exist.
    """
    try:
        with connector.engine.connect() as conn:
            # Create metadata table if needed
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS _uploaded_tables (
                    id SERIAL PRIMARY KEY,
                    table_name TEXT NOT NULL,
                    original_filename TEXT,
                    rows_imported INTEGER,
                    uploaded_at TIMESTAMP DEFAULT NOW()
                )
            """))

            conn.execute(text("""
                INSERT INTO _uploaded_tables
                    (table_name, original_filename, rows_imported)
                VALUES (:table_name, :filename, :rows)
            """), {"table_name": table_name, "filename": filename, "rows": rows})

            conn.commit()
    except Exception:
        pass  # Non-critical - metadata tracking is optional


def process_upload(
    file,
    connector: DatabaseConnector,
    custom_table_name: str = None,
) -> dict:
    """
    Full pipeline: read file → clean → import to PostgreSQL → register.

    Args:
        file: Uploaded file object from Streamlit
        connector: Active DatabaseConnector
        custom_table_name: Optional custom table name

    Returns:
        Import result dict
    """
    # Read file
    df = read_file(file)

    # Determine table name
    table_name = custom_table_name or clean_table_name(file.name)

    # Import to PostgreSQL
    result = import_to_postgres(df, table_name, connector)

    # Register in metadata
    register_uploaded_table(
        connector,
        table_name,
        file.name,
        result["rows_imported"],
    )

    return result