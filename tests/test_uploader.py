import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_connector import DatabaseConnector
from utils.file_uploader import import_to_postgres, clean_table_name
import pandas as pd

db = DatabaseConnector()
db.connect()

# Test direkt mit DataFrame statt FakeFile
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Carol"],
    "age": [30, 25, 35],
    "city": ["Berlin", "Munich", "Hamburg"]
})

table_name = "test_upload"
result = import_to_postgres(df, table_name, db)
print("import_to_postgres() OK:", result)

# Verify it was imported
rows = db.execute_query(f"SELECT * FROM {table_name}")
print(f"Rows in DB: {len(rows)}")
print("First row:", rows[0])

db.disconnect()
print("✅ file_uploader works!")