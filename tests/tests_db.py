import sys
import os
import traceback

# Hauptordner zum Pfad hinzufügen
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from dotenv import load_dotenv
# .env explizit im Hauptordner laden
load_dotenv(os.path.join(project_root, '.env'))

# Debug
print(f"HOST: {os.getenv('DB_HOST')}")
print(f"PORT: {os.getenv('DB_PORT')}")
print(f"NAME: {os.getenv('DB_NAME')}")
print(f"USER: {os.getenv('DB_USER')}")
print(f"PASS: {'*' * len(os.getenv('DB_PASSWORD', '')) or 'LEER'}")

from database.db_connector import DatabaseConnector

db = DatabaseConnector()

if db.connect():
    print("✅ Verbindung erfolgreich!")
    print("\n📋 Schema:")
    print(db.get_schema())
    db.disconnect()
else:
    print("❌ Verbindung fehlgeschlagen!")