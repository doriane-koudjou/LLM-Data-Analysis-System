from database.multi_db_connector import MultiDatabaseConnector
from dotenv import load_dotenv
import os

load_dotenv()
c = MultiDatabaseConnector()
c.add_connection(
    name='adventureworks',
    host=os.getenv('DB_HOST', 'localhost'),
    port=int(os.getenv('DB_PORT', 5432)),
    dbname=os.getenv('ADVENTUREWORKS_DB', 'adventureworks'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD')
)

schema = c.get_all_schemas()
# Suche nach humanresources
if 'humanresources' in schema.lower():
    idx = schema.lower().find('humanresources')
    print("✅ AdventureWorks-Schema gefunden:")
    print(schema[idx:idx+1000])
else:
    print("❌ humanresources immer noch nicht im Schema")
    print("Schema-Länge:", len(schema))
    print("Erste 500 Zeichen:", schema[:500])