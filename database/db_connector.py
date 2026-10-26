"""
db_connector.py
---------------
PostgreSQL-Datenbankanbindung mit automatischer Schema-Inspektion.
Stellt die Verbindung zur Datenbank her und liest das Schema aus,
das dem LangChain-Agenten als Kontext bereitgestellt wird.
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError

load_dotenv()


class DatabaseConnector:
    """
    Verwaltet die Verbindung zur PostgreSQL-Datenbank und
    stellt Methoden zur Schema-Inspektion und SQL-Ausführung bereit.
    """

    def __init__(
        self,
        host: str = None,
        port: int = None,
        dbname: str = None,
        user: str = None,
        password: str = None,
    ):
        self.host = host or os.getenv("DB_HOST", "localhost")
        self.port = port or int(os.getenv("DB_PORT", 5432))
        self.dbname = dbname or os.getenv("DB_NAME")
        self.user = user or os.getenv("DB_USER")
        self.password = password or os.getenv("DB_PASSWORD")
        self.engine = None

    def connect(self) -> bool:
        """
        Stellt die Verbindung zur PostgreSQL-Datenbank her.
        Gibt True zurück bei Erfolg, False bei Fehler.
        """
        try:
            url = (
                f"postgresql+psycopg2://{self.user}:{self.password}"
                f"@{self.host}:{self.port}/{self.dbname}"
            )
            self.engine = create_engine(url)
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except SQLAlchemyError as e:
            print(f"Verbindungsfehler: {e}")
            return False

    def get_schema(self) -> str:
        """
        Liest das Datenbankschema automatisch aus und gibt es
        als formatierten String zurück, der in den Agenten-Prompt
        eingebettet werden kann.
        Unterstützt alle PostgreSQL-Schemas (nicht nur 'public').
        """
        if not self.engine:
            raise ConnectionError("Keine aktive Datenbankverbindung.")

        inspector = inspect(self.engine)
        schema_parts = []

        for schema_name in inspector.get_schema_names():
            if schema_name in ('pg_catalog', 'information_schema'):
                continue

            for table_name in inspector.get_table_names(schema=schema_name):
                columns = inspector.get_columns(table_name, schema=schema_name)
                foreign_keys = inspector.get_foreign_keys(table_name, schema=schema_name)

                col_definitions = []
                for col in columns:
                    col_str = f"  {col['name']} ({col['type']})"
                    if not col.get("nullable", True):
                        col_str += " NOT NULL"
                    col_definitions.append(col_str)

                fk_definitions = []
                for fk in foreign_keys:
                    fk_str = (
                        f"  FK: {fk['constrained_columns']} -> "
                        f"{fk['referred_table']}.{fk['referred_columns']}"
                    )
                    fk_definitions.append(fk_str)

                # Schema-qualifizierter Tabellenname
                # z.B. humanresources.employee statt nur employee
                qualified_name = (
                    f"{schema_name}.{table_name}"
                    if schema_name != "public"
                    else table_name
                )

                table_schema = f"Tabelle: {qualified_name}\n"
                table_schema += "Spalten:\n" + "\n".join(col_definitions)
                if fk_definitions:
                    table_schema += "\nFremdschlüssel:\n" + "\n".join(fk_definitions)

                schema_parts.append(table_schema)

        return "\n\n".join(schema_parts)

    def execute_query(self, sql: str) -> list[dict]:
        """
        Führt eine SQL-Abfrage aus und gibt die Ergebnisse
        als Liste von Dictionaries zurück.
        """
        if not self.engine:
            raise ConnectionError("Keine aktive Datenbankverbindung.")

        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql))
                columns = result.keys()
                rows = result.fetchall()
                return [dict(zip(columns, row)) for row in rows]
        except SQLAlchemyError as e:
            raise ValueError(f"SQL-Ausführungsfehler: {e}")

    def get_table_names(self) -> list[str]:
        """Gibt eine Liste aller Tabellennamen zurück (alle Schemas)."""
        if not self.engine:
            raise ConnectionError("Keine aktive Datenbankverbindung.")
        inspector = inspect(self.engine)
        tables = []
        for schema_name in inspector.get_schema_names():
            if schema_name in ('pg_catalog', 'information_schema'):
                continue
            for t in inspector.get_table_names(schema=schema_name):
                qualified = (
                    f"{schema_name}.{t}"
                    if schema_name != "public"
                    else t
                )
                tables.append(qualified)
        return tables

    def get_connection_url(self) -> str:
        """Gibt die SQLAlchemy Connection URL zurück (ohne Passwort)."""
        return (
            f"postgresql+psycopg2://{self.user}:***"
            f"@{self.host}:{self.port}/{self.dbname}"
        )

    def disconnect(self):
        """Schließt die Datenbankverbindung."""
        if self.engine:
            self.engine.dispose()
            self.engine = None