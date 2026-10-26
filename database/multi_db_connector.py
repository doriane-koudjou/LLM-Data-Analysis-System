"""
multi_db_connector.py
---------------------
Manages multiple database connections simultaneously.
Allows the agent to query across different databases in a single session.
"""

from database.db_connector import DatabaseConnector


class MultiDatabaseConnector:
    """
    Manages multiple named database connections.
    The agent can query any registered database by name.
    """

    def __init__(self):
        self.connections: dict[str, DatabaseConnector] = {}

    def add_connection(
        self,
        name: str,
        host: str,
        port: int,
        dbname: str,
        user: str,
        password: str,
    ) -> bool:
        """
        Adds and connects a new database under a given name.
        Returns True on success, False on failure.
        """
        connector = DatabaseConnector(
            host=host,
            port=port,
            dbname=dbname,
            user=user,
            password=password,
        )
        if connector.connect():
            self.connections[name] = connector
            return True
        return False

    def remove_connection(self, name: str):
        """Disconnects and removes a database connection."""
        if name in self.connections:
            self.connections[name].disconnect()
            del self.connections[name]

    def get_connector(self, name: str) -> DatabaseConnector | None:
        """Returns the connector for a given database name."""
        return self.connections.get(name)

    def get_all_schemas(self) -> str:
        """
        Returns the schemas of all connected databases as a combined string.
        Used to give the agent full context of all available data.
        """
        if not self.connections:
            return "No databases connected."

        schema_parts = []
        for name, connector in self.connections.items():
            schema = connector.get_schema()
            schema_parts.append(
                f"=== Database: {name} ===\n{schema}"
            )

        return "\n\n".join(schema_parts)

    def execute_query(self, db_name: str, sql: str) -> list[dict]:
        """
        Executes a SQL query on a specific database by name.
        Raises ValueError if the database is not found.
        """
        connector = self.connections.get(db_name)
        if not connector:
            available = list(self.connections.keys())
            raise ValueError(
                f"Database '{db_name}' not found. "
                f"Available databases: {available}"
            )
        return connector.execute_query(sql)

    def get_connection_names(self) -> list[str]:
        """Returns a list of all connected database names."""
        return list(self.connections.keys())

    def disconnect_all(self):
        """Disconnects all database connections."""
        for connector in self.connections.values():
            connector.disconnect()
        self.connections.clear()

    def __len__(self):
        return len(self.connections)