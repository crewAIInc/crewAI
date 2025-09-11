import json
import logging
from typing import Any, Dict, List, Optional

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Integer,
    String,
    Table,
    MetaData,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import text

from crewai.memory.storage.interface import Storage

logger = logging.getLogger(__name__)
Base = declarative_base()


class LTMSQLiteStorage(Storage):
    """
    An updated implementation of the Storage interface using SQLite as the backend.
    This version includes improved querying methods and metadata handling.
    """

    def __init__(self, db_path: str = "ltm_storage.db", table_name: str = "long_term_memories"):
        self.db_path = db_path
        self.table_name = table_name
        self.engine = create_engine(f"sqlite:///{db_path}")
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        self._initialize_db()

    def _initialize_db(self):
        """Initialize the database and create the table if it doesn't exist."""
        Base.metadata.create_all(self.engine)
        
        # Create table dynamically
        metadata = MetaData()
        self.table = Table(
            self.table_name,
            metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("agent", String, nullable=False),
            Column("key", String, nullable=False),
            Column("value", JSON, nullable=False),
            Column("metadata", JSON, nullable=True),
            Column("created_at", DateTime, default=func.current_timestamp()),
            Column("updated_at", DateTime, default=func.current_timestamp(), onupdate=func.current_timestamp()),
            extend_existing=True
        )
        metadata.create_all(self.engine)

    def save(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save a value with the given key and optional metadata."""
        data = {
            "agent": "default",
            "key": key,
            "value": json.dumps(value) if not isinstance(value, str) else value,
            "metadata": metadata
        }
        
        # Check if the key already exists
        existing = self.session.execute(
            select(self.table).where(self.table.c.key == key)
        ).first()
        
        if existing:
            # Update existing record
            self.session.execute(
                self.table.update().where(self.table.c.key == key).values(**data)
            )
        else:
            # Insert new record
            self.session.execute(self.table.insert().values(**data))
        
        self.session.commit()

    def load(self, key: str) -> Optional[Any]:
        """Load the value associated with the given key."""
        result = self.session.execute(
            select(self.table).where(self.table.c.key == key)
        ).first()
        
        if result:
            value = result.value
            try:
                return json.loads(value) if isinstance(value, str) else value
            except json.JSONDecodeError:
                return value
        return None

    def delete(self, key: str) -> None:
        """Delete the value associated with the given key."""
        self.session.execute(
            self.table.delete().where(self.table.c.key == key)
        )
        self.session.commit()

    def exists(self, key: str) -> bool:
        """Check if a key exists in the storage."""
        result = self.session.execute(
            select(self.table).where(self.table.c.key == key)
        ).first()
        return result is not None

    def reset(self) -> None:
        """Reset the storage by deleting all records from the table."""
        # Use SQLAlchemy's table operations instead of raw SQL
        self.session.execute(self.table.delete())
        self.session.commit()
        logger.info(f"Table '{self.table_name}' has been reset.")

    def list_keys(self) -> List[str]:
        """List all keys in the storage."""
        results = self.session.execute(select(self.table.c.key)).all()
        return [row.key for row in results]

    def close(self) -> None:
        """Close the database connection."""
        self.session.close()
