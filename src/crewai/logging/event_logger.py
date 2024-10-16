import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import JSON, Column, DateTime, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()

Base = declarative_base()


class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True)
    event_name = Column(String)
    timestamp = Column(DateTime, default=datetime.now(timezone.utc))
    crew_id = Column(String)
    data = Column(JSON)
    error_type = Column(String)
    error_message = Column(String)
    traceback = Column(String)


DATABASE_URL = os.getenv("CREWAI_DATABASE_URL", "sqlite:///crew_events.db")
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)  # Use a different name to avoid confusion

# Create a session instance
session: Session = SessionLocal()


class EventLogger:
    def __init__(self, session: Session):
        self.session = session

    def log_event(self, event_name: str, *args: Any, **kwargs: Any) -> None:
        event = Event(
            event_name=event_name,
            crew_id=kwargs.get("crew_id", ""),
            data=kwargs,
            error_type=kwargs.get("error_type", ""),
            error_message=kwargs.get("error_message", ""),
            traceback=kwargs.get("traceback", ""),
        )
        self.session.add(event)
        self.session.commit()


event_logger = EventLogger(session)
