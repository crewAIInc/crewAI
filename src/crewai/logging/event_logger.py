import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from dotenv import load_dotenv
from sqlalchemy import JSON, Column, DateTime, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session as SQLAlchemySession
from sqlalchemy.orm import sessionmaker

from crewai.utilities.event_emitter import crew_events

load_dotenv()

Base = declarative_base()


class Session:
    _session_id: Optional[str] = None

    @classmethod
    def get_session_id(cls) -> str:
        if cls._session_id is None:
            cls._session_id = str(uuid.uuid4())  # Generate a new UUID
            print(f"Generated new session ID: {cls._session_id}")
        return cls._session_id


class Event(Base):
    __tablename__ = "events"
    id = Column(Integer, primary_key=True)
    event_name = Column(String)
    timestamp = Column(DateTime, default=datetime.now(timezone.utc))
    session_id = Column(String)
    data = Column(JSON)
    error_type = Column(String)
    error_message = Column(String)
    traceback = Column(String)


DATABASE_URL = os.getenv("CREWAI_DATABASE_URL", "sqlite:///crew_events.db")
engine = create_engine(DATABASE_URL)
Base.metadata.create_all(engine)
SessionLocal = sessionmaker(bind=engine)  # Use a different name to avoid confusion

# Create a session instance
session: SQLAlchemySession = SessionLocal()


class EventLogger:
    def __init__(self, session: SQLAlchemySession):
        self.session = session

    def log_event(self, *args: Any, **kwargs: Any) -> None:
        # Extract event name from kwargs
        event_name = kwargs.pop("event", "unknown_event")
        print("Logging event:", event_name)
        print("Args:", args)
        print("Kwargs:", kwargs)

        # Check if args is a single dictionary and unpack it
        if len(args) == 1 and isinstance(args[0], dict):
            args_dict = args[0]
        else:
            # Convert args to a dictionary with keys like 'arg0', 'arg1', etc.
            args_dict = {f"arg{i}": arg for i, arg in enumerate(args)}

        # Merge args_dict and kwargs into a single dictionary
        data = {**args_dict, **kwargs}

        print("Data:", data)

        event = Event(
            event_name=event_name,
            session_id=Session.get_session_id(),
            data=json.dumps(data),
            error_type=kwargs.get("error_type", ""),
            error_message=kwargs.get("error_message", ""),
            traceback=kwargs.get("traceback", ""),
        )

        self.session.add(event)
        self.session.commit()
        print("Successfully logged event:", event_name)


event_logger = EventLogger(session)


print("Connecting event_logger to all signals")
crew_events.on("*", event_logger.log_event)
print("Connected event_logger to all signals")
