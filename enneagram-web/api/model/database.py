from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class UserResponse(Base):
    __tablename__ = "user_responses"

    id = Column(Integer, primary_key=True, index=True)
    responses = Column(JSON)  # Store all 80 responses
    enneagram_type = Column(Integer)
    wing = Column(Integer)
    traits = Column(JSON)  # Store LLM-identified traits
    evidence = Column(JSON)  # Store evidence for each trait
    follow_up_questions = Column(JSON)  # Store follow-up questions
    created_at = Column(DateTime(timezone=True), server_default=func.now())

# Create SQLite database
engine = create_engine("sqlite:///./data/enneagram.db")
Base.metadata.create_all(bind=engine)
