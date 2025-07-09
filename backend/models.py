from sqlalchemy import create_engine, Column, String, Text, DateTime, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import json

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./memora_users.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    password_hash = Column(String)
    full_name = Column(String)
    age = Column(String)
    preferred_language = Column(String)
    background = Column(Text)
    interests = Column(Text)  # JSON string
    conversation_preferences = Column(Text)  # JSON string
    technology_usage = Column(String)
    conversation_goals = Column(Text)  # JSON string
    additional_info = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        """Convert user object to dictionary"""
        return {
            "id": self.id,
            "email": self.email,
            "full_name": self.full_name,
            "age": self.age,
            "preferred_language": self.preferred_language,
            "background": self.background,
            "interests": json.loads(self.interests) if self.interests else [],
            "conversation_preferences": json.loads(self.conversation_preferences) if self.conversation_preferences else [],
            "technology_usage": self.technology_usage,
            "conversation_goals": json.loads(self.conversation_goals) if self.conversation_goals else [],
            "additional_info": self.additional_info,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

# Create tables
Base.metadata.create_all(bind=engine)

def get_db():
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close() 