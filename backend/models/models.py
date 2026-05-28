import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, JSON, Text
from sqlalchemy.orm import relationship
from backend.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    reports = relationship("Report", back_populates="user", cascade="all, delete-orphan")

class Report(Base):
    __tablename__ = "reports"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(255), nullable=False)
    mime_type = Column(String(100), nullable=False)
    status = Column(String(50), default="PENDING")  # PENDING | PROCESSING | READY | FAILED
    extracted_text = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)  # Store generated summary text
    page_count = Column(Integer, default=0)
    is_deleted = Column(Integer, default=0)  # 0 = active, 1 = deleted (SQLite compatible boolean)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="reports")
    sessions = relationship("ChatSession", back_populates="report", cascade="all, delete-orphan")
    abnormalities = relationship("AbnormalValue", back_populates="report", cascade="all, delete-orphan")

class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    report_id = Column(String(36), ForeignKey("reports.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    messages = Column(JSON, default=list)  # List of {role: str, content: str, sources: list, timestamp: str}
    created_at = Column(DateTime, default=datetime.utcnow)

    report = relationship("Report", back_populates="sessions")

class AbnormalValue(Base):
    __tablename__ = "abnormal_values"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    report_id = Column(String(36), ForeignKey("reports.id", ondelete="CASCADE"), nullable=False)
    test_name = Column(String(100), nullable=False)
    value = Column(Float, nullable=False)
    unit = Column(String(50), nullable=True)
    normal_range = Column(String(100), nullable=False)
    status = Column(String(50), nullable=False)  # CRITICAL | HIGH | LOW | BORDERLINE | NORMAL
    created_at = Column(DateTime, default=datetime.utcnow)

    report = relationship("Report", back_populates="abnormalities")
