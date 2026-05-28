from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import List, Optional, Literal
from uuid import UUID
from datetime import datetime

# --- AUTH SCHEMAS ---

class UserRegister(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=100)

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    email: Optional[str] = None
    user_id: Optional[str] = None

class UserResponse(BaseModel):
    id: str
    username: str
    email: EmailStr
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

# --- ABNORMAL VALUES SCHEMAS ---

class AbnormalValueResponse(BaseModel):
    test_name: str
    value: float
    unit: str
    normal_range: str
    status: Literal["CRITICAL", "HIGH", "LOW", "BORDERLINE", "NORMAL"]
    explanation: Optional[str] = ""

    model_config = ConfigDict(from_attributes=True)

# --- REPORT SCHEMAS ---

class ReportResponse(BaseModel):
    id: str
    user_id: str
    filename: str
    mime_type: str
    status: str
    page_count: int
    created_at: datetime
    updated_at: datetime
    extracted_text_preview: Optional[str] = None
    abnormalities: List[AbnormalValueResponse] = []

    model_config = ConfigDict(from_attributes=True)

class ReportDetailResponse(BaseModel):
    id: str
    user_id: str
    filename: str
    mime_type: str
    status: str
    page_count: int
    extracted_text: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    abnormalities: List[AbnormalValueResponse] = []

    model_config = ConfigDict(from_attributes=True)

# --- CHAT SCHEMAS ---

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    session_id: Optional[UUID] = None

class SourceChunk(BaseModel):
    chunk_text: str
    chunk_index: int
    page_number: Optional[int] = 1
    relevance_score: Optional[float] = 0.0

class RAGResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    session_id: str
    tokens_used: int = 0

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    sources: Optional[List[SourceChunk]] = None
    timestamp: str

class ChatSessionResponse(BaseModel):
    id: str
    report_id: str
    messages: List[ChatMessage] = []
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
