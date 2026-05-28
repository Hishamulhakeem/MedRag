import json
import logging
import uuid
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from backend.database import get_db
from backend.models import Report, User, ChatSession
from backend.schemas import QuestionRequest, ChatSessionResponse
from backend.middleware.auth import get_current_user
from backend.services.rag_chain import RAGChainService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["Chat"])
rag_service = RAGChainService()

@router.post("/{report_id}/ask")
async def ask_question(
    report_id: str,
    body: QuestionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Asks a natural-language question about an uploaded report.
    Returns a streaming Server-Sent Events (SSE) stream.
    """
    # 1. Verify report exists and belongs to user
    report = db.query(Report).filter(
        Report.id == report_id,
        Report.user_id == current_user.id,
        Report.is_deleted == 0
    ).first()

    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found or has been deleted."
        )

    # 2. Get or create ChatSession
    session_id = str(body.session_id) if body.session_id else None
    
    if session_id:
        chat_session = db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.report_id == report_id,
            ChatSession.user_id == current_user.id
        ).first()
        if not chat_session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Specified chat session not found."
            )
    else:
        # Look for most recent session or create new
        chat_session = db.query(ChatSession).filter(
            ChatSession.report_id == report_id,
            ChatSession.user_id == current_user.id
        ).order_by(ChatSession.created_at.desc()).first()

        if not chat_session:
            chat_session = ChatSession(
                id=str(uuid.uuid4()),
                report_id=report_id,
                user_id=current_user.id,
                messages=[]
            )
            db.add(chat_session)
            db.commit()
            db.refresh(chat_session)
            session_id = chat_session.id
        else:
            session_id = chat_session.id

    # 3. Compile history messages
    history_messages = chat_session.messages or []

    # 4. Stream response generator
    async def sse_generator():
        # Buffer to aggregate AI response content for later insertion into DB
        full_answer = ""
        retrieved_sources = []

        try:
            async for data_str in rag_service.ask_question_stream(
                report_id=report_id,
                question=body.question,
                chat_history=history_messages
            ):
                # Clean prefix for raw content parsing
                if data_str.startswith("data: "):
                    content_json = data_str[6:].strip()
                    try:
                        parsed = json.loads(content_json)
                        if "token" in parsed:
                            full_answer += parsed["token"]
                        if "sources" in parsed and not retrieved_sources:
                            retrieved_sources = parsed["sources"]
                    except json.JSONDecodeError:
                        pass
                
                yield data_str

            # 5. Append message pair to session history and persist in DB
            db_session = SessionLocal()  # Open thread-safe session
            try:
                session_record = db_session.query(ChatSession).filter(ChatSession.id == session_id).first()
                if session_record:
                    # Deep copy list to avoid mutation alerts
                    messages_list = list(session_record.messages) if session_record.messages else []
                    
                    # Human Message
                    messages_list.append({
                        "role": "user",
                        "content": body.question,
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    })
                    # AI Message
                    messages_list.append({
                        "role": "assistant",
                        "content": full_answer,
                        "sources": retrieved_sources,
                        "timestamp": datetime.utcnow().isoformat() + "Z"
                    })
                    
                    session_record.messages = messages_list
                    db_session.commit()
            except Exception as db_err:
                logger.error(f"Failed to save chat message to history: {str(db_err)}")
            finally:
                db_session.close()

        except Exception as e:
            logger.error(f"Error in SSE streamer: {str(e)}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    from backend.database import SessionLocal  # Import locally to avoid circular dependencies
    return StreamingResponse(sse_generator(), media_type="text/event-stream")

@router.get("/{report_id}/history", response_model=ChatSessionResponse)
def get_chat_history(
    report_id: str,
    session_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Retrieves full conversation history for a given report and session."""
    # Verify report scope
    report = db.query(Report).filter(
        Report.id == report_id,
        Report.user_id == current_user.id,
        Report.is_deleted == 0
    ).first()

    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found."
        )

    if session_id:
        chat_session = db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.report_id == report_id,
            ChatSession.user_id == current_user.id
        ).first()
    else:
        chat_session = db.query(ChatSession).filter(
            ChatSession.report_id == report_id,
            ChatSession.user_id == current_user.id
        ).order_by(ChatSession.created_at.desc()).first()

    if not chat_session:
        # Return an empty default session
        new_sess = ChatSession(
            id=str(uuid.uuid4()),
            report_id=report_id,
            user_id=current_user.id,
            messages=[]
        )
        db.add(new_sess)
        db.commit()
        db.refresh(new_sess)
        return new_sess

    return chat_session

@router.delete("/{report_id}/history", status_code=status.HTTP_200_OK)
def clear_chat_history(
    report_id: str,
    session_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Clears/resets message history for the session."""
    # Verify report scope
    report = db.query(Report).filter(
        Report.id == report_id,
        Report.user_id == current_user.id,
        Report.is_deleted == 0
    ).first()

    if not report:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Report not found."
        )

    if session_id:
        chat_sessions = db.query(ChatSession).filter(
            ChatSession.id == session_id,
            ChatSession.report_id == report_id,
            ChatSession.user_id == current_user.id
        ).all()
    else:
        chat_sessions = db.query(ChatSession).filter(
            ChatSession.report_id == report_id,
            ChatSession.user_id == current_user.id
        ).all()

    for s in chat_sessions:
        s.messages = []
    
    db.commit()
    return {"message": "Chat history cleared successfully."}
