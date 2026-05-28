import logging
import os
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, BackgroundTasks
from sqlalchemy.orm import Session
from typing import List
from backend.database import get_db
from backend.models import Report, User, AbnormalValue
from backend.schemas import ReportResponse, ReportDetailResponse
from backend.middleware.auth import get_current_user
from backend.services.ingestion import validate_and_save_file
from backend.services.extraction import DocumentExtractor
from backend.services.preprocessing import extract_numerical_values
from backend.services.abnormality import analyze_abnormalities
from backend.services.chunking import chunk_document
from backend.services.vector_store import VectorStoreService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/reports", tags=["Reports"])

# Background Task for parsing and indexing reports
def process_report_task(report_id: str, user_id: str, db_session_factory):
    # Fetch DB session inside background thread to avoid thread safety issues
    db: Session = db_session_factory()
    try:
        # 1. Update status to PROCESSING
        report = db.query(Report).filter(Report.id == report_id).first()
        if not report:
            logger.error(f"Background task failed: Report {report_id} not found in DB.")
            return

        report.status = "PROCESSING"
        db.commit()

        # 2. Extract text (PDF / Image OCR)
        extractor = DocumentExtractor()
        extraction_res = extractor.extract(report.file_path, report.mime_type)
        
        report.extracted_text = extraction_res.raw_text
        report.page_count = extraction_res.page_count
        db.commit()

        # 3. Clean text & extract numerical test values
        extracted_values = extract_numerical_values(extraction_res.raw_text)
        
        # 4. Abnormality detection & store in DB
        abnormalities_list = analyze_abnormalities(extracted_values)
        for ab in abnormalities_list:
            db_ab = AbnormalValue(
                report_id=report_id,
                test_name=ab["test_name"],
                value=ab["value"],
                unit=ab["unit"],
                normal_range=ab["normal_range"],
                status=ab["status"]
            )
            db.add(db_ab)
        db.commit()

        # 5. Chunk and index into vector store
        chunks = chunk_document(extraction_res.raw_text, report_id, report.filename)
        vector_store = VectorStoreService()
        vector_store.index_report(report_id, chunks)

        # 6. Update status to READY
        report.status = "READY"
        db.commit()
        logger.info(f"Report {report_id} processed successfully.")

    except Exception as e:
        logger.exception(f"Error processing report {report_id} in background task: {str(e)}")
        # Attempt to mark status as FAILED
        try:
            report = db.query(Report).filter(Report.id == report_id).first()
            if report:
                report.status = "FAILED"
                db.commit()
        except Exception as db_ex:
            logger.error(f"Failed to write failure status to DB: {str(db_ex)}")
    finally:
        db.close()

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_report(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Validate and save file locally
    metadata = validate_and_save_file(file, current_user.id)

    # Insert initial pending report in DB
    db_report = Report(
        id=metadata["file_id"],
        user_id=current_user.id,
        filename=metadata["filename"],
        file_path=metadata["file_path"],
        mime_type=metadata["mime_type"],
        status="PENDING",
        page_count=0,
        is_deleted=0
    )
    db.add(db_report)
    db.commit()
    db.refresh(db_report)

    # Trigger async extraction & embedding process
    # We pass the SessionLocal factory so the task can open its own database session safely
    from backend.database import SessionLocal
    background_tasks.add_task(process_report_task, db_report.id, current_user.id, SessionLocal)

    return {
        "report_id": db_report.id,
        "filename": db_report.filename,
        "status": db_report.status,
        "extracted_text_preview": ""
    }

@router.get("/", response_model=List[ReportResponse])
def get_reports(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Retrieve all active reports for user
    reports = db.query(Report).filter(
        Report.user_id == current_user.id,
        Report.is_deleted == 0
    ).order_by(Report.created_at.desc()).all()
    
    # We construct ReportResponse items manually or map schema fields
    res = []
    for r in reports:
        preview = r.extracted_text[:200] + "..." if r.extracted_text else None
        res.append(ReportResponse(
            id=r.id,
            user_id=r.user_id,
            filename=r.filename,
            mime_type=r.mime_type,
            status=r.status,
            page_count=r.page_count,
            created_at=r.created_at,
            updated_at=r.updated_at,
            extracted_text_preview=preview,
            abnormalities=r.abnormalities
        ))
    return res

@router.get("/{report_id}", response_model=ReportDetailResponse)
def get_report_detail(
    report_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Retrieve report scoped to current user
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

    return report

@router.delete("/{report_id}", status_code=status.HTTP_200_OK)
def delete_report(
    report_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Soft delete report scoped to current user
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

    # Soft delete record
    report.is_deleted = 1
    db.commit()

    # Async delete vector collection in background
    vector_store = VectorStoreService()
    try:
        vector_store.delete_report_collection(report_id)
    except Exception as e:
        logger.warning(f"Failed to remove vector index: {str(e)}")

    # Safely clean up physical files
    if os.path.exists(report.file_path):
        try:
            os.remove(report.file_path)
        except Exception as e:
            logger.warning(f"Failed to remove physical file: {str(e)}")

    return {"message": "Report deleted successfully."}
