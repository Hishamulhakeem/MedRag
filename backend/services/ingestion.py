import os
import uuid
import shutil
from typing import Dict, Any, Tuple
from fastapi import UploadFile, HTTPException, status
from backend.config import settings

ALLOWED_MIME_TYPES = {
    "application/pdf": ".pdf",
    "image/png": ".png",
    "image/jpeg": ".jpeg",
    "image/jpg": ".jpg",
    "image/tiff": ".tiff",
    "image/webp": ".webp",
}

# Magic bytes (file signatures) for validation
MAGIC_BYTES = {
    b"%PDF": "application/pdf",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"II*\x00": "image/tiff",
    b"MM\x00*": "image/tiff",
    b"RIFF": "image/webp",  # WebP has RIFF....WEBP. We check RIFF and WEBP
}

def validate_and_save_file(upload_file: UploadFile, user_id: str) -> Dict[str, Any]:
    """
    Validates the size and type of the uploaded file, saves it to a secure 
    location scoped by user_id, and returns file metadata.
    """
    # 1. Size Validation (max 20MB = 20 * 1024 * 1024 bytes)
    max_size = 20 * 1024 * 1024
    upload_file.file.seek(0, os.SEEK_END)
    file_size = upload_file.file.tell()
    upload_file.file.seek(0)  # Reset pointer

    if file_size > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File size exceeds the 20MB limit."
        )

    # 2. Signature Validation (Magic Bytes)
    header = upload_file.file.read(16)
    upload_file.file.seek(0)  # Reset pointer

    detected_mime = None
    for signature, mime in MAGIC_BYTES.items():
        if header.startswith(signature):
            detected_mime = mime
            # Special check for WebP because it starts with RIFF
            if mime == "image/webp" and b"WEBP" not in header[8:16]:
                detected_mime = None
            break

    # If magic bytes didn't match, fall back to upload content_type or file extension
    if not detected_mime:
        content_type = upload_file.content_type
        if content_type in ALLOWED_MIME_TYPES:
            detected_mime = content_type

    if not detected_mime or detected_mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed formats: PDF, PNG, JPG, JPEG, TIFF, WEBP."
        )

    # 3. Create target directory
    user_upload_dir = os.path.join(settings.UPLOAD_DIR, user_id)
    os.makedirs(user_upload_dir, exist_ok=True)

    # Generate secure UUID filename
    file_id = str(uuid.uuid4())
    ext = ALLOWED_MIME_TYPES[detected_mime]
    secure_filename = f"{file_id}{ext}"
    file_path = os.path.join(user_upload_dir, secure_filename)

    # Save to disk
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}"
        )

    return {
        "file_id": file_id,
        "filename": upload_file.filename,
        "file_path": file_path,
        "mime_type": detected_mime,
        "size": file_size,
    }
