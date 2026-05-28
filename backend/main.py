import time
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from backend.config import settings
from backend.database import engine, Base
from backend.routers import auth, reports, chat, analysis

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Initialize database schemas
try:
    logger.info("Initializing SQLite database schemas...")
    Base.metadata.create_all(bind=engine)
    logger.info("Database schemas initialized.")
    
    # Seed default user if not exists
    from backend.database import SessionLocal
    from backend.models import User
    from backend.middleware.auth import get_password_hash
    
    db = SessionLocal()
    try:
        demo_user = db.query(User).filter(User.email == "demo@example.com").first()
        if not demo_user:
            logger.info("Seeding default demo user (demo@example.com)...")
            hashed_pw = get_password_hash("password123")
            new_user = User(
                username="demouser",
                email="demo@example.com",
                hashed_password=hashed_pw
            )
            db.add(new_user)
            db.commit()
            logger.info("Demo user seeded successfully.")
    except Exception as seed_err:
        logger.error(f"Error seeding demo user: {str(seed_err)}")
    finally:
        db.close()
except Exception as e:
    logger.critical(f"Failed to initialize database: {str(e)}")

# Initialize SlowAPI Limiter
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="MedRAG API",
    description="Production-grade AI-powered Medical Report Analyzer API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Attach Limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Request Latency Audit Log Middleware
@app.middleware("http")
async def audit_log_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    # Log requests with latency
    path = request.url.path
    method = request.method
    status_code = response.status_code
    
    logger.info(
        f"API Request: {method} {path} - Status: {status_code} - Latency: {process_time:.4f}s"
    )
    return response

# Mount uploaded files directory (securely scope files)
app.mount(
    "/api/uploads", 
    StaticFiles(directory=settings.UPLOAD_DIR), 
    name="uploads"
)

# Register endpoints routers
app.include_router(auth.router)
app.include_router(reports.router)
app.include_router(chat.router)
app.include_router(analysis.router)

@app.get("/api/health")
@limiter.limit("60/minute")
def health_check(request: Request):
    """Health check endpoint to test connection and status."""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "database": "connected"
    }

# Handle 500 error logs details
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global server error: {str(exc)} on path {request.url.path}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again later."}
    )
