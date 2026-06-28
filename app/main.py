from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import logging
import sys

from config.settings import settings
from app.models import HealthResponse
from app.routes import documents, queries, retrieval
from app.utils.exceptions import AIKnowledgeException
from app.utils.rate_limiter import check_rate_limit

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Knowledge Support System",
    description="An AI agent-based document knowledge and decision support system",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    if not check_rate_limit(client_ip):
        return JSONResponse(status_code=429, content={"detail": "Rate limit exceeded. Try again later."})
    return await call_next(request)


# Custom exception handlers
@app.exception_handler(AIKnowledgeException)
async def ai_knowledge_exception_handler(request, exc):
    """Handle custom AI Knowledge exceptions"""
    response = exc.to_response()
    return JSONResponse(
        status_code=response.status_code,
        content=response.model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "type": "validation_error"},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "type": "server_error"},
    )


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint to verify service is running"""
    logger.info("Health check called")
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        environment=settings.environment
    )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "AI Knowledge Support System",
        "version": "0.1.0",
        "docs_url": "/docs",
        "openapi_url": "/openapi.json"
    }


# Include routers
app.include_router(
    documents.router,
    tags=["documents"],
)
app.include_router(
    queries.router,
    tags=["queries"],
)
app.include_router(
    retrieval.router,
    tags=["retrieval"],
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development"
    )
