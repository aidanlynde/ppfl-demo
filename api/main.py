# api/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from config.settings import settings
from api.routers.fl_routes import router as fl_router
from fastapi.responses import JSONResponse
from fastapi.middleware.gzip import GZipMiddleware
from api.utils.session_manager import session_manager
import asyncio

app = FastAPI(
    title="Privacy-Preserving Federated Learning Demo",
    description="API for demonstrating federated learning with differential privacy",
    version="0.1.0"
)

# Configure CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)
# Add Gzip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

async def cleanup_sessions():
    while True:
        await asyncio.sleep(300)  # Run every 5 minutes
        session_manager.cleanup_expired_sessions()

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_sessions())

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )

# Add options handler for CORS preflight
@app.options("/{full_path:path}")
async def options_handler():
    return JSONResponse(
        content="OK",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        },
    )

# Include FL routes
app.include_router(fl_router, prefix="/api/fl", tags=["Federated Learning"])

@app.post("/test-init")
async def test_initialize():
    return {
        "status": "success",
        "message": "Test initialization endpoint working"
    }

@app.get("/")
async def root():
    """Root endpoint to verify API is running"""
    return {
        "status": "online",
        "message": "Welcome to the PPFL Demo API",
        "version": "0.1.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "environment": settings.ENVIRONMENT
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",  # Updated import path
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )