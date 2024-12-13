# api/main.py

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from config.settings import settings
from api.routers.fl_routes import router as fl_router

app = FastAPI(
    title="Privacy-Preserving Federated Learning Demo",
    description="API for demonstrating federated learning with differential privacy",
    version="0.1.0"
)

# Configure CORS for your Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include FL routes
app.include_router(fl_router, prefix="/api/fl", tags=["Federated Learning"])

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