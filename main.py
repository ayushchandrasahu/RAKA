"""
RAKA Domain AI — Unified Application Entrypoint
Exposes the FastAPI application from backend.main for root execution.
"""
import os
import uvicorn
from backend.main import app, db, RAKA_DEMO_MODE

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    uvicorn.run("backend.main:app", host=host, port=port, reload=True)
