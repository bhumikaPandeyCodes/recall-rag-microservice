from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database import (
    connect_to_mongo,
    close_mongo_connection,
    get_database,
)
from app.routes.ingest import router as ingest_router
from app.routes.query import router as query_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await connect_to_mongo()
    yield
    # Shutdown
    await close_mongo_connection()


app = FastAPI(
    title="Recall RAG Service",
    description="RAG and vector search service for Recall",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(ingest_router)
app.include_router(query_router)


@app.get("/")
async def read_root():
    return {"message": "Recall RAG Service is running"}


@app.get("/health")
async def health_check():
    try:
        db = get_database()
        await db.command("ping")
        return {
            "status": "healthy",
            "database": "connected",
            "db_name": db.name,
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
        }
