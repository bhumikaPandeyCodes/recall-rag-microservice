import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from app.config import MONGODB_URI

logger = logging.getLogger("uvicorn.default")


class Database:
    client: AsyncIOMotorClient | None = None
    db: AsyncIOMotorDatabase | None = None


db_instance = Database()


async def connect_to_mongo() -> None:
    """Initialize AsyncIOMotorClient and test connection."""
    if not MONGODB_URI:
        logger.error("MONGODB_URI is not set in .env!")
        raise ValueError("MONGODB_URI environment variable is missing")

    logger.info("Connecting to MongoDB via Motor...")
    db_instance.client = AsyncIOMotorClient(MONGODB_URI)

    # Use database from URI path if specified, else fallback to 'test' (matching Mongoose default)
    try:
        db_instance.db = db_instance.client.get_default_database()
    except Exception:
        db_instance.db = db_instance.client["test"]

    # Verify connection
    await db_instance.client.admin.command("ping")
    logger.info(f"Connected to MongoDB successfully. Target Database: '{db_instance.db.name}'")


async def close_mongo_connection() -> None:
    """Close MongoDB connection on application shutdown."""
    if db_instance.client:
        logger.info("Closing MongoDB connection...")
        db_instance.client.close()
        db_instance.client = None
        db_instance.db = None
        logger.info("MongoDB connection closed.")


def get_database() -> AsyncIOMotorDatabase:
    """Get active AsyncIOMotorDatabase instance."""
    if db_instance.db is None:
        raise RuntimeError("Database not initialized. Ensure connect_to_mongo() was called.")
    return db_instance.db


def get_chunks_collection():
    """Get 'chunks' collection."""
    return get_database()["chunks"]


def get_contents_collection():
    """Get 'contents' collection."""
    return get_database()["contents"]
