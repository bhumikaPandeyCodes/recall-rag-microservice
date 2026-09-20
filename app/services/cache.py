import json
import logging
from upstash_redis.asyncio import Redis as AsyncRedis
from app.config import UPSTASH_REDIS_REST_URL, UPSTASH_REDIS_REST_TOKEN

logger = logging.getLogger("uvicorn.default")

redis_client: AsyncRedis | None = None

if UPSTASH_REDIS_REST_URL and UPSTASH_REDIS_REST_TOKEN:
    try:
        redis_client = AsyncRedis(url=UPSTASH_REDIS_REST_URL, token=UPSTASH_REDIS_REST_TOKEN)
        logger.info("Upstash Redis client initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize Upstash Redis: {e}")


async def get_cached_response(key: str) -> dict | None:
    """Retrieve cached query response from Redis."""
    if not redis_client:
        return None
    try:
        val = await redis_client.get(key)
        if val:
            logger.info(f"Redis Cache HIT for key: {key}")
            if isinstance(val, str):
                return json.loads(val)
            elif isinstance(val, dict):
                return val
    except Exception as e:
        logger.warning(f"Redis get cache error: {e}")
    return None


async def set_cached_response(key: str, data: dict, ttl_seconds: int = 86400) -> None:
    """Save query response to Redis with TTL (default 24 hours)."""
    if not redis_client:
        return
    try:
        await redis_client.set(key, json.dumps(data), ex=ttl_seconds)
        logger.info(f"Saved response to Redis cache for key: {key} (TTL {ttl_seconds}s)")
    except Exception as e:
        logger.warning(f"Redis set cache error: {e}")
