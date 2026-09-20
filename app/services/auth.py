import logging
from fastapi import Header, HTTPException, status
from jose import JWTError, jwt
from app.config import JWT_SECRET

logger = logging.getLogger("uvicorn.default")


async def verify_auth(
    x_internal_secret: str | None = Header(None),
    authorization: str | None = Header(None),
) -> dict:
    """
    Verify incoming requests via shared secret header (x-internal-secret)
    or JWT Bearer token (Authorization header).
    """
    # 1. Check shared secret header
    if x_internal_secret and JWT_SECRET and x_internal_secret == JWT_SECRET:
        return {"authenticated": True, "method": "shared_secret"}

    # 2. Check Authorization header (Bearer token or secret)
    if authorization:
        token = authorization
        if token.startswith("Bearer "):
            token = token[7:].strip()

        # Check if direct token matches shared secret
        if JWT_SECRET and token == JWT_SECRET:
            return {"authenticated": True, "method": "shared_secret_bearer"}

        # Check if token is a valid JWT
        if JWT_SECRET:
            try:
                payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
                return {"authenticated": True, "method": "jwt", "payload": payload}
            except JWTError as e:
                logger.warning(f"JWT verification failed: {e}")

    logger.warning("Unauthorized access attempt")
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized: Valid shared secret or JWT token required",
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user_id(
    authorization: str | None = Header(None),
) -> str:
    """
    Extract and verify JWT token from Authorization header,
    returning the authenticated user's ID.
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header is required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = authorization
    if token.startswith("Bearer "):
        token = token[7:].strip()

    if not JWT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="JWT_SECRET is not configured on server",
        )

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        user_id = payload.get("id") or payload.get("_id") or payload.get("userId")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload: missing user id",
            )
        return str(user_id)
    except JWTError as e:
        logger.warning(f"JWT decode error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
