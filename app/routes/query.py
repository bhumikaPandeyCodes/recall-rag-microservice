import logging
from fastapi import APIRouter, Depends, HTTPException, status
from app.models.query import QueryRequest, QueryResponse, SourceItem
from app.services.auth import get_current_user_id
from app.services.query import process_rag_query

logger = logging.getLogger("uvicorn.default")

router = APIRouter(tags=["Query"])


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query document knowledge base via RAG",
)
async def query_knowledge_base(
    payload: QueryRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    RAG Query Endpoint:
    1. Authenticates via verified JWT and derives `userId` server-side.
    2. Accepts `{ question, contentId (optional) }`.
    3. Checks Redis cache `cache:{userId}:{hash(question+contentId)}`.
    4. On miss: Embeds question, runs vector search across user's chunks,
       and generates a grounded answer with Gemini.
    5. Caches the answer with a 24h TTL.
    6. Returns `{ answer, sources: [{ contentId, title, pageNumber }, ...] }`.
    """
    logger.info(f"Received /query for user: {user_id}, contentId: {payload.contentId}")

    try:
        result = await process_rag_query(
            question=payload.question,
            user_id=user_id,
            content_id=payload.contentId,
        )

        sources = [
            SourceItem(
                contentId=s["contentId"],
                title=s["title"],
                pageNumber=s.get("pageNumber"),
            )
            for s in result.get("sources", [])
        ]

        return QueryResponse(
            answer=result["answer"],
            sources=sources,
        )
    except Exception as e:
        logger.error(f"Error processing /query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}",
        )
