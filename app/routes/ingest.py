import logging
from fastapi import APIRouter, Depends, HTTPException, status
from app.models.ingest import IngestRequest, IngestResponse
from app.services.auth import verify_auth
from app.services.ingestion import process_document_ingestion

logger = logging.getLogger("uvicorn.default")

router = APIRouter(tags=["Ingestion"])


@router.post(
    "/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_200_OK,
    summary="Ingest and process a document for RAG",
)
async def ingest_document(
    payload: IngestRequest,
    auth_data: dict = Depends(verify_auth),
):
    """
    Ingest a document from Cloudinary:
    1. Authenticate request via shared secret or JWT.
    2. Download document from Cloudinary URL.
    3. Extract text into ~1000-character chunks with page numbers.
    4. Generate Gemini embeddings for each chunk.
    5. Save chunks to MongoDB with user/content isolation.
    6. Update Content document status to 'ready' (or 'failed').
    """
    logger.info(f"Received /ingest request for contentId: {payload.contentId} from user: {payload.userId}")
    
    try:
        total_chunks = await process_document_ingestion(
            content_id=payload.contentId,
            user_id=payload.userId,
            file_url=payload.fileUrl,
        )
        
        return IngestResponse(
            success=True,
            message="Document successfully processed, embedded, and saved",
            contentId=payload.contentId,
            totalChunks=total_chunks,
            status="ready",
        )
        
    except Exception as e:
        logger.error(f"Error processing /ingest request: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {str(e)}",
        )
