import io
import logging
from datetime import datetime, timezone
from bson import ObjectId
import httpx
import pypdf
import google.generativeai as genai
from app.config import GEMINI_API_KEY
from app.database import get_chunks_collection, get_contents_collection

logger = logging.getLogger("uvicorn.default")

# Configure Gemini AI once
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def chunk_text(text: str, chunk_size: int = 1000) -> list[str]:
    """Split text into chunks of approximately chunk_size characters."""
    if not text:
        return []
    
    # Split text cleanly preserving chunks up to chunk_size
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end
    return chunks


async def download_file(file_url: str) -> bytes:
    """Download document bytes from public Cloudinary URL."""
    logger.info(f"Downloading document from: {file_url}")
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        response = await client.get(file_url)
        response.raise_for_status()
        return response.content


def extract_chunks_from_pdf(file_bytes: bytes, chunk_size: int = 1000) -> list[dict]:
    """
    Extract text page-by-page from PDF bytes and create indexed chunks.
    Returns list of {"text": str, "pageNumber": int}.
    """
    extracted_chunks = []
    
    try:
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        total_pages = len(reader.pages)
        logger.info(f"PDF loaded successfully. Total pages: {total_pages}")
        
        for page_idx, page in enumerate(reader.pages):
            page_num = page_idx + 1
            page_text = page.extract_text() or ""
            page_text = page_text.strip()
            
            if not page_text:
                continue
                
            chunks = chunk_text(page_text, chunk_size=chunk_size)
            for c in chunks:
                extracted_chunks.append({
                    "text": c,
                    "pageNumber": page_num
                })
                
    except Exception as e:
        logger.warning(f"pypdf extraction error: {e}. Trying raw text fallback...")
        try:
            raw_text = file_bytes.decode("utf-8", errors="ignore").strip()
            chunks = chunk_text(raw_text, chunk_size=chunk_size)
            for c in chunks:
                extracted_chunks.append({
                    "text": c,
                    "pageNumber": 1
                })
        except Exception as fallback_err:
            logger.error(f"Fallback extraction failed: {fallback_err}")
            raise ValueError(f"Could not extract text from document: {e}")

    if not extracted_chunks:
        raise ValueError("No readable text found in document")
        
    return extracted_chunks


def generate_embedding(text: str) -> list[float]:
    """Generate 768/3072 dimension vector embedding via Gemini API."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured")
        
    response = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
    )
    return response["embedding"]


async def process_document_ingestion(content_id: str, user_id: str, file_url: str) -> int:
    """
    Complete ingestion pipeline:
    1. Download PDF from Cloudinary
    2. Extract and chunk text with page numbers
    3. Generate Gemini embeddings for each chunk
    4. Save chunks to MongoDB
    5. Update Content status to 'ready' (or 'failed' on error)
    """
    contents_col = get_contents_collection()
    chunks_col = get_chunks_collection()
    
    c_oid = ObjectId(content_id)
    u_oid = ObjectId(user_id)
    now = datetime.now(timezone.utc)
    
    try:
        logger.info(f"Starting ingestion for Content: {content_id}, User: {user_id}")
        
        # 1. Download file
        file_bytes = await download_file(file_url)
        
        # 2. Extract chunks
        raw_chunks = extract_chunks_from_pdf(file_bytes, chunk_size=1000)
        logger.info(f"Extracted {len(raw_chunks)} chunks from document")
        
        # 3. Clean up any existing chunks for this contentId (idempotency)
        await chunks_col.delete_many({"contentId": c_oid, "userId": u_oid})
        
        # 4. Generate embeddings and prepare documents
        chunk_docs = []
        for idx, item in enumerate(raw_chunks):
            embedding = generate_embedding(item["text"])
            chunk_doc = {
                "contentId": c_oid,
                "userId": u_oid,
                "text": item["text"],
                "embedding": embedding,
                "embeddingVersion": "gemini-embedding-001-v1",
                "pageNumber": item["pageNumber"],
                "chunkIndex": idx,
                "createdAt": now,
                "updatedAt": now,
            }
            chunk_docs.append(chunk_doc)
            
        # 5. Bulk insert chunks to MongoDB
        if chunk_docs:
            await chunks_col.insert_many(chunk_docs)
            logger.info(f"Successfully saved {len(chunk_docs)} chunks to MongoDB")
            
        # 6. Update Content document to 'ready'
        await contents_col.update_one(
            {"_id": c_oid},
            {
                "$set": {
                    "status": "ready",
                    "errorMessage": None,
                    "updatedAt": datetime.now(timezone.utc),
                }
            }
        )
        logger.info(f"Content {content_id} status updated to 'ready'")
        return len(chunk_docs)
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Ingestion failed for Content {content_id}: {error_msg}")
        
        # Update Content document to 'failed'
        await contents_col.update_one(
            {"_id": c_oid},
            {
                "$set": {
                    "status": "failed",
                    "errorMessage": error_msg,
                    "updatedAt": datetime.now(timezone.utc),
                }
            }
        )
        raise e
