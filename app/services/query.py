import hashlib
import logging
import math
from bson import ObjectId
import google.generativeai as genai
from app.config import GEMINI_API_KEY
from app.database import get_chunks_collection, get_contents_collection
from app.models.query import SourceItem
from app.services.cache import get_cached_response, set_cached_response

logger = logging.getLogger("uvicorn.default")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Calculate cosine similarity between two vector lists."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def generate_query_embedding(text: str) -> list[float]:
    """Generate embedding for user question via Gemini API."""
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not configured")
    response = genai.embed_content(
        model="models/gemini-embedding-001",
        content=text,
    )
    return response["embedding"]


async def search_chunks(
    user_id: str,
    query_vector: list[float],
    content_id: str | None = None,
    limit: int = 5,
) -> list[dict]:
    """
    Search chunks using MongoDB Atlas $vectorSearch with fallback to manual
    cosine similarity if index is not present.
    """
    chunks_col = get_chunks_collection()
    u_oid = ObjectId(user_id)
    filter_query: dict = {"userId": u_oid}
    if content_id:
        filter_query["contentId"] = ObjectId(content_id)

    # 1. Try Atlas $vectorSearch aggregation
    try:
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "vector_index",
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": max(limit * 5, 20),
                    "limit": limit,
                    "filter": filter_query,
                }
            }
        ]
        results = await chunks_col.aggregate(pipeline).to_list(length=limit)
        if results:
            logger.info(f"Atlas $vectorSearch returned {len(results)} chunks")
            return results
    except Exception as e:
        logger.warning(f"Atlas $vectorSearch not available or failed: {e}. Falling back to in-memory cosine ranking.")

    # 2. Fallback: Fetch user's candidate chunks and rank by cosine similarity
    try:
        candidates = await chunks_col.find(filter_query).to_list(length=300)
        if not candidates:
            return []

        scored_chunks = []
        for doc in candidates:
            emb = doc.get("embedding")
            if emb:
                score = cosine_similarity(query_vector, emb)
                scored_chunks.append((score, doc))

        # Sort descending by similarity score
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [doc for _, doc in scored_chunks[:limit]]
        logger.info(f"Cosine similarity fallback selected {len(top_chunks)} chunks")
        return top_chunks
    except Exception as fallback_err:
        logger.error(f"Fallback chunk search error: {fallback_err}")
        return []


async def process_rag_query(
    question: str,
    user_id: str,
    content_id: str | None = None,
) -> dict:
    """
    Complete RAG Query Pipeline:
    1. Check Redis cache using cache:{userId}:{hash(question+contentId)}
    2. On miss: Embed question & retrieve matching chunks
    3. Build context & generate grounded answer via Gemini
    4. Cache result with 24h TTL
    5. Return { answer, sources }
    """
    # 1. Compute Cache Key
    target_scope = content_id if content_id else "all_docs"
    q_hash = hashlib.sha256(f"{question.strip()}:{target_scope}".encode()).hexdigest()
    cache_key = f"cache:{user_id}:{q_hash}"

    # Check cache
    cached = await get_cached_response(cache_key)
    if cached:
        return cached

    logger.info(f"Cache miss. Processing RAG query for user: {user_id}, contentId: {content_id}")

    # 2. Embed question & retrieve chunks
    query_vector = generate_query_embedding(question)
    chunks = await search_chunks(
        user_id=user_id,
        query_vector=query_vector,
        content_id=content_id,
        limit=5,
    )

    # 3. If no chunks found, return empty context response
    if not chunks:
        result = {
            "answer": "I don't have any relevant document context to answer this question. Please upload your documents first.",
            "sources": [],
        }
        await set_cached_response(cache_key, result, ttl_seconds=86400)
        return result

    # 4. Fetch document titles for source attribution
    contents_col = get_contents_collection()
    content_oids = list({c["contentId"] for c in chunks if "contentId" in c})
    content_docs = await contents_col.find({"_id": {"$in": content_oids}}).to_list(length=100)
    title_map = {str(d["_id"]): d.get("title", "Document") for d in content_docs}

    # 5. Prepare context and source citations
    context_blocks = []
    sources = []
    seen_sources = set()

    for c in chunks:
        cid_str = str(c.get("contentId"))
        title = title_map.get(cid_str, "Document")
        page_num = c.get("pageNumber")
        
        page_label = f", Page {page_num}" if page_num else ""
        context_blocks.append(f"[Document: '{title}'{page_label}]:\n{c.get('text', '')}")

        source_key = (cid_str, page_num)
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append(
                SourceItem(
                    contentId=cid_str,
                    title=title,
                    pageNumber=page_num,
                )
            )

    context_text = "\n\n---\n\n".join(context_blocks)

    # 6. Generate grounded answer via Gemini
    prompt = f"""You are a helpful, precise document assistant for Recall.
Answer the user's question using ONLY the provided context below.
If the answer cannot be found in or deduced from the context, state: "I don't know based on the provided documents."
Do not make up facts or extrapolate beyond what is explicitly stated in the context.

Context:
{context_text}

Question: {question}

Answer:"""

    try:
        model = genai.GenerativeModel("models/gemini-2.5-flash")
        response = model.generate_content(prompt)
        answer = response.text.strip()
    except Exception as e:
        logger.error(f"Gemini generation error: {e}")
        # Fallback to flash-latest if needed
        model = genai.GenerativeModel("models/gemini-flash-latest")
        response = model.generate_content(prompt)
        answer = response.text.strip()

    # 7. Format output & save in Redis cache
    result_data = {
        "answer": answer,
        "sources": [s.model_dump() for s in sources],
    }

    await set_cached_response(cache_key, result_data, ttl_seconds=86400)
    return result_data
