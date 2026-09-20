from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Question asked by the user")
    contentId: str | None = Field(
        None,
        description="Optional contentId for single-document query. If omitted, searches across all documents of the user.",
    )


class SourceItem(BaseModel):
    contentId: str
    title: str
    pageNumber: int | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceItem] = Field(default_factory=list)
