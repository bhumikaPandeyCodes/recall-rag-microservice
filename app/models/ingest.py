from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    contentId: str = Field(..., description="MongoDB ObjectID of the Content document")
    userId: str = Field(..., description="MongoDB ObjectID of the User")
    fileUrl: str = Field(..., description="Public Cloudinary URL of the uploaded document")


class IngestResponse(BaseModel):
    success: bool
    message: str
    contentId: str
    totalChunks: int = 0
    status: str
