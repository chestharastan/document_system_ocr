from pydantic import BaseModel
from app.schemas.common import Block, Box

# ---------- /predict ----------
# Request: just multipart file upload (no body schema needed)
# Response:
class ProcessImageResponse(BaseModel):
    width: int
    height: int
    blocks: list[Block]

# ---------- /regions ----------
class RegionInput(BaseModel):
    type: str  # "title" | "paragraph" | "header" | "image" | "caption"
    box: Box

class ProcessRegionsRequest(BaseModel):
    page_id: str
    regions: list[RegionInput]
    recognize: bool = True

class ProcessRegionsResponse(BaseModel):
    width: int
    height: int
    blocks: list[Block]

# ---------- /lines ----------
class LineInput(BaseModel):
    x: float
    y: float
    width: float
    height: float

class BlockInput(BaseModel):
    type: str
    box: Box
    lines: list[LineInput] = []

class ProcessLinesRequest(BaseModel):
    page_id: str
    blocks: list[BlockInput]
    recognize: bool = True

class ProcessLinesResponse(BaseModel):
    width: int
    height: int
    blocks: list[Block]
