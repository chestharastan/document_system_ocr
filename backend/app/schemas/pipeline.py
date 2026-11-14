from pydantic import BaseModel, Field
from app.schemas.common import Block, Box

# uuid4().hex as issued by /process; anything else (paths, globs, non-strings) -> 422
def PageId():  # noqa: N802
    return Field(pattern=r"^[0-9a-f]{32}$", max_length=32)

# ---------- /predict ----------
# Request: just multipart file upload (no body schema needed)
# Response:
class ProcessImageResponse(BaseModel):
    width: int
    height: int
    blocks: list[Block]

# ---------- /regions ----------
class RegionInput(BaseModel):
    type: str  # v1 UI type (title/paragraph/header/caption/image) or a v2 class name (app/taxonomy.py)
    box: Box

class ReprocessRequest(BaseModel):
    page_id: str = PageId()


class ProcessRegionsRequest(BaseModel):
    page_id: str = PageId()
    regions: list[RegionInput]
    recognize: bool = True

class ProcessRegionsResponse(BaseModel):
    width: int
    height: int
    blocks: list[Block]

# ---------- /lines ----------
class LineInput(BaseModel):
    x: float = Field(allow_inf_nan=False)
    y: float = Field(allow_inf_nan=False)
    width: float = Field(ge=0, allow_inf_nan=False)
    height: float = Field(ge=0, allow_inf_nan=False)

class BlockInput(BaseModel):
    type: str
    box: Box
    lines: list[LineInput] = []

class ProcessLinesRequest(BaseModel):
    page_id: str = PageId()
    blocks: list[BlockInput]
    recognize: bool = True

class ProcessLinesResponse(BaseModel):
    width: int
    height: int
    blocks: list[Block]
