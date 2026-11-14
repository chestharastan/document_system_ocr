from pydantic import BaseModel, Field

# Request boxes must be finite numbers (JSON Infinity / NaN -> 422); they are clipped to the page by the pipeline.
_finite = dict(allow_inf_nan=False)


class Box(BaseModel):
    x: float = Field(**_finite)
    y: float = Field(**_finite)
    width: float = Field(ge=0, **_finite)
    height: float = Field(ge=0, **_finite)


class Line(BaseModel):
    box: Box
    text: str = ""
    reading_order: int = 0
    indent: int = 0
    tab: bool = False


class Block(BaseModel):
    id: str
    type: str  # v1 UI type (title/paragraph/header/caption/image) or a v2 class name (app/taxonomy.py)
    label: str
    reading_order: int
    box: Box
    lines: list[Line] = []
    crop: dict | None = None  # {"url": "...", "file": "..."}
    text: str = ""
