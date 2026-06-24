from pydantic import BaseModel, Field

class Box(BaseModel):
    x: float
    y: float
    width: float
    height: float

class Line(BaseModel):
    box: Box
    text: str = ""
    reading_order: int = 0
    indent: int = 0
    tab: bool = False

class Block(BaseModel):
    id: str
    type: str  # "title" | "paragraph" | "header" | "image" | "caption"
    label: str
    reading_order: int
    box: Box
    lines: list[Line] = []
    crop: dict | None = None  # {"url": "...", "file": "..."}
    text: str = ""