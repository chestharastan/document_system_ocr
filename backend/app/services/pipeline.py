from pathlib import Path
import cv2
from PIL import Image

from app.services.model_pool import ModelPool
from app.core.utils import merge_and_filter_boxes, dedupe_lines, overlap_ratio
from app.schemas.common import Block, Line, Box

class PipelineService:
    """Encapsulates your entire pipeline logic. Stateless per-request."""
    
    def __init__(self, pool: ModelPool):
        self.pool = pool
        self.LAYOUT_PAD = 8
        self.LABEL_TO_TYPE = {...}  # your existing maps
        self.TEXT_TYPES = {...}
        self.RECOGNIZE_TYPES = {...}
    
    def process_image(self, image_path: Path, recognize: bool = True) -> dict:
        """Your existing process_image(), but using self.pool.layout, etc."""
        page_bgr = cv2.imread(str(image_path))
        page_h, page_w = page_bgr.shape[:2]
        
        # Stage 1: Layout
        blocks = self._layout_blocks(image_path)
        
        # Stage 2: Detection (page-wide)
        page_lines = self._page_lines(image_path) if blocks else []
        blocks = self._recover_orphan_lines(blocks, page_lines, page_w, page_h)
        
        # Stage 3: Recognition
        recog = self.pool.recognizer if recognize else None
        out_blocks = self._build_blocks(blocks, page_bgr, page_lines, ...)
        
        return {"width": page_w, "height": page_h, "blocks": out_blocks}
    
    def process_regions(self, regions: list, ...) -> dict:
        """Your existing process_regions()."""
        ...
    
    def process_lines(self, blocks_in: list, ...) -> dict:
        """Your existing process_lines()."""
        ...
    
    # --- Private: delegate to specific services or keep inline ---
    def