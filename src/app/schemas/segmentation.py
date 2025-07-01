from pydantic import BaseModel
from typing import List, Optional

class LabelRequest(BaseModel):
    image_url: str
    labels: List[str]
    threshold: Optional[float] = None
    polygon_refinement: Optional[bool] = False
    