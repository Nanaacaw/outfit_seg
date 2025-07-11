from pydantic import BaseModel,HttpUrl
from typing import List, Optional

# schema for image url
class LabelRequest(BaseModel):
    image_url: HttpUrl
    labels: Optional[List[str]] = None # default is None
    threshold: Optional[float] = None # default is None
    polygon_refinement: Optional[bool] = False # default is False
