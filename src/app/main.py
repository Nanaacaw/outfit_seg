from fastapi import FastAPI

# Import router dari app.api
from src.app.api.outfit_seg import router as outfit_seg

app = FastAPI(
    title="Outfit Detection API",
    description="API to detect and segment outfit images",
    version="1.0"
)

# Register router
app.include_router(outfit_seg)
