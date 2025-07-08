import os
from src.app.utils import suppress_tensorflow_warnings

# Suppress TensorFlow warnings early
suppress_tensorflow_warnings()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import logging
# Import setting to control debug mode
from src.app.settings.setting import DEBUG_MODE

# Configure logging based on debug mode
if DEBUG_MODE:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
else:
    # Set logging to WARNING level to hide INFO logs when debug is disabled
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

# Import router from app.api
from src.app.api.full_detection_api import router as full_detection_api

app = FastAPI(
    title="Outfit Detection API",
    description="API to detect and segment outfit images",
    version="1.0"
)

# Register router
app.include_router(full_detection_api)

# Mount static files
app.mount("/results", StaticFiles(directory="results", html=True), name="results")