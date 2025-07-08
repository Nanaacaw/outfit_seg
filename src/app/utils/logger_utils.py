# app/utils/logger_utils.py
import logging
from typing import Optional
from src.app.settings.setting import DEBUG_MODE

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance that respects the DEBUG_MODE setting.
    
    Args:
        name: The name of the logger (usually __name__)
        
    Returns:
        A configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set the logger level based on DEBUG_MODE
    if DEBUG_MODE:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    
    return logger

def debug_log(message: str, logger: Optional[logging.Logger] = None):
    """
    Log a debug message only if DEBUG_MODE is True.
    
    Args:
        message: The message to log
        logger: Optional logger instance, if not provided will use root logger
    """
    if DEBUG_MODE:
        if logger is None:
            logger = logging.getLogger()
        logger.info(message)

def debug_print(message: str):
    """
    Print a debug message only if DEBUG_MODE is True.
    Useful for quick debugging without setting up logger.
    
    Args:
        message: The message to print
    """
    if DEBUG_MODE:
        print(f"[DEBUG] {message}")

def is_debug_mode() -> bool:
    """
    Check if debug mode is currently enabled.
    
    Returns:
        True if DEBUG_MODE is True, False otherwise
    """
    return DEBUG_MODE 