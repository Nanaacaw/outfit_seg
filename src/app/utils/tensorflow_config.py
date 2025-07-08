import os
import logging
from typing import Optional

def suppress_tensorflow_warnings():
    """
    Suppress TensorFlow warnings and oneDNN messages.
    Call this function early in your application startup.
    """
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Optional: suppress CUDA warnings

def configure_tensorflow_logging(verbose: bool = False):
    """
    Configure TensorFlow logging level.
    
    Args:
        verbose: If True, show TensorFlow logs. If False, suppress them.
    """
    if not verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

def get_tensorflow_logger(name: str = "tensorflow") -> logging.Logger:
    """
    Get a logger specifically for TensorFlow operations.
    
    Args:
        name: Logger name
        
    Returns:
        Configured TensorFlow logger
    """
    logger = logging.getLogger(name)
    return logger 