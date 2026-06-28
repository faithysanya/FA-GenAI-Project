"""Retry logic and error recovery utilities."""
import time
import logging
from functools import wraps
from typing import Callable, Type

logger = logging.getLogger(__name__)


def retry(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
):
    """Decorator for exponential backoff retry logic."""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            wait = delay_seconds
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt == max_attempts:
                        logger.error(f"{func.__name__} failed after {max_attempts} attempts: {e}")
                        raise
                    logger.warning(f"{func.__name__} attempt {attempt} failed: {e}. Retrying in {wait:.1f}s")
                    time.sleep(wait)
                    wait *= backoff
        return wrapper
    return decorator
