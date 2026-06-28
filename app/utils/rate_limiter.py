"""Simple in-memory rate limiter."""
import time
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

# Limit: 60 requests per minute per IP
RATE_LIMIT = 60
WINDOW_SECONDS = 60

_request_log: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(client_ip: str) -> bool:
    """
    Returns True if the request is allowed, False if rate limited.
    Slides a 60-second window per IP.
    """
    now = time.time()
    window_start = now - WINDOW_SECONDS
    _request_log[client_ip] = [t for t in _request_log[client_ip] if t > window_start]

    if len(_request_log[client_ip]) >= RATE_LIMIT:
        logger.warning(f"Rate limit exceeded for {client_ip}")
        return False

    _request_log[client_ip].append(now)
    return True
