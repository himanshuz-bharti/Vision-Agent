import time
import logging
from contextlib import contextmanager

logger = logging.getLogger(__name__)

@contextmanager
def time_it(name: str):
    """Context manager to measure and log the execution time of a block of code."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = (time.perf_counter() - start) * 1000
        logger.debug(f"{name} took {elapsed:.2f} ms")
