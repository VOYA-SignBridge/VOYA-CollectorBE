"""logging_config.py
Central logging configuration honoring settings.debug_logging.
"""
import logging
from app.config import settings

def configure_logging():
    level = logging.DEBUG if settings.debug_logging else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if settings.debug_logging:
        logging.getLogger(__name__).debug("Debug logging enabled")
