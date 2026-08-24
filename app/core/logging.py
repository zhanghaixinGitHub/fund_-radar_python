"""Central logging helpers that avoid exposing secrets."""

import logging


def configure_logging(log_level: str) -> None:
    """Configure process logging once using the project log prefix convention."""
    logging.basicConfig(
        level=log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger for a module without adding duplicate handlers."""
    return logging.getLogger(name)
