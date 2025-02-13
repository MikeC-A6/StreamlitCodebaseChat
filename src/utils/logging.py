import logging

def setup_logger(name: str) -> logging.Logger:
    """Configure and return a logger instance."""
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger(name)
