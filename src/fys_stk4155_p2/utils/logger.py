import logging
import sys

# [Timestamp] | File Name | Severity Level | Message
LOG_FORMAT = "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"

logging.basicConfig(
    level=logging.INFO,  # Capture INFO, WARNING, ERROR, and CRITICAL
    format=LOG_FORMAT,
    handlers=[
        logging.StreamHandler(sys.stdout)  # Explicitly direct all logs to the terminal console
    ],
)


def get_logger(module_name: str) -> logging.Logger:
    """
    Returns a configured logger instance named after the importing module.
    """
    return logging.getLogger(module_name)
