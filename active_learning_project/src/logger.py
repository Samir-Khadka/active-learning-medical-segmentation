import logging
import os
from datetime import datetime

def setup_logging(output_dir: str = "results/logs") -> logging.Logger:
    """
    Initializes a professional logging system with both file and console output.
    Safe to call multiple times — duplicate handlers are not added.
    """
    logger = logging.getLogger("ActiveLearning")

    # Guard: if handlers already attached, return the existing logger
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # Prevent double-logging via root logger

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"experiment_{timestamp}.log")

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    )

    # Console Handler — INFO and above only
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # File Handler — full DEBUG trace
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    logger.info(f"Logging initialised. Log file: {log_file}")
    return logger
