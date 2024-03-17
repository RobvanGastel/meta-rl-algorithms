import os
import sys
import logging


def configure_logger(
    name,
    log_dir,
    log_level=logging.DEBUG,
    log_format="%(asctime)s | %(levelname)-8s | %(message)s",
    date_format="%Y-%m-%dT%T%Z",
):
    # Setup logging instance and logging directory
    os.makedirs(os.path.join(log_dir), exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format=log_format,
        datefmt=date_format,
        stream=sys.stdout,
    )

    # File handler configuration
    log_file_path = os.path.join(log_dir, f"{name}.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setFormatter(logging.Formatter(log_format, date_format))

    # Add file handler to the root logger
    logging.getLogger().addHandler(file_handler)
