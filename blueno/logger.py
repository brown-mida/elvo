"""
Functions to handle logging logic.
"""

import logging


def configure_parent_logger(file_name,
                            stdout=True,
                            level=logging.DEBUG):
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    handler = logging.FileHandler(file_name)
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    if stdout:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)


def configure_job_logger(file_path, level=logging.DEBUG):
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    handler = logging.FileHandler(file_path)
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)
