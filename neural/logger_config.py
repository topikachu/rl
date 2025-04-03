import logging
import os

def setup_logger(log_file='robocode_ai.log'):
    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(log_dir, log_file))
    fh.setLevel(logging.DEBUG)

    # Create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    # Add the handlers to the root logger
    root_logger.addHandler(fh)
    root_logger.addHandler(ch)

def get_logger(name):
    return logging.getLogger(name)