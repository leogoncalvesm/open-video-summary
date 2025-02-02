import logging


class Logger:
    def __init__(self, log_file="app.log"):
        # Create a logger instance
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Create a formatter
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(log_format)

        # Create a console handler for logging to the console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        # Create a file handler for logging to a file
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        # Add handlers to the logger
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def debug(self, message):
        self.logger.debug(message)

    def info(self, message):
        self.logger.info(message)

    def warning(self, message):
        self.logger.warning(message)

    def error(self, message):
        self.logger.error(message)

    def critical(self, message):
        self.logger.critical(message)
