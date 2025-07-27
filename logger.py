import logging
import sys

# Create a logger
logger = logging.getLogger("SuvidhaChatbot")
logger.setLevel(logging.DEBUG)  # Set the lowest level to capture all messages

# Create a formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create a handler for stdout
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.INFO)  # Log INFO and above to the console
stdout_handler.setFormatter(formatter)

# Create a handler for a log file
file_handler = logging.FileHandler('app.log')
file_handler.setLevel(logging.DEBUG)  # Log DEBUG and above to the file
file_handler.setFormatter(formatter)

# Add the handlers to the logger
logger.addHandler(stdout_handler)
logger.addHandler(file_handler)
