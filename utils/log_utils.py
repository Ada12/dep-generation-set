import logging

import constants

formatter = '%(asctime)s [%(levelname)s] %(message)s'

logging.basicConfig(level=logging.INFO,
                    format=formatter)

file_handler = logging.FileHandler(filename=constants.LOGGING_FILENAME, encoding='utf-8')
file_handler.setFormatter(logging.Formatter(formatter))
logging.getLogger().addHandler(file_handler)

LOGGER = logging
