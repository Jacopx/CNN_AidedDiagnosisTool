# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                          CNN_AidedDiagnosisTool                         *
# *            https://github.com/Jacopx/CNN_AidedDiagnosisTool             *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

import logging
import coloredlogs


logger = None


def get_logger():
    global logger
    if logger is None:
        coloredlogs.install(level="DEBUG")
        logger = logging.getLogger()


def print_debug(message):
    global logger
    get_logger()
    logger.debug(message)


def print_info(message):
    global logger
    get_logger()
    logger.info(message)


def print_error(message):
    global logger
    get_logger()
    logger.error(message)
