# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                          CNN_AidedDiagnosisTool                         *
# *            https://github.com/Jacopx/CNN_AidedDiagnosisTool             *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

import logging
import coloredlogs
import sys


logger = None
verbose = None


def get_logger():
    global logger, verbose
    try:
        verbose = sys.argv[2]
    except Exception:
        verbose = None
    if verbose is not None:
        if logger is None:
            coloredlogs.install(level="DEBUG")
            logger = logging.getLogger()
    else:
        logger = None


def print_debug(message):
    global logger
    get_logger()
    if logger is not None:
        logger.debug(message)


def print_info(message):
    global logger
    get_logger()
    if logger is not None:
        logger.info(message)


def print_error(message):
    global logger
    get_logger()
    if logger is not None:
        logger.error(message)
