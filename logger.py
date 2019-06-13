# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                          CNN_AidedDiagnosisTool                         *
# *            https://github.com/Jacopx/CNN_AidedDiagnosisTool             *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                 logger.py : tool for logging mechanism                  *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

import logging
import coloredlogs
import arguments


logger = None
verbose = arguments.get_verbose()


def get_logger():
    global logger, verbose
    if verbose is not False:
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
