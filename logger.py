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
            #logging.basicConfig(level=logging.DEBUG)
            logging.getLogger('PIL.PngImagePlugin').setLevel(logging.CRITICAL)
            logger = logging.getLogger('CnnSoftware')
            coloredlogs.install(level='DEBUG')

            def print_debug(message):
                global logger
                get_logger()
                if logger is not None:
                    logger.debug(message)
            """ #JUST FOR TEST
            print_debug("message")
            print_info("message")
            print_error("message")
            print_warning("message") """
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


def print_warning(message):
    global logger
    get_logger()
    if logger is not None:
        logger.warning(message)