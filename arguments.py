# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                          CNN_AidedDiagnosisTool                         *
# *            https://github.com/Jacopx/CNN_AidedDiagnosisTool             *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *         arguments.py : utils for command line arguments parsing         *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *

import argparse

parser = argparse.ArgumentParser(description="A CNN implementation for Aided Diagnosis")
#parser.add_argument('dataset_folder', action="store")
parser.add_argument('-v', action='store_true', default=False,
                    dest='verbose',
                    help='Set verbose to true')
results = parser.parse_args()


# Return the parsed dataset folder
"""def get_dataset_folder():
    return results.dataset_folder"""


# Return the verbose option value
def get_verbose():
    return results.verbose
