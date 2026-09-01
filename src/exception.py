"""Custom exception that keeps the file and line number of the original error.

Wrap anything that touches disk, unpickles a model or runs a fit so that a
failure deep inside a component still reports where it actually happened::

    try:
        ...
    except Exception as e:
        raise CustomException(e, sys)
"""

import sys


def error_message_detail(error, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is None:
        return f'Error occurred: {error}'
    file_name = exc_tb.tb_frame.f_code.co_filename
    return (
        f'Error occurred in python script [{file_name}] '
        f'line [{exc_tb.tb_lineno}] message [{error}]'
    )


class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(str(error_message))
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message
