import logging
import os
import sys
import time


project_name = os.getcwd().split('/')[-1]
_logger = logging.getLogger(project_name)
_logger.addHandler(logging.StreamHandler())


def _log_prefix():
    # Returns (filename, line number) for the first non-logger stack frame.
    def _get_file_line():
        f = sys._getframe()
        our_file = f.f_code.co_filename
        f = f.f_back
        while f:
            code = f.f_code
            if code.co_filename != our_file:
                return code.co_filename, f.f_lineno
            f = f.f_back
        return '<unknown>', 0

    # current time
    now = time.time()
    now_tuple = time.localtime(now)
    now_millisecond = int(1e3 * (now % 1.0))

    # current filename and line
    filename, line = _get_file_line()
    basename = os.path.basename(filename)

    s = '%02d-%02d %02d:%02d:%02d.%03d %s:%d] ' % (
        now_tuple[1],  # month
        now_tuple[2],  # day
        now_tuple[3],  # hour
        now_tuple[4],  # min
        now_tuple[5],  # sec
        now_millisecond,
        basename,
        line)

    return s


def logging_verbosity(verbosity=0):
    _logger.setLevel(verbosity)


def info(msg, *args, **kwargs):
    _logger.info('I ' + project_name + ' ' + _log_prefix() + msg, *args, **kwargs)
