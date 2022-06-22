#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file logger.py
# @brief
# @author QRS
# @version 1.0
# @date 2022-03-17 16:39

import sys
import os
import traceback
import logging
import queue
import threading
import multiprocessing
from contextlib import contextmanager

from . import SingletonType


class MultiProcessingLogHandler(logging.Handler, metaclass=SingletonType):
    def __init__(self, name, handlers=None):
        super(MultiProcessingLogHandler, self).__init__()
        if handlers is None or len(handlers) == 0:
            handlers = [logging.StreamHandler()]
        self.handlers = handlers
        self.queue = multiprocessing.Queue(-1)
        self._is_closed = False
        self._receive_thread = threading.Thread(target=self._receive, name=name)
        self._receive_thread.daemon = True
        self._receive_thread.start()

    def setLevel(self, level):
        super(MultiProcessingLogHandler, self).setLevel(level)
        for handler in self.handlers:
            handler.setLevel(level)

    def setFormatter(self, fmt):
        super(MultiProcessingLogHandler, self).setFormatter(fmt)
        for handler in self.handlers:
            handler.setFormatter(fmt)

    def _receive(self):
        while True:
            try:
                if self._is_closed and self.queue.empty():
                    break
                record = self.queue.get(timeout=0.3)
                for handler in self.handlers:
                    handler.emit(record)
            except (KeyboardInterrupt, SystemExit):
                raise
            except (BrokenPipeError, EOFError):
                break
            except queue.Empty:
                pass  # This periodically checks if the logger is closed.
            except Exception:
                traceback.print_exc(file=sys.stderr)

        self.queue.close()
        self.queue.join_thread()

    def _send(self, s):
        self.queue.put_nowait(s)

    def _format_record(self, record):
        if record.args:
            record.msg = record.msg % record.args
            record.args = None
        if record.exc_info:
            self.format(record)
            record.exc_info = None

        return record

    def emit(self, record):
        try:
            s = self._format_record(record)
            self._send(s)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)

    def close(self):
        if not self._is_closed:
            self._is_closed = True
            self._receive_thread.join(5.0)
            for handler in self.handlers:
                handler.close()
            super(MultiProcessingLogHandler, self).close()


def easy_get_logger(name, level=logging.DEBUG, filepath=None, backup_count=-1, console=True, mp=False):
    logger = logging.getLogger(name)
    logger.handlers.clear()
    if isinstance(level, str):
        if level in ('D', 'DEBUG', 'd', 'debug'):
            level = logging.DEBUG
        elif level in ('I', 'INFO', 'i', 'info'):
            level = logging.INFO
        elif level in ('E', 'ERROR', 'e', 'error'):
            level = logging.ERROR
        else:
            level = logging.WARNING
    logger.setLevel(level)
    handlers = []
    #  %(filename)s
    # formatter = logging.Formatter('%(asctime)s - %(funcName)s:%(lineno)d - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if console:
        handlers.append(logging.StreamHandler())
    if filepath:
        if backup_count > 0:
            filelog = logging.handlers.TimedRotatingFileHandler(
                filename=filepath,
                when='D',
                backupCount=backup_count,
                encoding='utf-8')
        else:
            filelog = logging.FileHandler(filepath)
        handlers.append(filelog)

    if mp: # multiprocessing
        handlers = [MultiProcessingLogHandler(name, handlers)]

    for handler in handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


class EasyLogger(object):
    logger = easy_get_logger('default')

    @staticmethod
    def setup_logger(name, level='debug', filepath=None, backup_count=-1, console=True, mp=False):
        EasyLogger.logger = easy_get_logger(
                name, level=level,
                filepath=filepath, backup_count=backup_count,
                console=console, mp=mp)
        return EasyLogger.logger

    @staticmethod
    def set(logger):
        EasyLogger.logger = logger

    @staticmethod
    def get():
        return EasyLogger.logger

    @staticmethod
    def prefix():
        frame = sys._getframe().f_back.f_back
        filename = os.path.basename(frame.f_code.co_filename)
        funcname = frame.f_code.co_name
        lineno = frame.f_lineno
        return '{} {}:{}'.format(filename, funcname, lineno)

    @staticmethod
    def debug(message):
        EasyLogger.logger.debug(f'{EasyLogger.prefix()} - {message}')

    @staticmethod
    def info(message):
        EasyLogger.logger.info(f'{EasyLogger.prefix()} - {message}')

    @staticmethod
    def warning(message):
        EasyLogger.logger.warning(f'{EasyLogger.prefix()} - {message}')

    @staticmethod
    def error(message):
        if isinstance(message, dict) and 'errtxt' in message:
            tb = None
            if 'traceback' in message['errtxt']:
                tb = message['errtxt'].pop('traceback')
            EasyLogger.logger.error(f'{EasyLogger.prefix()}:{message}')
            if tb:
                EasyLogger.logger.error(f'traceback: {tb}')
        else:
            EasyLogger.logger.error(f'{EasyLogger.prefix()}:{message}')


@contextmanager
def logger_subprocess(func, *args):
    logger = EasyLogger.get()

    def _target(logger, *args):
        from frepai.utils.logger import EasyLogger
        EasyLogger.set(logger)
        func(*args)

    yield multiprocessing.Process(target=_target, args=(logger, *args), daemon=True)
