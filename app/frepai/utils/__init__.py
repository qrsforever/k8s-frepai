#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file __init__.py
# @brief
# @author QRS
# @version 1.0
# @date 2022-03-17 16:43

import os
import errno
import threading
import subprocess
import ssl
import shutil
from urllib import request, parse

ssl._create_default_https_context = ssl._create_unverified_context


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def rmdir_p(path):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


class SingletonType(type):
    _instance_lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            with SingletonType._instance_lock:
                if not hasattr(cls, "_instance"):
                    cls._instance = super(SingletonType,cls).__call__(*args, **kwargs)
        return cls._instance


def easy_wget(url, path='/tmp'):
    url = parse.quote(url, safe=':/?-=')
    if os.path.isdir(path):
        path = os.path.join(path, os.path.basename(url))
    res = request.urlretrieve(url, path)
    return res[0]


def run_shell(cmd):
    try:
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT)
    except Exception as err:
        output = err.output
    return output.decode()
