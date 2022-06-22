#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file errcodes.py
# @brief
# @author QRS
# @version 1.0
# @date 2022-03-15 19:22


import functools
import traceback
import json


def errmsg(code, err):
    msg = {
        'errno': code,
        'result': {
            'errtext': str(err),
            'traceback': traceback.format_exc(limit=10)
        }
    }
    return msg


class HandlerOk(object):
    def __init__(self, message=''):
        self.message = message

    def __str__(self):
        return json.dumps({'errno': 0, 'message': self.message})


class HandlerError(Exception):
    def __init__(self, code, message):
        self.code = code
        self.message = message
        super().__init__(self.message)

    def __str__(self):
        return f'[{self.code}] {self.message}'


def catch_error(func):
    @functools.wraps(func)
    def decorator(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except HandlerError as err0:
            return json.dumps(errmsg(err0.code, err0))
        except json.decoder.JSONDecodeError as err1:
            return json.dumps(errmsg(90001, err1))
        except ImportError as err2:
            return json.dumps(errmsg(90002, err2))
        except KeyError as err3:
            return json.dumps(errmsg(90003, err3))
        except ValueError as err4:
            return json.dumps(errmsg(90004, err4))
        except AssertionError as err5:
            return json.dumps(errmsg(90005, err5))
        except AttributeError as err6:
            return json.dumps(errmsg(90006, err6))
        except Exception as err99:
            return json.dumps(errmsg(90099, err99))
    return decorator
