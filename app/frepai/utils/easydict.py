#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file easydict.py
# @brief
# @author QRS
# @version 1.0
# @date 2022-03-16 20:38


from collections.abc import Iterable, Mapping
from copy import deepcopy


class DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, *args, **kwargs):
        for arg in args:
            if isinstance(arg, dict):
                for k, v in arg.items():
                    self[k] = v
        if kwargs:
            for k, v in kwargs.items():
                self[k] = v

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)


class MeldDict(dict):
    meld_iters = False
    remove_emptied = False

    def add(self, other):
        if not isinstance(other, Mapping):
            raise TypeError('can only add Mapping '
                            '(not "{}")'.format(type(other).__name__))
        for key, that in other.items():
            if key not in self:
                self[key] = that
                continue
            this = self[key]
            if isinstance(this, Mapping) and isinstance(that, Mapping):
                self[key] = MeldDict(this).add(that)
            elif isinstance(this, Iterable) and isinstance(that, Iterable) and \
                  not (isinstance(this, str) or isinstance(that, str)) and \
                  self.meld_iters:
                self[key] = list(this) + list(that)
            else:
                self[key] = that
        return self

    def subtract(self, other):
        if not isinstance(other, Mapping):
            raise TypeError('can only subtract Mapping '
                            '(not "{}")'.format(type(other).__name__))
        to_remove = []
        for key, this in self.items():
            if key not in other:
                continue
            that = other[key]
            if isinstance(this, Mapping) and isinstance(that, Mapping):
                self[key] = MeldDict(this).subtract(that)
                if not self[key] and self.remove_emptied:
                    to_remove.append(key)
            elif isinstance(this, Iterable) and isinstance(that, Iterable) and \
                  not (isinstance(this, str) or isinstance(that, str)) and \
                  self.meld_iters:
                self[key] = [item for item in this if item not in that]
                if not self[key] and self.remove_emptied:
                    to_remove.append(key)
            else:
                to_remove.append(key)
        for key in to_remove:
            del self[key]
        return self

    def __add__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        return MeldDict(deepcopy(self)).add(other)

    def __radd__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        return MeldDict(deepcopy(other)).add(self)

    def __iadd__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        return self.add(other)

    def __sub__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        return MeldDict(deepcopy(self)).subtract(other)

    def __rsub__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        return MeldDict(deepcopy(other)).subtract(self)

    def __isub__(self, other):
        if not isinstance(other, Mapping):
            return NotImplemented
        return self.subtract(other)
