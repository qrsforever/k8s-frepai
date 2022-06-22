#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file registry.py
# @brief
# @author QRS
# @blog qrsforever.gitee.io
# @version 1.0
# @date 2022-03-17 19:35


from abc import ABC, abstractmethod
from collections import defaultdict


class Registrable(ABC):
    _registry = defaultdict(dict)

    @abstractmethod
    def register(*arg, **kwargs):
        """
        """

    def get_runner(name):
        for _, register in Registrable._registry.items():
            if name in register:
                return register[name][0]
        raise LookupError(name)

    def get_module(name):
        for _, register in Registrable._registry.items():
            if name in register:
                return register[name][1]
        raise LookupError(name)


class FunctionRegister(Registrable):
    def register(name):
        registry = Registrable._registry['funcs']

        def _add_func(func):
            if name not in registry:
                registry[name] = (func, (func.__module__, func.__name__))

        return _add_func


class ClassRegister(Registrable):
    def register(name):
        registry = Registrable._registry['clses']

        def _add_class(cls):
            if name not in registry:
                registry[name] = (cls, (cls.__module__, cls.__name__))

        return _add_class
