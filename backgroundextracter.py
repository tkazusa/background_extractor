# -*- coding: utf-8 -*-
#
# Author: taketoshi.kazusa
#
from abc import ABCMeta, abstractmethod


class BackGroundExtracterBase(object):
    """Back Ground Extracter interface

    Methods:
        train:Trains and generates background model.
        save_background_model:Saves background model generated by train method to specified file path.
        load_background_model:Loads background model from specified file path.
        extract_background:Extracts background with background model generated by train method.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def _set_algorithm(self):
        raise NotImplementedError

    @abstractmethod
    def train(self):
        raise NotImplementedError

    @abstractmethod
    def save_background_model(self):
        raise NotImplementedError

    @abstractmethod
    def load_background_model(self):
        raise NotImplementedError

    @abstractmethod
    def extract_background(self):
        raise NotImplementedError
