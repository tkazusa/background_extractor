# -*- coding: utf-8 -*-
#
# Author: taketoshi.kazusa
#
from abc import ABCMeta, abstractmethod


class RunnerBase(object):
    """Runner base
    Provides interface for runner class, which run background extracters.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        raise NotImplementedError

    @abstractmethod
    def _set_extracter(self):
        """ :rtype: BackGroundExtracterBase """
        raise NotImplementedError

    @abstractmethod
    def _fetch_images_for_backgroundmodel(self):
        raise NotImplementedError

    @abstractmethod
    def run_train(self):
        raise NotImplementedError

    @abstractmethod
    def _fetch_observed_image(self):
        raise NotImplementedError

    @abstractmethod
    def run_extraction(self):
        raise NotImplementedError