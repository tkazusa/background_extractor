﻿# -*- coding: utf-8 -*-
#
# Author: taketoshi.kazusa
#
import os
import datetime

import cv2
import pandas as pd
from sklearn.externals import joblib


class Util:

    @classmethod
    def mkdir(cls, dr):
        if not os.path.exists(dr):
            os.makedirs(dr)

    @classmethod
    def mkdir_file(cls, path):
        dr = os.path.dirname(path)
        if not os.path.exists(dr):
            os.makedirs(dr)

    @classmethod
    def save_image(cls, filename, image):
        cv2.imwrite(filename, image)

    @classmethod
    def read_image(cls, filename):
        return cv2.imread(filename)

    @classmethod
    def read_image_gray(cls, filename):
        return cv2.imread(filename, 0)

    @classmethod
    def dump(cls, obj, filename, compress=0):
        cls.mkdir_file(filename)
        joblib.dump(obj, filename, compress=compress)

    @classmethod
    def dumpc(cls, obj, filename):
        cls.mkdir_file(filename)
        cls.dump(obj, filename, compress=3)

    @classmethod
    def load(cls, filename):
        return joblib.load(filename)

    @classmethod
    def read_csv(cls, filename, sep="\t"):
        return pd.read_csv(filename, sep=sep)

    @classmethod
    def to_csv(cls, _df, filename, index=False, sep="\t"):
        cls.mkdir_file(filename)
        _df.to_csv(filename, sep=sep, index=index)

    @classmethod
    def nowstr(cls):
        return str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"))

    @classmethod
    def nowstrhms(cls):
        return str(datetime.datetime.now().strftime("%H-%M-%S"))

