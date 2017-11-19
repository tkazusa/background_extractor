# -*- coding: utf-8 -*-
#
# Author: taketoshi.kazusa
#
import os

import numpy as np
import cv2

from runner import RunnerBase
from extracter_median import BackGroundExtracterMedian
from util import Util

PATH_TO_DATA_DIR = os.path.join("/root", "data", "CDnet2014_DataSet_Subset", "baseline", "highway", "input")
PATH_TO_OBS_IMAGE = os.path.join("/root", "data", "CDnet2014_DataSet_Subset", "baseline", "highway", "input",
                                 "in000375.jpg")


class RunnerBackGroundExtracterMedian(RunnerBase):
    """Runner of eigen background method

    Args:
        run_name (str): Run name for image saving.
        n_images (int): Number of images for training.
        time (str): Timestamp for image saving.
        save_filename (str): Filename for image saving.

    Methods:
        run_train: Run train background extracter with specified images.
        run_extraction: Run extract_background with specified images.
    """

    def __init__(self, run_name, n_images):
        super(RunnerBase, self).__init__()
        self.run_name = run_name
        self.n_images = n_images
        self.time = Util.nowstr()
        self.save_filename = self.time + "_" + run_name + "_" + str(n_images) + ".jpg"

    def _set_extracter(self, height, width):
        """Set extracter.
        """
        return BackGroundExtracterMedian(height, width)

    def _fetch_images_for_backgroundmodel(self, dr):
        #TODO: 画像データどう読み込んでくる？

        """Fetch images for background model tranning.

        Args:
            dr (str): Directory path of train data.
        """
        for i in range(self.n_images):
            filename = all_images[i]
            img = Util.read_image_gray(os.path.join(dr, filename))
            self.height, self.width = img.shape[0], img.shape[1]
            img_vector = img.reshape(1, self.height * self.width)

            if i == 0:
                self.imgs = img_vector
            else:
                self.imgs = np.concatenate((self.imgs, img_vector), axis=0)

    def run_train(self):
        """Run train background extracter with specified images.
        """
        self._fetch_images_for_backgroundmodel(PATH_TO_DATA_DIR)
        self.model = self._set_extracter(self.height, self.width)
        self.model.train(self.imgs)

    def _fetch_observed_image(self, filepath):
        self.observed_image = Util.read_image_gray(filepath)

    def run_extraction(self):
        """Run extract_background with specified images.
        """
        self._fetch_observed_image(PATH_TO_OBS_IMAGE)
        background_image = self.model.extract_background(self.observed_image)
        Util.save_image(self.save_filename, background_image)


if __name__ == "__main__":
    all_images = os.listdir(PATH_TO_DATA_DIR)
    runner = RunnerBackGroundExtracterMedian(run_name="median", n_images=5)
    runner.run_train()
    runner.run_extraction()

    runner = RunnerBackGroundExtracterMedian(run_name="median", n_images=10)
    runner.run_train()
    runner.run_extraction()

    runner = RunnerBackGroundExtracterMedian(run_name="median", n_images=15)
    runner.run_train()
    runner.run_extraction()
