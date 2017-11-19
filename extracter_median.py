# -*- coding: utf-8 -*-
#
# Author: taketoshi.kazusa
#
import numpy as np

from backgroundextracter import BackGroundExtracterBase


class BackGroundExtracterMedian(BackGroundExtracterBase):
    """Background extracter with eigen method.
    Args:
        height (int): Height of image.
        width (int): Width of image.

    Methods:
        train: Train set algorithm and generate a background model.
        save_background_model: Save a background model.
        load_background_model: Load a background model.
        extract_background: Extract a background image from an observed image.
    """

    def __init__(self, height, width):
        super(BackGroundExtracterBase, self).__init__()
        self.height = height
        self.width = width


    def _set_algorithm(self):
        """Set algorithm
        """
        pass

    def train(self, images):
        """Train and generate background model.

           Args:
            images (ndarray): Extracter input.
                This is images shape (n_images, height*width).
        """
        self.images_for_background = images

    def save_background_model(self):
        pass

    def load_background_model(self):
        pass

    def extract_background(self, observed_image):
        """Extract background from new observed image.
           Args:
            observed_image: np.array data of new observed image.

           Return:
            Background image of observed image removed foreground.
        """
        v_observed_image = observed_image.reshape(1,self.height*self.width)
        images_array = np.concatenate([self.images_for_background, v_observed_image], axis=0)
        background_image = np.median(images_array, axis=0).reshape(self.height, self.width)

        return background_image
