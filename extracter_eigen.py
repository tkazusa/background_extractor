# -*- coding: utf-8 -*-
#
# Author: taketoshi.kazusa
#
from sklearn.decomposition import PCA

from backgroundextracter import BackGroundExtracterBase


class BackGroundExtracterEigen(BackGroundExtracterBase):
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
        """Set algorithm.
        """
        return PCA()

    def train(self, images):
        """Train and generate background model.

           Args:
            images (ndarray): Extracter input.
                This is images shape (n_images, height*width).
        """
        algo = self._set_algorithm()
        algo.fit(images)
        self.E = algo.components_[:1].T
        self.x_avg = algo.mean_.reshape(self.height * self.width, 1)

    def save_background_model(self):
        pass

    def load_background_model(self):
        pass

    def extract_background(self, observed_image):
        """Extract background from new observed image.
           Args:
            observed_image: ndarray data of new observed image.

           Return:
            Background image of observed image removed foreground.
        """
        x_t = observed_image.reshape(self.height * self.width, 1)
        tmp = self.E.T.dot(x_t - self.x_avg)
        xb_t = self.E.dot(tmp) + self.x_avg
        background_image = xb_t.reshape(self.height, self.width)

        return background_image
