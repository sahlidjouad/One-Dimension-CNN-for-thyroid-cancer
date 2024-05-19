import h5py
import matplotlib.pyplot as plt
import numpy as np
import skimage.color
from radiomics import featureextractor, getTestCase, imageoperations
import SimpleITK as sitk
import six
import os
import logging
import radiomics


class RadiomicsExtractor:

    def __init__(self):

        logger = radiomics.logger
        logger.setLevel(logging.DEBUG)

        # Write out all log entries to a file
        handler = logging.FileHandler(
            filename='./log/radiomicsLog.txt', mode='w')
        formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        self.paramsFile = os.path.abspath(
            os.path.join('./settings', 'radiomics.yaml'))

        self.extractor = featureextractor.RadiomicsFeatureExtractor(
            self.paramsFile)

    def extract(self, image, mask):
        image_sitk = sitk.GetImageFromArray(image)
        mask_sitk = sitk.GetImageFromArray(mask)
        return self.extractor.execute(image_sitk, mask_sitk)
