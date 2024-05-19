import unittest
from utils.core import Load_data
import pandas as pd
from Model.preprocessing.pipelines import ProcessData
from Model.preprocessing.datasetManager import DatasetManger
from Model.base.train import cross_validation
from Model.base.model import make_model
from Model.base.logger import LoggerManager
import tensorflow as tf
from Model.base.evaluator import EvalutionClassifer, Drow
import os


class Test_evaluator(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.list_prediction = [
            0.9 for x in range(13)] + [0.3 for x in range(32)] + [0.7 for x in range(4)] + [0.3 for x in range(23)]
        self.list_real_value = [
            1 for x in range(13)] + [0 for x in range(32)] + [0 for x in range(4)] + [1 for x in range(23)]

        self.df_dataset = Load_data("./dataset/metadata.csv")
        self.df_dataset.iloc[109] = self.df_dataset.iloc[109].fillna(0.9)
        self.df_dataset = self.df_dataset.drop(["annot_id"], axis=1)
        self.X = self.df_dataset.drop("histopath_diagnosis", axis=1)
        self.Y = self.df_dataset['histopath_diagnosis']

        categorical_features = ["sex", "location"]

        numerical_features = ["ti-rads_level",
                              "size_x", "size_y", "size_z", "age"]

        Ti_rads_featurea = ["ti-rads_composition", "ti-rads_echogenicity",
                            "ti-rads_margin", "ti-rads_shape", "ti-rads_echogenicfoci"]

        self.X = ProcessData(
            self.X, categorical_features, numerical_features, Ti_rads_featurea)
        self.manager_dataset = DatasetManger(
            self.X, self.Y, batch_size=10, n_splits=2, test_size=0.20, Shuffle=False)

        self.dataset = self.manager_dataset.create_dataset()[0]
        self.model = model = tf.keras.models.load_model(
            "./Model/trained_model/13model0.keras")

        self.object = EvalutionClassifer([self.model], [self.dataset[0]])

    @classmethod
    def tearDownClass(self):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_Get_actual_label(self):
        train_set = self.dataset[0]
        output = self.object.Get_actual_label(train_set)
        self.assertEqual(len(output), len(self.Y))

        pass

    def test_accuracy(self):
        calculator = tf.keras.metrics.BinaryAccuracy(name='accuracy')
        calculator.update_state(self.list_real_value,
                                self.list_prediction)
        accuracy = calculator.result().numpy()
        self.assertEqual(accuracy, 0.695)
