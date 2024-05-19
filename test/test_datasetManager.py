
import unittest
from Model.preprocessing.datasetManager import DatasetManger
from utils.core import Load_data
import pandas as pd
import numpy as np
# python -m unittest discover -s test -p "test_*.py"


class Test_datasetManager(unittest.TestCase):

    @classmethod
    def setUpClass(self):

        self.df_dataset = Load_data("./dataset/metadata.csv")
        self.df_dataset.iloc[109] = self.df_dataset.iloc[109].fillna(0.9)
        self.df_dataset = self.df_dataset.drop(["annot_id"], axis=1)
        self.X = self.df_dataset.drop("histopath_diagnosis", axis=1)
        self.Y = self.df_dataset['histopath_diagnosis']
        self.object = DatasetManger(self.X, self.Y, Shuffle=True)

    @classmethod
    def tearDownClass(self):
        pass

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_split(self):
        results = self.object.split()

        for result in results:

            self.assertEqual(len(result), 4)

            train_x, train_y, test_x, test_y = result

            self.assertIsInstance(train_x, pd.DataFrame)
            self.assertIsInstance(train_y, pd.Series)
            self.assertIsInstance(test_x, pd.DataFrame)
            self.assertIsInstance(test_y, pd.Series)

            self.assertEqual(len(train_x.index), len(train_y.index))
            self.assertEqual(len(test_x.index), len(test_y.index))
            self.assertEqual(list(train_x.columns), list(test_x.columns))
            # self.assertEqual(list(train_y), list(test_y))

            self.assertEqual(train_x.isna().all(axis=None), False)
            self.assertEqual(test_x.isna().all(axis=None), False)
            self.assertEqual(train_y.isna().all(axis=None), False)
            self.assertEqual(test_y.isna().all(axis=None), False)

    def is_consistency(self, ds, df_original):

        ds = ds.unbatch()
        for x, y in ds:

            x["histopath_diagnosis"] = y

            dict_of_values = {key: np.squeeze(value.numpy()).item(
                0) for key, value in x.items()}

            dict_of_values = {key: value.decode(
            )if type(value) is bytes else value for key, value in dict_of_values.items()}

            copy_of_df = df_original.copy()
            for key, value in dict_of_values.items():
                copy_of_df = copy_of_df.loc[copy_of_df[key] == value]

            if copy_of_df.empty:
                return False

        return True

    def is_column_indenticality(self, ds, df_original):
        value = ds.take(1)
        value = value.unbatch()
        dic = value.as_numpy_iterator()
        print(dic)

    def test_conver_df_to_dataset(self):

        dataset = self.object.convert_df_to_dataset(self.X, self.Y)

        self.assertEqual(self.is_consistency(dataset, self.df_dataset), True)
        self.is_column_indenticality(dataset, self.df_dataset)
        # self.assertEqual(self.x.columns.name == )
