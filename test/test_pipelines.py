import unittest
from utils.core import Load_data
from Model.preprocessing.pipelines import ProcessData


class Test_ProcessData(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.df_dataset = Load_data("./../dataset/metadata.csv")
        pass

    @classmethod
    def tearDownClass(self):
        pass

    def setUp(self):
        pass

    def tearDwon(self):
        pass

    def test_ProcessData(self):
        """
        This test should check if there is any miss data or there is missing column
        """

        pass
