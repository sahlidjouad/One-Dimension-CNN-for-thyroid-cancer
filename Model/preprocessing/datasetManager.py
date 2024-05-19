from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import tensorflow as tf


class DatasetManger:
    def __init__(self, x, y, batch_size=10, n_splits=2, test_size=0.2, Shuffle=False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.n_splits = n_splits
        self.test_size = test_size
        self.Shuffle = Shuffle

    def split(self):
        spliter = StratifiedShuffleSplit(
            n_splits=self.n_splits, test_size=self.test_size)
        listOfsets = []
        for index_training, index_test in spliter.split(self.x, self.y):
            train_value = self.x.iloc[index_training]
            train_label = self.y.iloc[index_training]
            test_value = self.x.iloc[index_test]
            test_label = self.y.iloc[index_test]

            listOfsets.append(
                (train_value, train_label, test_value, test_label))

        return listOfsets

    def convert_df_to_dataset(self, x, y):
        df = x.copy()
        labels = y.copy()
        df = {key: np.array(value)[:, None] for key, value in x.items()}
        ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
        if self.Shuffle:
            ds = ds.shuffle(buffer_size=len(df))
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(self.batch_size)
        return ds

    def convert_df_to_dataset_x(self, x):
        df = x.copy()
        df = {key: np.array(value)[:, None] for key, value in x.items()}
        ds = tf.data.Dataset.from_tensor_slices(dict(df))
        if self.Shuffle:
            ds = ds.shuffle(buffer_size=len(df))
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(self.batch_size)
        return ds

    def create_dataset(self):
        listofDataset = []
        for (trainx, trainy, testx, testy) in self.split():
            dstr = self.convert_df_to_dataset(trainx, trainy)
            save_state = self.Shuffle
            self.Shuffle = False
            dste = self.convert_df_to_dataset(testx, testy)
            self.Shuffle = save_state
            listofDataset.append((dstr, dste))
        return listofDataset
