import numpy as np
def Get_Numerical_Columns(dataset):
    return dataset.select_dtypes(include=np.number).columns.values