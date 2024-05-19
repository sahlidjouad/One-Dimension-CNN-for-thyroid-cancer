import pandas as pd


def Get_DataFrame_Info(dataset):
    """
    Get info dataset
    arg1: DataFrame (dataset)

    return DataFrame
    """
    column_name = [ x for x in dataset.columns]
    count_non_null_value = [dataset[x].count() for x in column_name]
    column_type = [dataset[x].dtype.name for x in column_name]

    return pd.DataFrame(
        {
            "Column Name": column_name,
            "Non-Null Count": count_non_null_value,
            "Data type:": column_type
        }
    )


def Get_list_unique_value_and_percentile(dataset, column_name):
    """
    
    """
    lis = []
    for item in dataset[column_name].unique():
        count = [value == item  for value in dataset[column_name]].count(True)
        precetail = (count / dataset[column_name].size) *100
        lis.append((item, f'{precetail:.2f}%'))  
    return lis


def Get_ragne_value_column(dataset, column_name):
    return [dataset[column_name].min(), dataset[column_name].max()]


def Get_DataFrame_Value_info(dataset,column_names, typeofFeature):
    
    ranges = [ str(Get_list_unique_value_and_percentile(dataset, column_name)) if typeofFeature != 'Quantitative' else str(Get_ragne_value_column(dataset, column_name))  for column_name in column_names]
            
    return pd.DataFrame({
        'Columns': column_names,
        'Range': ranges
    })
    

