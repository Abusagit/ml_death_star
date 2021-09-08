import pandas as pd


def get_unique_values(df) -> list:
    """
    Helps to understand how many unique values each categorical columm has


    :param df: pandas DataFrame[<with categorical featurs only>]
    :return: list of pairs (<Column name>, <number of unique features>)
    """
    categorical_columns = tuple(col for col in df.columns if df[col].dtype == "object")
    object_nunique = tuple(map(lambda col: df[col].nunique(), categorical_columns))
    d = dict(zip(df.columns, object_nunique))
    return sorted(d.items(), key=lambda x: x[1])


def mostly_correlated(column, df):
    return df.corrwith(df[column]).sort_values(ascending=False)


def scatter_matrix(df, **kwargs):
    from pandas.plotting import scatter_matrix
    return scatter_matrix(df, **kwargs)


def concat_df(train_data, test_data):
    # Returns a concatenated df of training and test set
    return pd.concat([train_data, test_data], sort=True).reset_index(drop=True)


def display_missing(df):
    missing = []
    for col in df.columns.tolist():
        null_exist = df[col].isnull().sum()
        if null_exist:
            print(f'{col} column missing values: {null_exist}')
            missing.append(col)
    print("\n")
    if not missing:
        print("Values which missing MISSING are!")
    return missing

