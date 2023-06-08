import os
import pandas as pd


def choose_class(df, level, value):
    """ choose certain class of data

    Args:
        df (pd.DataFrame): original dataset
        level (str): TaxonomyLevel
        value (str): TaxonomyValue

    Returns:
        pd.DataFrame: df containing data we want
    """
    df_out = df[
        (df['TaxonomyLevel'] == level)
        &
        (df['TaxonomyValue'] == value)
    ]

    return df_out


def clean_data(df_node, df_edge):
    """ remove invalid edges

    Args:
        df_node (pd.DataFrame): df of nodes
        df_edge (pd.DataFrame): df of edges

    Returns:
        pd.DataFrame: df after data cleaning
    """
    node_ava_id = df_node['InstitutionId'].unique().tolist()
    df_edge_clean = df_edge[
        df_edge['InstitutionId'].isin(node_ava_id) &
        df_edge['DegreeInstitutionId'].isin(node_ava_id)
    ]

    return df_edge_clean


if __name__ == '__main__':

    # create a folder for PyG data
    if not os.path.exists('pyg_data/raw'):
        os.makedirs('pyg_data/raw')

    # load raw data
    df_node = pd.read_csv('raw_data/institution-stats.csv')
    df_edge = pd.read_csv('raw_data/edge-lists.csv')

    # drop data containing NaN
    node_nan_col = ['InstitutionId', 'NonAttritionEvents', 'AttritionEvents', 'PrestigeRank']
    edge_nan_col = ['InstitutionId', 'DegreeInstitutionId', 'Total']
    df_node = df_node.dropna(subset=node_nan_col)
    df_edge = df_edge.dropna(subset=edge_nan_col)

    # unify data type
    node_int_col = ['InstitutionId', 'NonAttritionEvents', 'AttritionEvents']
    edge_int_col = ['DegreeInstitutionId', 'Total', 'Men', 'Women']
    df_node[node_int_col] = df_node[node_int_col].astype('int')
    df_edge[edge_int_col] = df_edge[edge_int_col].astype('int')

    # create subset for training models
    fields = ['Chemistry', 'Mathematics', 'Physics', 'Computer Science']
    for _field in fields:
        _df_node = choose_class(df_node, 'Field', _field)
        _df_edge = choose_class(df_edge, 'Field', _field)
        _df_edge = clean_data(_df_node, _df_edge)
        csv_name = _field.lower().replace(' ', '_')
        _df_node.to_csv(f'pyg_data/raw/{csv_name}_node.csv', index=False)
        _df_edge.to_csv(f'pyg_data/raw/{csv_name}_edge.csv', index=False)
