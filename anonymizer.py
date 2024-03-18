import argparse
import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from tqdm import tqdm
import argparse


def get_span_of_df(df_, partition_, categorical_columns_, scale=None):
    """
    Function for getting the span of the dataframe. For example the number of unique items
    in the columns of the dataframe.
    :param df_: pandas dataframe for which the span will be calculated
    :param partition_: dataframe partition which will be used to calculate the spans
    :param categorical_columns_: list of columns with string data type
    :param scale: scaling factor for scaling the data in the dataframe
    :return: returns the spa of ech column in a dictionary
    """
    spans = {}
    for column in df_.columns:
        if column in categorical_columns_:
            span = len(df_[column][partition_].unique())
        else:
            span = df_[column][partition_].max() - df_[column][partition_].min()
        if scale is not None:
            span = span / scale[column]
        spans[column] = span
    return spans


def split_dataframe(df_, partition_, categorical_columns_, column_):
    """
    Function for splitting the given dataframe into two partitions based on the median value of
    the chosen attribute, aiming to distribute records as evenly as possible.
    :param df_: pandas dataframe that is to be split
    :param partition_: The partition to split
    :param categorical_columns_: list of columns with string data type
    :param column_: The column along which the dataframe will be split
    :return: return a tuple that contains a split of the input partition
    """
    df_partition_ = df_[column_][partition_]
    if column_ in categorical_columns_:
        values = df_partition_.unique()
        lv = set(values[:len(values) // 2])
        rv = set(values[len(values) // 2:])
        return df_partition_.index[df_partition_.isin(lv)], df_partition_.index[df_partition_.isin(rv)]
    else:
        median = df_partition_.median()
        df_split_1 = df_partition_.index[df_partition_ < median]
        df_split_2 = df_partition_.index[df_partition_ >= median]
        df_split_tuple = (df_split_1, df_split_2)
        return df_split_tuple


def is_k_anonymous(partition_, k_=3):
    """
    Checks if the given partition is k anonymous
    :param partition_: dataset partition to check
    :param k_: number of anonymous
    :return: whether it is k anonymous or not
    """
    if len(partition_) < k_:
        return False
    return True


def create_data_partition(df_, quasi_identifiers_, sensitive_attribute_, categorical_columns_, scale_, is_k_anonymous_, k_=3):
    """
    Function for partitioning the dataset  ensuring that each partition has at least k records.
    This step is important for the anonymization process (Mondrian).
    :param df_: pandas dataframe that is to be partitioned
    :param quasi_identifiers_: list of quasi-identifiers in the dataframe
    :param sensitive_attribute_: the name of the sensitive attribute of the dataset
    :param categorical_columns_: list of columns with string data type
    :param scale_: scaling factor for scaling the values in the dataframe
    :param is_k_anonymous_: this a function that is used to validate the validity of the partition by checking if it is k anonymous
    :param k_: number of anonymous
    :return: returns the list of valid partitions
    """
    created_partitions = []
    init_partitions = [df_.index]
    while init_partitions:
        partition = init_partitions.pop(0)
        spans = get_span_of_df(
            df_=df_[quasi_identifiers_],
            partition_=partition,
            categorical_columns_=categorical_columns_,
            scale=scale_
        )
        for column, span in sorted(spans.items(), key=lambda x: -x[1]):
            part_1, part_2 = split_dataframe(
                df_=df_,
                partition_=partition,
                categorical_columns_=categorical_columns_,
                column_=column
            )
            if not is_k_anonymous_(part_1, k_) or not is_k_anonymous_(part_2, k_):
                continue
            init_partitions.extend((part_1, part_2))
            break
        else:
            created_partitions.append(partition)
    return created_partitions


def combine_categorical_data(value_list):
    """
    Combines the values of the rows with unique values of categorical columns together to make them anonymous
    :param value_list: list of unique values of categorical columns to combine
    :return: combined value for the given list
    """
    return [','.join(set(value_list))]


def combine_numerical_column(value_list):
    """
    Converts numerical values to a range to make them anonymous
    :param value_list: list of values of numerical columns to combine
    :return: a range of numerical values for the given list
    """
    if min(value_list) == max(value_list):
        return f'{min(value_list)}'
    return f'{min(value_list)}-{max(value_list)}'  # series.mean()


def anonymize_dataset(df_, partitions_, quasi_identifiers_, categorical_columns_, sensitive_attribute_, max_partitions=None):
    """
    Function to anonymize a dataset.
    :param df_: input dataset as a pandas dataframe
    :param partitions_: partitions of the dataset
    :param quasi_identifiers_: quasi identifiers in the dataset for which we want to anonymize the dataset
    :param categorical_columns_: list of columns with string data type
    :param sensitive_attribute_: sensitive attribute of the dataset
    :param max_partitions: maximum allowed partition for the anonymized dataset
    :return: k-anonymized dataset
    """
    aggregations = {}
    for column in quasi_identifiers_:
        if column in categorical_columns_:
            aggregations[column] = combine_categorical_data
        else:
            aggregations[column] = combine_numerical_column
    rows = []
    for i, partition in enumerate(tqdm(partitions_)):
        if max_partitions is not None and i > max_partitions:
            break
        grouped_columns = df_.loc[partition].agg(aggregations, squeeze=False)
        for gck in grouped_columns.index:
            grouped_columns[gck] = grouped_columns[gck]
        sensitive_counts = df_.loc[partition].groupby(sensitive_attribute_).agg({sensitive_attribute_: 'count'})
        values = grouped_columns.to_dict()
        for sensitive_value, count in sensitive_counts[sensitive_attribute_].items():
            if count == 0:
                continue
            values.update({
                sensitive_attribute_: sensitive_value,
                'Count': count,

            })
            rows.append(values.copy())
    return pd.DataFrame(rows)


def comma_separated_strings(value):
    """Custom argparse type for comma-separated strings."""
    if not value:
        raise argparse.ArgumentTypeError("Value cannot be empty.")
    return value.split(',')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""
            This script anonymizes the Adult dataset, ensuring k-anonymity to protect individual privacy. 
            The Adult dataset includes demographic and employment information, with the goal of predicting whether an individual earns more than $50K a year.
            
            Identifiers in the Adult dataset are categorized as follows:
            - Explicit Identifiers: Attributes that can directly identify an individual, such as name or social security number. The Adult dataset typically does not contain explicit identifiers for privacy reasons.
            - Quasi-Identifiers: Attributes that, when combined, could potentially identify an individual. Examples include 'age', 'workclass', 'education', 'marital-status', 'occupation', 'race', 'sex', and 'native-country'.
            - Sensitive Attributes: Attributes that provide sensitive information about an individual, which could lead to privacy concerns if disclosed. In the Adult dataset, 'income' is considered a sensitive attribute, as it reveals personal financial status.
            
            The anonymization process focuses on modifying quasi-identifiers to prevent the re-identification of individuals while maintaining the utility of the dataset for analysis.
        """
    )

    parser.add_argument('-k', '--k_value', type=int, default=3,
                        help='The k-value for k-anonymity, determining the level of privacy. Default is 3.')
    parser.add_argument('-q', '--quasi_identifiers', type=comma_separated_strings, default='age,education-num,hours-per-week',
                        help='A list of column names to be treated as quasi-identifiers, separated by commas. Defaults to "age,education-num,hours-per-week".')
    parser.add_argument('-s', '--sensitive_attribute', type=str, default='income',
                        help='The name of the sensitive attribute in the dataset. This attribute will receive special handling to maintain privacy. Defaults to "income"')
    parser.add_argument('-o', '--output', type=str, default='anonymized_data.csv',
                        help='Path to save the anonymized dataset. Default is "anonymized_data.csv".')

    args = parser.parse_args()

    print(f'Anonymizing dataset for sensitive attribute: {args.sensitive_attribute} and quasi identifiers: {",".join(args.quasi_identifiers)}')

    adult_dataset = fetch_ucirepo(id=2)
    adult_df = adult_dataset.data.features
    adult_df['income'] = adult_dataset.data.targets

    categorical_columns = adult_df.select_dtypes(include=['object']).columns.tolist()
    numerical_columns = adult_df.select_dtypes(include=['int64']).columns.tolist()

    dataset_span = get_span_of_df(
        df_=adult_df,
        partition_=adult_df.index,
        categorical_columns_=categorical_columns
    )

    dataset_partitions = create_data_partition(
        df_=adult_df,
        quasi_identifiers_=args.quasi_identifiers,
        sensitive_attribute_=args.sensitive_attribute,
        categorical_columns_=categorical_columns,
        scale_=dataset_span,
        is_k_anonymous_=is_k_anonymous,
        k_=args.k_value
    )

    print('Total number of partitions: {}'.format(len(dataset_partitions)))

    anonymized_df = anonymize_dataset(
        df_=adult_df,
        partitions_=dataset_partitions,
        quasi_identifiers_=args.quasi_identifiers,
        sensitive_attribute_=args.sensitive_attribute,
        categorical_columns_=categorical_columns
    )

    print('Done anonymizing dataset.')
    print('Anonymized dataset is saved to {}'.format(args.output))
    anonymized_df.to_csv(args.output)

