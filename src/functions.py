import pandas as pd
import itertools
from scipy.stats import chi2_contingency
from sklearn.preprocessing import StandardScaler


def extract_categorical_cols(dataset: pd.DataFrame) -> pd.DataFrame:
    """ 
    Extracts columns with integer data type from the given DataFrame because 
    semantically these columns are categorical, most of them binary, and any analysis 
    on them will treat them as categorical variables. Age and PatientID are excluded
    because they are not semantically categorical. 

    Args:
        dataset: The input DataFrame from which to extract integer columns.
    Returns:    
        A DataFrame containing only the integer columns from the input dataset.
    """

    dataset_cat = dataset.select_dtypes(include='int')
    dataset_cat = dataset_cat[[col for col in dataset_cat.columns if col not in ['Age','PatientID']]]
    
    return dataset_cat


def extract_numerical_cols(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts columns with float data type from the given DataFrame because 
    semantically these columns are numerical. Age is explicitely included 
    because semantically it is numerical, even though it is an integer type in the dataset.

    Args:
        dataset: The input DataFrame from which to extract float columns.
    Returns:    
        A DataFrame containing only the float columns from the input dataset.
    """

    dataset_num = dataset.select_dtypes(include = 'float')
    dataset_num = dataset[list(dataset_num.columns) + ['Age']]
    
    return dataset_num


def calculate_pair_wise_correlation(dataset: pd.DataFrame) -> pd.DataFrame:

    """
    Calculates the Pearson correlation for each combination of columns and returns a DataFrame 
    containing the results. Only numerical variables are considered for this calculation.

    Args:
        dataset: The input DataFrame containing the columns for pairwise correlation.
    Returns:    
        A DataFrame containing the pair-wise correlation rounded to the second decimal place.
    """
    
    dataset_numeric = extract_numerical_cols(dataset)
    corr_matrix = dataset_numeric.corr(method='pearson').round(2)

    return corr_matrix


def calculate_pair_wise_chi_square_test(dataset: pd.DataFrame, p_value_threshold: float) -> pd.DataFrame:

    """
    Calculates the Chi-square test for independence for each combination of categorical columns.

    Args:
        dataset: The input DataFrame containing the columns for pairwise chi-square test. 
        p_value_threshold: The p_value threshold to determine significance in the chi-square test.

    Returns:    
        A DataFrame containing the pair-wise chi-square test results, including the statistic, p-value.
    
    """

    dataset_categorical = extract_categorical_cols(dataset)
    pairs = list(itertools.combinations(dataset_categorical.columns, 2))
    output_df = pd.DataFrame(columns=['first_col', 'second_col', 'chi_square_stat', 'p_value'])

    for pair in pairs:
        contingency_table = pd.crosstab(dataset_categorical[pair[0]], dataset_categorical[pair[1]])
        chi2_stat, p_val  = chi2_contingency(contingency_table)[:2]

        if p_val <= p_value_threshold:
            new_row = pd.DataFrame({'first_col': [pair[0]], 'second_col': [pair[1]], 'chi_square_stat': [chi2_stat], 'p_value': [p_val]})
            output_df = pd.concat([output_df, new_row], ignore_index=True)

        output_df = output_df.sort_values(by = ['first_col', 'second_col'])

    return output_df
    

def drop_cols_no_cardinality(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Drops columns that have no cardinality (i.e., all values are the same) from the DataFrame.

    Args:
        dataset: The input DataFrame from which to drop columns with no cardinality.
    
    Returns:    
        A DataFrame with columns that have no cardinality removed.
    """
    
    cols_to_drop = [col for col in dataset.columns if dataset[col].nunique() <= 1]
    dataset = dataset.drop(columns=cols_to_drop, axis = 1)
    return dataset


def standardize_numeric_variables(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Standardizes the numeric variables in the DataFrame by subtracting the mean and dividing by the standard deviation.

    Args:
        dataset: The input DataFrame containing numeric variables to be standardized.
    
    Returns:    
        A DataFrame with standardized numeric variables.
    """

    dataset_numeric = extract_numerical_cols(dataset)
    
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(dataset_numeric)
    standardized_df = pd.DataFrame(standardized_data, columns=dataset_numeric.columns)
    
    return standardized_df