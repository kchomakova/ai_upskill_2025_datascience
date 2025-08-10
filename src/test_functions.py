import unittest
import pandas as pd
from functions import extract_categorical_cols, extract_numerical_cols, drop_cols_no_cardinality


class TestCustomFunctions(unittest.TestCase):

    def setUp(self):
        self.data = pd.DataFrame({
            'Age': [25, 30, 35],
            'PatientID': [1, 2, 3],
            'Category1': [0, 1, 0],
            'Category2': [1, 0, 1],
            'NumericCol': [10.5, 20.3, 30.1],
            'NoCardinalityCol': ['text', 'text', 'text']
        })

    def tearDown(self):
        del self.data

    def test_extract_categorical_cols(self):
        # Call the function to extract categorical columns which are not Age and PatentID
        actual_data = extract_categorical_cols(self.data)

        expected_data = pd.DataFrame({
            'Category1': [0, 1, 0],
            'Category2': [1, 0, 1]
        })

        # Assert that the extracted DataFrame matches the actual DataFrame
        pd.testing.assert_frame_equal(actual_data, expected_data)

    def test_extract_numerical_cols(self):
        # Call the function to extract numeric (float) columns or Age
        actual_data = extract_numerical_cols(self.data)

        expected_data = pd.DataFrame({
            'NumericCol': [10.5, 20.3, 30.1],
            'Age': [25, 30, 35]
        })

        # Assert that the extracted DataFrame matches the actual DataFrame
        pd.testing.assert_frame_equal(actual_data, expected_data)

    def test_drop_cols_no_cardinality(self):
        # Call the function to drop columns with no cardinality
        actual_data = drop_cols_no_cardinality(self.data)

        expected_data = pd.DataFrame({
            'Age': [25, 30, 35],
            'PatientID': [1, 2, 3],
            'Category1': [0, 1, 0],
            'Category2': [1, 0, 1],
            'NumericCol': [10.5, 20.3, 30.1]
        })

        # Assert that the DataFrame after dropping columns matches the expected DataFrame
        pd.testing.assert_frame_equal(actual_data, expected_data)   



if __name__ == '__main__':
    unittest.main()