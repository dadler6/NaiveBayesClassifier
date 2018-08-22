"""
test_NaiveBayesClassifier.py

Written by Dan Adler
Email: (daadler0309@gmail.com)
GitHub: https://github.com/dadler6/

Test cases for NaiveBayesClassifier.py

Package requirements:
numpy
pandas
pytest
../src/NaiveBayesClassifier
"""

# Imports
import sys
import pytest
import numpy as np
import pandas as pd

# Import specific module
sys.path.insert(0, '../src')
from NaiveBayesClassifier import NaiveBayesClassifier as Nb


@pytest.fixture(scope='module')
def set_format_input_params():
    """
    Setup params to be used for naive bayes testing
    """
    # List
    test_data_list_x = ['Hello, my name is Dan', 'Hello my name is Jill']
    test_data_list_y = ['Sad', 'Happy']

    # Series
    test_data_series_x = pd.Series(data=test_data_list_x)
    test_data_series_y = pd.Series(data=test_data_list_y)

    # DataFrame
    test_data_series_x_df = pd.DataFrame({'X': test_data_list_x})
    test_data_series_y_df = pd.DataFrame({'Y': test_data_list_y})

    return {
        'list_x': test_data_list_x,
        'list_y': test_data_list_y,

        'series_x': test_data_series_x,
        'series_y': test_data_series_y,

        'df_x': test_data_series_x_df,
        'df_y': test_data_series_y_df,
    }


# noinspection 801,PyShadowingNames
def test_format_input_correct(set_format_input_params):
    """
    Test the nb.format_input function

    :param set_format_input_params: dict<str:object>, the params to test
        inputting different objects to program
    """
    # Get inputs
    test_data_list = set_format_input_params['list_x']
    test_data_series = set_format_input_params['series_x']
    test_data_df = set_format_input_params['df_x']

    # Test
    result_list = Nb.format_input(test_data_list, 'X')
    result_series = Nb.format_input(test_data_series, 'X')
    result_df = Nb.format_input(test_data_df, 'X')

    # Assertions
    assert test_data_list == result_list
    assert test_data_list == result_series
    assert test_data_list == result_df
