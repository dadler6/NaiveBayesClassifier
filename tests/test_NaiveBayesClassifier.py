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
from NaiveBayesClassifier import NaiveBayesClassifier as nb


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
    test_data_series_x_df = pd.Series({'X': test_data_list_x})
    test_data_series_y_df = pd.Series({'Y': test_data_list_y})

    pass


def test_format_input_correct(set_format_input_params):
    """
    Test the nb.format_input function
    """
    test_data_list = ['Hello, my name is Dan', 'Hello my name is Jill']
    test_data_df = pd.DataFrame()