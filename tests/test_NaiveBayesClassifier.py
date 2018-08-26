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
    test_data_list_y = [0, 1]

    # Series
    test_data_series_x = pd.Series(data=test_data_list_x)
    test_data_series_y = pd.Series(data=test_data_list_y)

    # DataFrame
    test_data_df_x = pd.DataFrame({'X': test_data_list_x})
    test_data_df_y = pd.DataFrame({'Y': test_data_list_y})

    # np.ndarray
    test_data_arr_x = np.array(test_data_list_x)
    test_data_arr_y = np.array(test_data_list_y)

    # np.matrix
    test_data_matrix_x = np.matrix(test_data_list_x).transpose()
    test_data_matrix_y = np.matrix(test_data_list_y).transpose()

    return {
        'list_x': test_data_list_x,
        'list_y': test_data_list_y,

        'series_x': test_data_series_x,
        'series_y': test_data_series_y,

        'df_x': test_data_df_x,
        'df_y': test_data_df_y,

        'arr_x': test_data_arr_x,
        'arr_y': test_data_arr_y,

        'matrix_x': test_data_matrix_x,
        'matrix_y': test_data_matrix_y,
    }


# noinspection 801,PyShadowingNames
def test_format_input_correct(set_format_input_params):
    """
    Test the Nb.format_input function

    :param set_format_input_params: dict<str:object>, the params to test
        inputting different objects to program
    """
    # Get inputs
    test_data_list = set_format_input_params['list_x']
    test_data_series = set_format_input_params['series_x']
    test_data_df = set_format_input_params['df_x']
    test_data_arr = set_format_input_params['arr_x']
    test_data_matrix = set_format_input_params['matrix_x']

    # Test
    result_list = Nb.format_input(test_data_list, 'X')
    result_series = Nb.format_input(test_data_series, 'X')
    result_df = Nb.format_input(test_data_df, 'X')
    result_arr = Nb.format_input(test_data_arr, 'X')
    result_matrix = Nb.format_input(test_data_matrix, 'X')

    # True assertions
    assert test_data_list == result_list
    assert test_data_list == result_series
    assert test_data_list == result_df
    assert test_data_list == result_arr
    assert test_data_list == result_matrix

    # Problematic assertions
    with pytest.raises(ValueError):
        Nb.format_input(test_data_df.transpose(), 'X')

    with pytest.raises(ValueError):
        Nb.format_input(test_data_matrix.transpose(), 'X')


# noinspection 801,PyShadowingNames
def test_check_inputs(set_format_input_params):
    """
       Test the Nb.check_inputs function

       :param set_format_input_params: dict<str:object>, the params to test
           inputting different objects to program
       """
    # Get inputs
    test_data_list_x = set_format_input_params['list_x']
    test_data_df_x = set_format_input_params['df_x']
    test_data_arr_x = set_format_input_params['arr_x']
    test_data_matrix_x = set_format_input_params['matrix_x']

    test_data_list_y = set_format_input_params['list_y']
    test_data_arr_y = set_format_input_params['arr_y']
    test_data_matrix_y = set_format_input_params['matrix_y']

    # True df
    true_df = pd.DataFrame({
        'Documents': test_data_list_x,
        'Class': test_data_list_y
    })

    # Test
    result_list_x_list_y = Nb.check_inputs(test_data_list_x, test_data_list_y)
    result_arr_x_arr_y = Nb.check_inputs(test_data_arr_x, test_data_arr_y)
    result_df_x_arr_y = Nb.check_inputs(test_data_df_x, test_data_arr_y)
    result_matrix_x_arr_y = Nb.check_inputs(test_data_matrix_x, test_data_arr_y)

    # Assertions
    assert pd.testing.assert_frame_equal(
        true_df,
        result_list_x_list_y
    ) is None

    assert pd.testing.assert_frame_equal(
        true_df,
        result_arr_x_arr_y
    ) is None

    assert pd.testing.assert_frame_equal(
        true_df,
        result_df_x_arr_y
    ) is None

    assert pd.testing.assert_frame_equal(
        true_df,
        result_matrix_x_arr_y
    ) is None

    # Problematic assertions
    with pytest.raises(ValueError):
        Nb.check_inputs(test_data_df_x.transpose(), test_data_arr_y)

    with pytest.raises(ValueError):
        Nb.check_inputs(test_data_df_x, test_data_matrix_y.transpose())


@pytest.fixture(scope='module')
def setup_training_data():
    """
    Setup params to be used for the training dataset.
    """
    x_train = pd.DataFrame({
        'Docs': [
            'sunny hot high false',
            'sunny hot high true',
            'overcast hot high false',
            'rainy mild high false',
            'rainy cool normal false',
            'rainy cool normal true',
            'overcast cool normal true',
            'sunny mild high false',
            'sunny cool normal false',
            'rainy mild normal false',
            'sunny mild normal true ',
            'overcast mild high true',
            'overcast hot normal false',
            'rainy mild high true'
        ],
    })

    y_train = pd.DataFrame({
        'Class': [
            'no',
            'no',
            'yes',
            'yes',
            'yes',
            'no',
            'yes',
            'no',
            'yes',
            'yes',
            'yes',
            'yes',
            'yes',
            'no'
        ]
    })

    nb = Nb()
    nb.train(x_train, y_train)

    x_test = pd.DataFrame({
        'Docs': ['sunny cool high true']
    })

    return {
        'nb': nb,
        'x_test': x_test
    }


# noinspection 801,PyShadowingNames
def test_training_prior_dist(setup_training_data):
    """
    Test the training of the naive bayes classifier with the prior dist

    :param setup_training_data: dict<str: obj>: Dictionary holding the
        training data
    """
    # Get training data
    nb = setup_training_data['nb']

    # Get the prior dist
    priors = nb.get_prior_dist()

    # True class
    true_no_prior = 5.0 / 14
    true_yes_prior = 9.0 / 14

    # Test
    assert priors['no'] == true_no_prior
    assert priors['yes'] == true_yes_prior


# noinspection 801,PyShadowingNames
def test_training_likelihood_dist(setup_training_data):
    """
    Test the training of the naive bayes classifier with the likelihood dist

    :param setup_training_data: dict<str: obj>: Dictionary holding the
        training data
    """
    # Get training data
    nb = setup_training_data['nb']

    # Get the prior dist
    likelihood = nb.get_likelihood_dist()

    # True class
    true_sunny_yes = 2.0 / 9
    true_sunny_no = 3.0 / 5

    true_overcast_yes = 4.0 / 9
    true_overcast_no = 0.0 / 5

    true_rainy_yes = 3.0 / 9
    true_rainy_no = 2.0 / 5

    true_hot_yes = 2.0 / 9
    true_hot_no = 2.0 / 5

    true_mild_yes = 4.0 / 9
    true_mild_no = 2.0 / 5

    true_cool_yes = 3.0 / 9
    true_cool_no = 1.0 / 5

    true_high_yes = 3.0 / 9
    true_high_no = 4.0 / 5

    true_normal_yes = 6.0 / 9
    true_normal_no = 1.0 / 5

    true_false_yes = 6.0 / 9
    true_false_no = 2.0 / 5

    true_true_yes = 3.0 / 9
    true_true_no = 3.0 / 5

    # Test
    assert likelihood['yes']['sunny'] == true_sunny_yes
    assert likelihood['no']['sunny'] == true_sunny_no

    assert likelihood['yes']['overcast'] == true_overcast_yes
    assert likelihood['no']['overcast'] == true_overcast_no

    assert likelihood['yes']['rainy'] == true_rainy_yes
    assert likelihood['no']['rainy'] == true_rainy_no

    assert likelihood['yes']['hot'] == true_hot_yes
    assert likelihood['no']['hot'] == true_hot_no

    assert likelihood['yes']['mild'] == true_mild_yes
    assert likelihood['no']['mild'] == true_mild_no

    assert likelihood['yes']['cool'] == true_cool_yes
    assert likelihood['no']['cool'] == true_cool_no

    assert likelihood['yes']['high'] == true_high_yes
    assert likelihood['no']['high'] == true_high_no

    assert likelihood['yes']['normal'] == true_normal_yes
    assert likelihood['no']['normal'] == true_normal_no

    assert likelihood['yes']['false'] == true_false_yes
    assert likelihood['no']['false'] == true_false_no

    assert likelihood['yes']['true'] == true_true_yes
    assert likelihood['no']['true'] == true_true_no


# noinspection 801,PyShadowingNames
def test_predict(setup_training_data):
    """
    Test the training of the naive bayes classifier with the likelihood dist

    :param setup_training_data: dict<str: obj>: Dictionary holding the
        training data
    """
    # Get vars
    nb = setup_training_data['nb']
    x_test = setup_training_data['x_test']

    # Predict
    result_y = nb.predict(x_test)

    # Test
    true_y = ['no']

    assert result_y == true_y
