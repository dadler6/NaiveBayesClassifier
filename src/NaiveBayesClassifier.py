"""
NaiveBayesClassifier.py

Written by Dan Adler
Email: (daadler0309@gmail.com)
GitHub: https://github.com/dadler6/

Self-implementation of a Naive Bayes Classifier for text
data.

Package requirements:
numpy
pandas
"""

# Imports
import pandas as pd
import numpy as np
from collections import Counter


class NaiveBayesClassifier(object):
    """
    Naive Bayes classifier object.  The object will be able to take
    a cleaned set of documents and classifications of those documents,
    and simply will use a Naive Bayes algorithm (applying bayes rule
    with naive assumption of conditional independents of features between
    classes), to develop a way to classify new documents by calculating
    the new document's most likely class.

    Parameters:
        self._prior_dist: Counter<object: int>, the prior probability
            distribution of each class, based upon the number of times
            that class appears amongst a set of documents.
        self._likelihoods: dict<object: Counter<string: int> >, the
            likelihoods for each word occurring given the word is from
            a certain class.  The dictionary has the form {v_j: {word: freq}}
            where v_j is a class, word is a word, and freq is the number
            of times the word occurs given a class.
    """

    def __init__(self):
        """
        Initialization of the Naive Bayes Classifier parameters.
        """
        # Initialize all parameters
        self._prior_dist = None
        self._likelihoods = {}

    @staticmethod
    def format_input(data, name):
        """
        Check an input (X or y) to see if it is in a correct format.


        :param data: A 1D iterable (list, pd.DataFrame, series, etc), each
                  entry is a specific document
        :param name: <str>, the name of the data variable input (x or y)
        :return: list<object>, the data as a list
        """
        val_error = ValueError(
            name + ' data input is not in correct format. ' +
            name + ' data must be 1-dimensional, and ' +
            name + ' data can be a numpy array, matrix, pd.Series,'
            'pd.DataFrame, or a list'
        )
        if (type(data) == pd.DataFrame) and (data.shape[0] == 1):
            data = list(data.iloc[:, 0].values)
        elif type(data) == pd.Series:
            data = list(data.values)
        elif (type(data) == np.matrix) and (data.shape[0] == 1):
            data = list(data[:, 0])
        elif type(data) == np.ndarray:
            data = list(data)
        elif type(data) == list:
            pass
        else:
            raise val_error

        return data

    @staticmethod
    def check_inputs(x, y):
        """
        Check the inputs (X and y) to see if they are the same length,
        and are 1D.  Also will place X and y together.

        :param x: A 1D iterable (list, pd.DataFrame, series, etc), each
                  entry is a specific document
        :param y: A 1D iterable (list, pd.DataFrame, series, etc), each entry
                  is the class of a specific document, positionally
                  within the same order as X

        :return: pd.DataFrame, a formatted version of X with y appended
                 on as a second column
        """
        # Check if x exists
        x = NaiveBayesClassifier.format_input(x, 'X')
        y = NaiveBayesClassifier.format_input(y, 'Y')

        df = pd.DataFrame({'Documents': x, 'Class': y})

        return df

    def develop_prior_dist(self, x_w_class):
        """
        Develop the prior probability distribution P(v_j) where
        v_j is a class.

        P(v_j) will simply be the number of documents with class j
        divided by the total number of documents.

        :param x_w_class: pd.DataFrame, a two column df, where the
            first column is a set of documents, and the second column
            is the class each document belongs to.
        """
        self._prior_dist = Counter(x_w_class['Class'].values)

    def develop_likelihood_dist(self, x_w_class):
        """
        Develop the likelihood distributions for each word within a
        class.

        :param x_w_class: pd.DataFrame, a two column df, where the
            first column is a set of documents, and the second column
            is the class each document belongs to.
        """
        for c in x_w_class.Class.unique():
            temp_df = x_w_class.loc[x_w_class['Class'] == c, :]
            all_docs_c = ' '.join(temp_df['Documents'].values)
            all_docs_c_split = all_docs_c.split(' ')
            self._likelihoods[c] = Counter(all_docs_c_split)

    def train(self, x, y):
        """
        Train the Naive Bayes Classifier to the dataset provided.

        :param x: A 1D iterable (list, pd.DataFrame, series, etc), each
                  entry is a specific document
        :param y: A 1D iterable (list, pd.DataFrame, series, etc), each entry
               is the class of a specific document, positionally
                  within the same order as X
        """
        x_clean = self.check_inputs(x, y)
        self.develop_prior_dist(x_clean)
        self.develop_likelihood_dist(x_clean)

    def predict(self, x):
        """
        Use the classifier to predict a specific class of a
        document.

        :param x: A 1D iterable (list, pd.DataFrame, series), each
                  entry is a specific document
        :return y: list<object>, each entry is a class belonging to
                   each entry in X
        """
        pass