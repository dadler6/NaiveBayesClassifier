# NaiveBayesClassifier


Author: Dan Adler
GitHub: https://github.com/dadler6
Website: http://dadler.co/

This is my own implementation of a Naive Bayes Classifier for text data, purely for learning purposes.  The file folder structure is as follows:

**src**
The code for the naive bayes classifier implementation.

src/NaiveBayesClassifeir.py:
The Naive Bayes classifier.  It has two main functions, train and predict.  See implementation notes for more details.

**examples**
Examples of running the various files in src.

**data**
Data for the various example cases I'm developing.

**tests**
Test cases of the src code.

tests/test\_NaiveBayesClassifier.py:
Test cases for the Naive Bayes Classifier function.  Tests were built using pytest, and can be run within the command *pytest test\_NaiveBayesClassifier.py*.


## Implementation Notes

The file, src/NaiveBayesClassifeir.py, has one class, called "NaiveBayesClassifier".  The function docstring is as follows:

```python
class NaiveBayesClassifier(object):
    """
    Naive Bayes classifier object.  The object will be able to take
    a cleaned set of documents and classifications of those documents,
    and simply will use a Naive Bayes algorithm (applying bayes rule
    with naive assumption of conditional independence of features between
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
```

There are currently no parameters you need to enter when initializing a NaiveBayesClassifier object, and thus initialization is simple and can be done like so:

```python
from NaiveBayesClassifier import NaiveBayesClassifier as nb

nb = NaiveBayesClassifier()
```

The two main functions are called "train" and "predict".  The train function is defined as follows:

```python
def train(self, x, y):
    """
    Train the Naive Bayes Classifier to the dataset provided.

    :param x: A 1D iterable (list, pd.DataFrame, series, etc), each
              entry is a specific document
    :param y: A 1D iterable (list, pd.DataFrame, series, etc), each entry
           is the class of a specific document, positionally
              within the same order as X
    """
```

Note that "x" should be a one-dimensional object (or can be a column matrix, or pd.DataFrame with a single column), and should have text that has been cleaned.  Currently, each word must be separated by a single space: " ".

"y" can be a string, integer, or float, but should be categorical.  It also must be one-dimensional.

Once the NaiveBayesClassifier is trained, you can predict using the class on new data, using the same formatted "x" as utilized in the "train" function.  The predict doctstring is defined as follows:

```python
def predict(self, x):
    """
    Use the classifier to predict a specific class of a
    document.

    :param x: A 1D iterable (list, pd.DataFrame, series), each
              entry is a specific document
    :return y: list<object>, each entry is a class belonging to
               each entry in X
```