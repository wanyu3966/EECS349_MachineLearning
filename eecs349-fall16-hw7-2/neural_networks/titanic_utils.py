import tensorflow
import numpy as np
import csv


def to_categorical(y, nb_classes):
    """ to_categorical.

        Convert class vector (integers from 0 to nb_classes)
        to binary class matrix, for use with categorical_crossentropy.

        Arguments:
            y: `array`. Class vector to convert.
            nb_classes: `int`. Total number of classes.
    """
    y = np.asarray(y, dtype='int32')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y


def load_csv(filepath, target_column=-1, columns_to_ignore=None,
             has_header=True, categorical_labels=False, n_classes=None):
    """ load_csv.

        Load data from a CSV file. By default the labels are considered to be the
        last column, but it can be changed by filling 'target_column' parameter.

        Arguments:
            filepath: `str`. The csv file path.
            target_column: The id of the column representing the labels.
                Default: -1 (The last column).
            columns_to_ignore: `list of int`. A list of columns index to ignore.
            has_header: `bool`. Whether the csv file has a header or not.
            categorical_labels: `bool`. If True, labels are returned as binary
                vectors (to be used with 'categorical_crossentropy').
            n_classes: `int`. Total number of class (needed if
                categorical_labels is True).

        Returns:
            A tuple (data, target).
    """

    from tensorflow.python.platform import gfile

    with gfile.Open(filepath) as csv_file:
        data_file = csv.reader(csv_file)
        if not columns_to_ignore:
            columns_to_ignore = []
        if has_header:
            header = next(data_file)
        data, target = [], []
        # Fix column to ignore ids after removing target_column
        for i, c in enumerate(columns_to_ignore):
            if c > target_column:
                columns_to_ignore[i] -= 1
        for i, d in enumerate(data_file):
            target.append(d.pop(target_column))
            data.append([_d for j, _d in enumerate(d) if j not in columns_to_ignore])
        if categorical_labels:
            assert isinstance(n_classes, int), "n_classes not specified!"
            target = to_categorical(target, n_classes)
    return data, target