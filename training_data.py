import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold


def split_data(x, y, test_size = 0.2, random_state = None):

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    return x_train, x_test, y_train, y_test


def get_cv_folds(n_splits = 5, random_state = None):
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )