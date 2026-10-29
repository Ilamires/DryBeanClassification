import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold


def split_data(x, y, class_names, test_size = 0.2, random_state = None):

    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    print("\nБаланс классов в обучающей выборке")
    for name, count in zip(*np.unique(y_train, return_counts=True)):
        print(f"  {class_names[name]}: {count} ({100 * count / len(y_train):.1f}%)")
    print(f"\nБаланс классов в тестовой выборке")
    for name, count in zip(*np.unique(y_test, return_counts=True)):
        print(f"{class_names[name]}: {count} ({100 * count / len(y_test):.1f}%)")

    return x_train, x_test, y_train, y_test


def get_cv_folds(n_splits = 5, random_state = 42):
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )