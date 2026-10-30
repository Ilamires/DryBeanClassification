from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV


def train_decision_tree(x_train, y_train, cv, random_state=None):
    param_grid = {
        'max_depth': [3, 5, 7, 10, 15, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 5, 10],
        'criterion': ['gini'],
    }

    gs = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=random_state),
        param_grid=param_grid,
        cv=cv,
        scoring='f1_macro',
        n_jobs=-1
    )
    gs.fit(x_train, y_train)

    best_model = gs.best_estimator_
    best_model.fit(x_train, y_train)

    return best_model, gs