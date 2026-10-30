import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import CSVCreate, graphics, viewer, training_data, models
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

RANDOM_STATE = None
TEST_SIZE = 0.2
TARGET = "class"
ARCHIVE_PATH = "./datasets/Page_Blocks"
DATASET_NAME = "/page_blocks_classification"
CSV_PATH  = ARCHIVE_PATH + DATASET_NAME + ".csv"
XLSX_PATH = ARCHIVE_PATH + DATASET_NAME + ".xlsx"
ARFF_PATH = ARCHIVE_PATH + DATASET_NAME + ".arff"

def base_datas(dataframe):
    print("\nРазмер датасета:", dataframe.shape)

    print("\nПервые 5 строк:")
    print(dataframe.head().to_string())

    print("\nТипы признаков:")
    print(dataframe.dtypes)

    print("\nОписательные статистики:")
    print(dataframe.describe().T.to_string())

    print("\nПропущенных значений:", dataframe.isna().sum(), sep="\n")

    print("\nРаспределение классов:")
    print(dataframe[TARGET].value_counts())

def show_graphics(dataframe, x, y, target, show_boxplots = True, graphics_name = "full"):

    plots = {
        f"corr {graphics_name}": lambda ax: graphics.correlation_graphic(ax, x),
        f"pca {graphics_name}": lambda ax: graphics.pca_graphic(ax, x, y, class_names, 0, random_state=RANDOM_STATE),
        f"anova {graphics_name}": lambda ax: graphics.anova_graphic(ax, x, y),
        f"mi {graphics_name}": lambda ax: graphics.mi_graphic(ax, x, y, random_state=RANDOM_STATE),
        "class balance": lambda ax: graphics.class_balance_graphic(ax, dataframe, target),
    }

    viewer.add_viewer(plots)

    if show_boxplots:
        boxplots = dict()
        for feature in map(str, dataframe.drop(columns=[target]).columns):
            boxplots[feature] = lambda ax, f=feature: graphics.boxplot_single(ax, dataframe, f, target)
        viewer.add_viewer(boxplots)

def decision_tree_training_test(x, y):
    x_train, x_test, y_train, y_test = training_data.split_data(x, y, TEST_SIZE, random_state=RANDOM_STATE)

    cv = training_data.get_cv_folds(random_state=RANDOM_STATE)
    model, gs = models.train_decision_tree(x_train, y_train, cv, random_state=RANDOM_STATE)

    y_pred = model.predict(x_test)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision_macro': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_test, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'precision_micro': precision_score(y_test, y_pred, average='micro', zero_division=0),
        'recall_micro': recall_score(y_test, y_pred, average='micro', zero_division=0),
        'f1_micro': f1_score(y_test, y_pred, average='micro', zero_division=0)
    }
    print("\nЛучшие параметры:", gs.best_params_)
    print("CV F1-macro:", gs.best_score_)
    print("Test метрики:", metrics)

    cv_results = pd.DataFrame(gs.cv_results_)
    top_10 = cv_results.nlargest(10, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]
    print(top_10.to_string())



if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        CSVCreate.make_csv(XLSX_PATH, CSV_PATH, ARFF_PATH)
    df = CSVCreate.load_data(CSV_PATH)
    le = LabelEncoder()

    x = df.drop(columns=[TARGET])
    y = le.fit_transform(df[TARGET])
    class_names = list(le.classes_)

    print("Датасет")
    base_datas(df)

    # DROP = ["Perimeter", "MajorAxisLength", "MinorAxisLength", "ConvexArea", "EquivDiameter",
    #         "Eccentricity", "ShapeFactor2"]
    # DROP = ["Perimeter", "MajorAxisLength", "MinorAxisLength", "ConvexArea", "EquivDiameter",
    #         "AspectRation", "Eccentricity", "ShapeFactor2", "ShapeFactor3"]
    # DROP += ["Solidity"]
    # DROP += ["ShapeFactor1"]
    # df_uncorr = df.drop(columns=DROP)
    # x_uncorr = df_uncorr.drop(columns=[TARGET])
    # y_uncorr = le.fit_transform(df_uncorr[TARGET])

    DROP = ["area", "blackand", "wb_trans"]
    # DROP += ["eccen"]
    DROP += ["blackpix"]
    df_uncorr = df.drop(columns=DROP)
    x_uncorr = df_uncorr.drop(columns=[TARGET])
    y_uncorr = le.fit_transform(df_uncorr[TARGET])

    print("\n\n\nДатасет без сильно коррелированных признаков")
    base_datas(df_uncorr)

    decision_tree_training_test(x, y)
    decision_tree_training_test(x_uncorr, y_uncorr)

    # df_uncorr = df_uncorr.drop(columns=["Solidity"])
    # x_uncorr = df_uncorr.drop(columns=[TARGET])
    # y_uncorr = le.fit_transform(df_uncorr[TARGET])
    # decision_tree_training_test(x_uncorr, y_uncorr)
    #
    # df_uncorr = df_uncorr.drop(columns=["ShapeFactor1"])
    # x_uncorr = df_uncorr.drop(columns=[TARGET])
    # y_uncorr = le.fit_transform(df_uncorr[TARGET])
    # decision_tree_training_test(x_uncorr, y_uncorr)

    # show_graphics(df, x, y, TARGET, False)
    # show_graphics(df_uncorr, x_uncorr, y_uncorr, TARGET, graphics_name="uncorr")
    #
    # plt.show()


