import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import CSVCreate
import graphics
import os
import viewer
import training_data

from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 42
ARCHIVE_PATH = "./archive/Dry_Bean_Dataset/"
CSV_PATH  = ARCHIVE_PATH + "Dry_Bean_Dataset.csv"
XLSX_PATH = ARCHIVE_PATH + "Dry_Bean_Dataset.xlsx"
ARFF_PATH = ARCHIVE_PATH + "Dry_Bean_Dataset.arff"

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
    print(dataframe["Class"].value_counts())

def show_graphics():

    plots = {
        "corr full": lambda ax: graphics.correlation_graphic(ax, x),
        "pca full": lambda ax: graphics.pca_graphic(ax, x, y, class_names, 0, random_state=RANDOM_STATE),
        "anova full": lambda ax: graphics.anova_graphic(ax, x, y),
        "mi full": lambda ax: graphics.mi_graphic(ax, x, y, random_state=RANDOM_STATE),
        "class balance": lambda ax: graphics.class_balance_graphic(ax, df),
        "corr uncorr": lambda ax: graphics.correlation_graphic(ax, x_uncorr),
        "pca uncorr": lambda ax: graphics.pca_graphic(ax, x_uncorr, y_uncorr, class_names, 0,
                                                      random_state=RANDOM_STATE),
        "anova uncorr": lambda ax: graphics.anova_graphic(ax, x_uncorr, y_uncorr),
        "mi uncorr": lambda ax: graphics.mi_graphic(ax, x_uncorr, y_uncorr, random_state=RANDOM_STATE),
    }
    boxplots = {
        "area": lambda ax: graphics.boxplot_single(ax, df, "Area"),
        "extent": lambda ax: graphics.boxplot_single(ax, df, "Extent"),
        "solidity": lambda ax: graphics.boxplot_single(ax, df, "Solidity"),
        "roundness": lambda ax: graphics.boxplot_single(ax, df, "roundness"),
        "compactness": lambda ax: graphics.boxplot_single(ax, df, "Compactness"),
        "shapefactor1": lambda ax: graphics.boxplot_single(ax, df, "ShapeFactor1"),
        "shapefactor4": lambda ax: graphics.boxplot_single(ax, df, "ShapeFactor4"),
    }

    viewer.add_viewer(plots)
    viewer.add_viewer(boxplots)

    plt.show()

if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        CSVCreate.make_csv(XLSX_PATH, CSV_PATH, ARFF_PATH)
    df = CSVCreate.load_data(CSV_PATH)
    le = LabelEncoder()

    x = df.drop(columns=["Class"])
    y = le.fit_transform(df["Class"])
    class_names = list(le.classes_)

    print("Датасет")
    base_datas(df)

    # DROP = ["Perimeter", "MajorAxisLength", "MinorAxisLength", "ConvexArea", "EquivDiameter",
    #         "Eccentricity", "ShapeFactor2"]
    DROP = ["Perimeter", "MajorAxisLength", "MinorAxisLength", "ConvexArea", "EquivDiameter",
            "AspectRation", "Eccentricity", "ShapeFactor2", "ShapeFactor3"]
    # DROP += ["Solidity"]
    # DROP += ["ShapeFactor1"]
    df_uncorr = df.drop(columns=DROP)
    x_uncorr = df_uncorr.drop(columns=["Class"])
    y_uncorr = le.fit_transform(df_uncorr["Class"])
    class_names = list(le.classes_)

    print("\n\n\nДатасет без сильно коррелированных признаков")
    base_datas(df_uncorr)

    show_graphics()

    x_train, x_test, y_train, y_test = training_data.split_data(x, y, class_names)


