import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.decomposition import PCA

RANDOM_STATE = 42

def correlation_graphic(ax, x):
    sns.heatmap(x.corr(), cmap="coolwarm", center=0.0,
                annot=True, fmt=".2f", annot_kws={"size": 7},
                square=True, ax=ax)
    ax.set_title("Корреляция между признаками")


def pca_graphic(ax, x, y, class_names, first = 0, second = 1):
    if first < 0 or second < 0:
        return
    if first > second:
        first, second = second, first
    x_scaled = StandardScaler().fit_transform(x)
    pca = PCA(n_components=second + 1, random_state=RANDOM_STATE)
    x_pca = pca.fit_transform(x_scaled)

    scatter = ax.scatter(x_pca[:, first], x_pca[:, second], c=y, cmap="tab10", s=12, alpha=0.6)
    handles, _ = scatter.legend_elements()
    ax.legend(handles, class_names, title="Class",
              loc="lower right", fontsize=8)
    ax.set_xlabel(f"PC{first + 1} ({pca.explained_variance_ratio_[first]:.1%})")
    ax.set_ylabel(f"PC{second + 1} ({pca.explained_variance_ratio_[second]:.1%})")
    ax.set_title("PCA: проекция данных на 2 главные компоненты")

def f_scores(x, y):
    f, _ = f_classif(x, y)
    return pd.Series(f, index=x.columns).sort_values(ascending=True)

def anova_graphic(ax, x, y):
    scores = f_scores(x, y)
    scores.plot.barh(ax=ax, color="steelblue")
    ax.set_title("ANOVA F-statistic по признакам")
    ax.set_xlabel("F-statistic")

def class_balance_graphic(ax, df, target="Class"):
    order = df[target].value_counts().index
    sns.countplot(data=df, x=target, hue=target,
                  order=order, palette="viridis",
                  legend=False, ax=ax)
    ax.set_title("Распределение классов")
    ax.set_xlabel("Класс")
    ax.set_ylabel("Объектов")

def mi_graphic(ax, x, y):
    mi = pd.Series(mutual_info_classif(x, y, random_state=RANDOM_STATE),
                   index=x.columns).sort_values(ascending=True)
    mi.plot.barh(ax=ax, color="darkorange")
    ax.set_title("Mutual Information по признакам")
    ax.set_xlabel("MI (bits)")

def boxplot_single(ax, df, feature, target="Class"):
    sns.boxplot(data=df, x=target, y=feature, hue=target,
                palette="Set2", legend=False, ax=ax)
    ax.set_title(f"Распределение '{feature}' по классам")
    ax.set_ylabel(feature)