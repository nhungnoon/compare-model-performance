"""
Explore sklearn data set wine
@article{scikit-learn, 
title={Scikit-learn: Machine Learning in {P}ython}, 
author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V. and Thirion, B. 
and Grisel, O. and Blondel, M. and Prettenhofer, P. and Weiss, R. and Dubourg, V. and Vanderplas, J. 
and Passos, A. and Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.}, 
journal={Journal of Machine Learning Research}, volume={12}, pages={2825--2830}, year={2011} }
"""

from sklearn.decomposition import PCA
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn import tree
import matplotlib.pyplot as plt


def generate_train_test_wine_dataset():
    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html

    ### Explore sklearn dataset digits
    # load in data
    wine_df = df = datasets.load_wine(as_frame=True)

    # Split into train/test
    X = wine_df["data"]
    y = wine_df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )

    return X_train, X_test, y_train, y_test


# Explore feature importance


def return_ranking_features(features_names, X_train, y_train, X_test, y_test):
    """
    return
    (1) features ranking based on trained DecisionTreeClassifier
    (2) features importance based on permutance importance
    """
    rfe = RFE(
        estimator=DecisionTreeClassifier(random_state=0), n_features_to_select=5, step=1
    )
    rfe.fit(X_train, y_train)
    ranking = rfe.ranking_
    print("rfe features ranking")
    for name, rank in zip(features_names, ranking):
        print(f"{name}: {rank}")

    permutation_importance_fit = permutation_importance(
        rfe, X_test, y_test, n_repeats=20, random_state=0
    )
    print("features permutation importance")
    for name, features_importance in zip(
        features_names, permutation_importance_fit.importances_mean
    ):
        print(f"{name}: {features_importance}")


def loop_through_parameters(
    operation_steps,
    parameters_experiment,
    X_train,
    y_train,
):
    score_param_estimators_list = {}

    for name in operation_steps.keys():

        gs = GridSearchCV(
            estimator=operation_steps[name],
            param_grid=parameters_experiment[name],
            scoring="accuracy",
            n_jobs=-1,
            cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=0),
        )

        gs = gs.fit(X_train, y_train)

        score_param_estimators_list.append(
            [gs.best_score_, gs.best_params_, gs.best_estimator_]
        )

    return score_param_estimators_list


def gridsearch_best_model(X_train, y_train):
    """
    # Create a pipeline to tune parameters
    # based on
    # https://github.com/yuxiaohuang/teaching/blob/main/machine_learning_I/fall_2020/code/p2_shallow_learning/p2_c2_supervised_learning/p2_c2_s5_tree_based_models/code_example/code_example.ipynb
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pipeline_with_multiple_classifier = {
        "logistic_regression": LogisticRegression(random_state=42),
        "kneighbors": KNeighborsClassifier(),
        "dtree": DecisionTreeClassifier(random_state=45),
        "kmeans": KMeans(random_state=0),
    }

    operation_steps = {}

    for name, clf in pipeline_with_multiple_classifier.items():
        operation_steps[name] = Pipeline(
            [("StandardScaler", StandardScaler()), ("clf", clf)]
        )

    parameters_experiment = {}

    params = [
        {
            "clf__multi_class": ["ovr"],
            "clf__solver": ["newton-cg", "sag", "saga"],
            "clf__C": [10**i for i in range(-2, 1)],
        },
        {
            "clf__multi_class": ["multinomial"],
            "clf__solver": ["newton-cg", "lbfgs", "saga"],
            "clf__C": [10**i for i in range(-1, 2)],
        },
    ]

    parameters_experiment["logistic_regression"] = params

    parameters_experiment["kneighbors"] = [{"clf__n_neighbors": list(range(1, 3))}]
    parameters_experiment["kmeans"] = [{"clf__n_clusters": list(range(1, 3))}]

    parameters_experiment["dtree"] = [
        {"clf__min_samples_split": [2, 10, 30], "clf__min_samples_leaf": [1, 10, 20]}
    ]

    best_score = loop_through_parameters(
        operation_steps,
        parameters_experiment,
        X_train,
        y_train,
    )

    return best_score


def generate_train_test_digits_dataset():
    # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html

    ### Explore sklearn dataset digits
    digits_df = datasets.load_digits(as_frame=True)

    X = digits_df["data"]
    y = digits_df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, stratify=y, random_state=42
    )

    return X_train, X_test, y_train, y_test


def grid_search_digits_data(X_train, y_train):
    """
    # Create a pipeline to tune parameters
    # based on
    # https://github.com/yuxiaohuang/teaching/blob/main/machine_learning_I/fall_2020/code/p2_shallow_learning/p2_c2_supervised_learning/p2_c2_s5_tree_based_models/code_example/code_example.ipynb
    """
    # Create a pipeline to tune parameters

    pipeline_with_multiple_classifier = {
        "dtree": DecisionTreeClassifier(random_state=45),
    }

    operation_steps = {}

    for name, clf in pipeline_with_multiple_classifier.items():
        operation_steps[name] = Pipeline([("clf", clf)])
    parameters_experiment = {}
    parameters_experiment["kneighbors"] = [{"clf__n_neighbors": list(range(1, 3))}]
    parameters_experiment["dtree"] = [
        {"clf__min_samples_split": [2, 10, 30], "clf__min_samples_leaf": [1, 10, 20]}
    ]

    best_estimators = loop_through_parameters(
        operation_steps,
        parameters_experiment,
        X_train,
        y_train,
    )

    return best_estimators


def visualize_dt_digits(best_estimators, input_data, max_depth):
    plt.figure(figsize=(40, 20))
    _ = tree.plot_tree(
        best_estimators[0][2][0],
        feature_names=input_data["data"].columns.tolist(),
        filled=True,
        fontsize=6,
        rounded=True,
        max_depth=max_depth,
    )
    plt.show()
