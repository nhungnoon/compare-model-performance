from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier


def lr_model_for_nlp():
    return LogisticRegression(C=0.8, class_weight="balanced")


def dt_classifier():
    dt_classifier = DecisionTreeClassifier(
        random_state=0,
        min_samples_split=5,
        class_weight="balanced",
    )
    return dt_classifier


def multinominal_nb():
    return MultinomialNB(alpha=0.8)


def rf_classifier():
    rf_classified = RandomForestClassifier(
        max_depth=3, min_samples_split=5, class_weight="balanced", random_state=16
    )
    return rf_classified
