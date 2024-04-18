from sklearn.linear_model import LogisticRegression

def lr_model_for_nlp():
  return LogisticRegression(C= 0.8, class_weight="balanced")
