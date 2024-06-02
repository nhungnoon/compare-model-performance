from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


def simple_lasso():
  return Lasso(random_state = 0)

def grid_search_lasso():
  model = Lasso(random_state = 0)
  grid_values = {'alpha': [0.1, 1, 10, 100],
              'max_iter': [100, 50],
               'tol': [0.1, 0.05],
              }

  # default metric to optimize over grid parameters: accuracy
  grid_search_model = GridSearchCV(model, param_grid = grid_values)
  return grid_search_model

def grid_search_ridge():
  model = Ridge(random_state=0)
  grid_values = {'alpha': [0.1, 1, 10, 100],
              'max_iter': [100, 50],
               'tol': [0.1, 0.05],
              }

  # default metric to optimize over grid parameters: accuracy
  grid_search_model = GridSearchCV(model, param_grid = grid_values)
  return grid_search_model
