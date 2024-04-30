from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


"""
Based on
https://scikit-learn.org/stable/auto_examples/neighbors/plot_nca_dim_reduction.html#sphx-glr-auto-examples-neighbors-plot-nca-dim-reduction-py
https://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html#sphx-glr-auto-examples-decomposition-plot-pca-iris-py
"""

# Load Digits dataset
X, y = datasets.load_digits(return_X_y=True)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, stratify=y, random_state=42
)


pca = make_pipeline(StandardScaler(), PCA(n_components=3, random_state=0))
pca.fit(X_train)


x_transformed = pca.transform(X_train)
x_test_transformed = pca.transform(X_test)

fig = plt.figure(1, figsize=(10, 12))
ax1 = fig.add_subplot(121, projection="3d", elev=40, azim=130)
surf1 = ax1.scatter(
    x_transformed[:, 0],
    x_transformed[:, 1],
    x_transformed[:, 2],
    c=y_train,
    s=5,
    cmap="tab10",
)
ax1.xaxis.set_ticklabels([])
ax1.yaxis.set_ticklabels([])
ax1.zaxis.set_ticklabels([])
plt.title("x_train")
ax2 = fig.add_subplot(122, projection="3d", elev=40, azim=130)
surf2 = ax2.scatter(
    x_test_transformed[:, 0],
    x_test_transformed[:, 1],
    x_test_transformed[:, 2],
    c=y_test,
    s=5,
    cmap="tab10",
)
ax2.xaxis.set_ticklabels([])
ax2.yaxis.set_ticklabels([])
ax2.zaxis.set_ticklabels([])
plt.title("x_test")
