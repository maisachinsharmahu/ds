import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

d = load_breast_cancer(as_frame=True)
X, y = d.data, d.target

print(
    "breast cancer data:",
    X.shape,
    "| benign",
    int(y.sum()),
    "| malignant",
    int((1 - y).sum()),
)

Z = StandardScaler().fit_transform(X)
pca = PCA().fit(Z)
cum = np.cumsum(pca.explained_variance_ratio_)

print(
    "explained variance of PC1..PC5:",
    np.round(pca.explained_variance_ratio_[:5] * 100, 2),
)
print("cumulative:", np.round(cum[:6] * 100, 1))

for t in (0.80, 0.90, 0.95, 0.99):
    print(
        f"components for {t * 100:.0f} %: {int(np.searchsorted(cum, t) + 1)}"
    )

print(
    "sum of all explained shares:",
    round(pca.explained_variance_ratio_.sum(), 4),
    "| eigenvalues above 1 (Kaiser):",
    int((pca.explained_variance_ > 1).sum()),
)

L = pd.DataFrame(
    pca.components_[:2].T, index=X.columns, columns=["PC1", "PC2"]
)

print(
    "\nstrongest loadings on PC1:",
    L.PC1.abs().sort_values(ascending=False).head(3).index.tolist(),
)
print(
    "strongest loadings on PC2:",
    L.PC2.abs().sort_values(ascending=False).head(3).index.tolist(),
)

S = pca.transform(Z)

print(
    "correlation between PC1 and PC2 scores:",
    round(np.corrcoef(S[:, 0], S[:, 1])[0, 1], 6),
)
print(
    "mean PC1 of malignant",
    round(S[y == 0, 0].mean(), 2),
    "| of benign",
    round(S[y == 1, 0].mean(), 2),
)

print("\naccuracy of logistic regression (5-fold)")

for k in (1, 2, 5, 10, 30):
    Zk = PCA(n_components=k).fit_transform(Z)
    print(
        f"  with {k:>2} components: {cross_val_score(LogisticRegression(max_iter=2000), Zk, y, cv=5).mean() * 100:.2f} %"
    )
Zi = PCA(n_components=0.95).fit(Z)

print(
    "keeping 95 % variance needs",
    Zi.n_components_,
    "components; inverse transform error",
    round(
        float(
            np.mean((Z - Zi.inverse_transform(Zi.transform(Z))) ** 2)
        ),
        4,
    ),
)

fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
ax[0].plot(range(1, 31), pca.explained_variance_, "ko-", ms=3)
ax[0].axhline(1, color="grey", ls="--")
ax[0].set_title("Scree plot (eigenvalues)")
ax[1].scatter(
    S[y == 1, 0],
    S[y == 1, 1],
    s=12,
    facecolors="none",
    edgecolors="grey",
    label="benign",
)
ax[1].scatter(
    S[y == 0, 0],
    S[y == 0, 1],
    s=12,
    marker="x",
    color="black",
    label="malignant",
)
ax[1].legend()
ax[1].set_title("First two components")
ax[2].plot(range(1, 31), cum * 100, "ko-", ms=3)
ax[2].axhline(95, color="grey", ls="--")
ax[2].set_title("Cumulative variance (%)")
plt.tight_layout()
plt.show()
